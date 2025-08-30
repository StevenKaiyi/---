import pickle
import os
import re
import jieba
import numpy as np  
from urllib.request import urljoin
from bs4 import BeautifulSoup
import requests
import time
from url_normalize import url_normalize
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, Counter  

documents = [] #存储文档id，url，正文。格式：{"doc_id": id, "url": url, "content": 文本}
doc_id_counter = 0  #文档ID计数器
inverted_index = defaultdict(list)  #存储 (docid, tf) 元组，格式： {"人工智能": [(0,2), (2,5)]}
doc_tfidf = defaultdict(dict)  # 存储所有文档的TF-IDF权重，格式 {docid: {word: tfidf_weight}} 一张大表
doc_length = []  # 存储文档向量长度（归一化用），doc_length[docid] 
total_docs = 0  # 总文档数

# 线程锁
lock = threading.Lock()
doc_lock = threading.Lock()
index_lock = threading.Lock()

def load_stopwords(filename='中文停用词表.txt'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f.readlines()]
        return set(stopwords)
    except FileNotFoundError:
        print("警告：停用词表文件未找到，将不进行停用词过滤")
        return set()
    

def save_state(queue, all_urlset, used_urlset, count, filename="crawler_state.pkl"):
    global inverted_index, doc_tfidf, doc_length, total_docs
    state = {
        'queue': queue,
        'all_urlset': all_urlset,
        'used_urlset': used_urlset,
        'count': count,
        'index': inverted_index, 
        'documents': documents,
        'doc_id_counter': doc_id_counter,
        'doc_tfidf': doc_tfidf, 
        'doc_length': doc_length,  
        'total_docs': total_docs 
    }
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    print(f"状态（含TF-IDF/向量）已保存到 {filename}")


# 2. 加载状态
def load_state(filename="crawler_state.pkl"):
    global documents, doc_id_counter, inverted_index, doc_tfidf, doc_length, total_docs
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        documents = state['documents']
        doc_id_counter = state['doc_id_counter']
        inverted_index = state['index']
        doc_tfidf = state['doc_tfidf']
        doc_length = state['doc_length']
        total_docs = state['total_docs']
        print(f"已加载状态：总文档数={total_docs}，索引词数={len(inverted_index)}")
        return state
    except Exception as e:
        print(f"加载失败：{str(e)}，将从头开始")
        return None


# 3. 提取文本
def extract_text(html_doc):
    soup = BeautifulSoup(html_doc, 'lxml')
    raw_text = soup.get_text()
    text_with_space = raw_text.replace('\t', ' ').replace('\n', ' ')
    clean_text = re.sub(r'\s+', ' ', text_with_space) 
    return clean_text


# 4. 处理单个网页（存储docid，url，html文本）
def crawl_one(url, headers, stopwords):
    global doc_id_counter, documents
    html_doc = get_html(url, headers=headers)
    if html_doc is None:
        with lock:
            used_urlset.add(url)
        return None 

    text_content = extract_text(html_doc)
    if len(text_content) < 2:
        print(f"文档 {url} 文本过短跳过")
        with lock:
            used_urlset.add(url)
        return None

    with doc_lock:
        doc_id = doc_id_counter
        documents.append({
            "doc_id": doc_id,
            "url": url,
            "content": text_content
        })
        doc_id_counter += 1

    url_sets = crawl_all_urls(html_doc, url)
    print(f"从 {url} 找到 {len(url_sets)} 个链接（文档ID：{doc_id}）")
    with lock:
        for new_url in url_sets:
            if new_url not in all_urlset and 'https://xsc' in new_url and '.htm' in new_url:
                queue.append(new_url)
                all_urlset.add(new_url)

    with lock:
        used_urlset.add(url)

    #统计当前文档的词频（返回给主函数）
    words = jieba.lcut(text_content)
    filtered_words = [w for w in words if w not in stopwords and len(w) >= 2]
    word_freq = Counter(filtered_words)  # 统计词频：{word: tf}
    return doc_id, word_freq  # 返回（单个文档的ID，词频字典）

def crawl_all_urls(html_doc, url):
    all_links = set()
    soup = BeautifulSoup(html_doc, 'html.parser')
    for anchor in soup.find_all('a'):
        href = anchor.attrs.get("href")
        if href and href != "":
            if not href.startswith('http'):
                href = urljoin(url, href)
            all_links.add(url_normalize(href))
    return all_links


def get_html(uri, headers={}, timeout=10):
    try:
        r = requests.get(uri, headers=headers, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding 
        return r.text
    except Exception as e:
        print(f"获取 {uri} 失败: {str(e)}")
        return None

# 5. 构建倒排索引（包含词频统计+DF计算）
def build_inverted_index(stopwords):
    global inverted_index, total_docs
    total_docs = len(documents)  
    inverted_index.clear()  # 清除旧索引

    # 遍历所有文档，构建“词-（文档ID，词频）”倒排索引字典
    for doc in documents:
        doc_id = doc["doc_id"]
        content = doc["content"]
        words = jieba.lcut(content)
        filtered_words = [w for w in words if w not in stopwords and len(w) >= 2]
        word_freq = Counter(filtered_words)  # 统计当前文档词频

        with index_lock:
            for word, tf in word_freq.items():
                inverted_index[word].append((doc_id, tf))  # 以元组形式存储 (docid, tf)

    # 去重同一文档的重复词（避免文档内同一词被记录到多过元组内）
    with index_lock:
        for word in inverted_index:
            doc_set = set()
            unique_postings = []
            for (docid, tf) in inverted_index[word]:
                if docid not in doc_set:
                    doc_set.add(docid)
                    unique_postings.append((docid, tf))
            inverted_index[word] = unique_postings  # 去重后保留一个（docid, tf）

    print(f"倒排索引构建完成：{len(inverted_index)} 个词，总文档数：{total_docs}")


# 6. 计算TF-IDF权重（对应PPT TF-IDF公式：(1+logTF)×log(N/DF)）
def compute_tfidf():
    global doc_tfidf, doc_length, total_docs
    doc_tfidf.clear()
    doc_length.clear()

    # 计算所有词的IDF（DF=倒排索引中，一个词在多少个文档中出现，N=总文档数）
    idf_dict = {}
    for word in inverted_index:
        df = len(inverted_index[word])  
        if df == 0:
            idf_dict[word] = 0.0
        else:
            idf_dict[word] = np.log10(total_docs / df)  

    # 遍历每个文档，计算TF-IDF权重和向量长度
    for doc in documents:
        doc_id = doc["doc_id"]
        content = doc["content"]
        words = jieba.lcut(content)
        filtered_words = [w for w in words if w not in stopwords and len(w) >= 2]
        word_freq = Counter(filtered_words)

        tfidf_vec = {}
        for word, tf in word_freq.items():
            if word not in idf_dict:
                continue
            # PPT的TF加权：1+log10(tf)（tf>0时）
            tf_weight = 1 + np.log10(tf) if tf > 0 else 0.0
            # PPT的TF-IDF公式：TF权重 × IDF权重
            tfidf_vec[word] = tf_weight * idf_dict[word]

        # 保存当前文档的TF-IDF权重
        doc_tfidf[doc_id] = tfidf_vec

        # 计算文档向量长度（归一化）
        if not tfidf_vec:
            doc_length.append(0.0)
        else:
            vec_values = np.array(list(tfidf_vec.values()))
            vec_norm = np.sqrt(np.sum(vec_values ** 2)) 
            doc_length.append(vec_norm)

    print(f"TF-IDF计算完成：{len(doc_tfidf)} 个文档，向量长度已保存")

def cosine_search(query, top_k=10):
    global inverted_index, doc_tfidf, doc_length, total_docs, stopwords
    #处理查询：分词，过滤，统计查询词频
    query_words = jieba.lcut(query)
    filtered_q_words = [w for w in query_words if w not in stopwords and len(w) >= 2]
    if not filtered_q_words:
        print("无效查询")
        return []
    q_word_freq = Counter(filtered_q_words)  # 查询词频：{word: tf}

    # 计算查询的TF-IDF权重（与文档向量格式一致）
    q_tfidf = {}
    for word in q_word_freq:
        if word not in inverted_index:
            continue  # 查询词不在索引中，跳过，避免报错
        # 计算查询词的TF权重
        q_tf = q_word_freq[word]
        q_tf_weight = 1 + np.log10(q_tf) if q_tf > 0 else 0.0
        # 计算查询词的IDF
        df = len(inverted_index[word])
        q_idf = np.log10(total_docs / df) if df > 0 else 0.0
        q_tfidf[word] = q_tf_weight * q_idf

    
    q_vec_values = np.array(list(q_tfidf.values()))
    # 计算查询向量长度
    q_vec_norm = np.sqrt(np.sum(q_vec_values ** 2)) if len(q_vec_values) > 0 else 1.0

    # 评分
    doc_scores = defaultdict(float)  # 格式{docid: 得分}
    for word, q_weight in q_tfidf.items():
        # 获取该词的倒排索引（(docid, tf)列表）
        postings = inverted_index.get(word, [])
        for (docid, _) in postings:
            # 累加得分：文档TF-IDF × 查询TF-IDF（向量内积的一部分）
            d_weight = doc_tfidf[docid].get(word, 0.0)
            doc_scores[docid] += d_weight * q_weight

    # 归一化得分
    sorted_docs = []
    for docid, raw_score in doc_scores.items():
        if docid >= len(doc_length) or doc_length[docid] == 0.0:
            continue  # 文档长度为0，跳过（无有效内容）
        # 余弦相似度公式：cosθ = 内积/(|len_q|×|len_d|)
        cos_score = raw_score / (q_vec_norm * doc_length[docid])
        # 获取文档信息
        doc_info = next((d for d in documents if d["doc_id"] == docid), None)
        if doc_info:
            # 内容预览（前150字）
            preview = doc_info["content"][:150] + "..." if len(doc_info["content"]) > 150 else doc_info["content"]
            sorted_docs.append((cos_score, docid, doc_info["url"], preview))

    # 6. 按相似度得分降序排序，返回Top K
    sorted_docs.sort(reverse=True, key=lambda x: x[0])  # 得分越高越靠前
    return sorted_docs[:top_k]  # 返回前K个结果

def search_interface():
    global inverted_index, total_docs
    state_file = "crawler_state.pkl"
    state = load_state(state_file)
    if not state or total_docs == 0:
        print("无爬取数据，请先运行爬虫")
        return


    while True:
        query = input("\n请输入查询关键词：").strip()
        if query.lower() == 'q':
            print("退出搜索")
            break
        if not query:
            print("请输入有效关键词")
            continue

        # 输入返回结果数量（默认10）
        top_k_input = input("请输入返回结果数量（默认10）：").strip()
        top_k = int(top_k_input) if top_k_input.isdigit() else 10

        # 执行相关性搜索
        results = cosine_search(query, top_k)

        # 展示结果
        if not results:
            print(f"未找到与「{query}」相关的文档")
            continue
        print(f"\n找到 {len(results)} 个相关文档：")
        for i, (score, docid, url, preview) in enumerate(results, 1):
            print(f"\n{i}. 相似度得分：{score:.4f} | 文档ID：{docid}")
            print(f"   URL：{url}")
            print(f"   内容预览：{preview}")

#爬虫主程序
def main_crawl():
    input_urls = ['https://xsc.ruc.edu.cn/']
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    wait_time = 0.3
    max_count = 10000  
    max_workers = 5  
    global queue, all_urlset, used_urlset, stopwords

    # 初始化/恢复状态
    resume = input("是否从上次状态继续？(y/n): ").lower().startswith('y')
    state_file = "crawler_state.pkl"
    if resume:
        state = load_state(state_file)
        if state:
            queue = state['queue']
            all_urlset = state['all_urlset']
            used_urlset = state['used_urlset']
            count = state['count']
        else:
            queue = []
            all_urlset = set()
            used_urlset = set()
            count = 0
            for url in input_urls:
                if url not in all_urlset:
                    queue.append(url)
                    all_urlset.add(url)
    else:
        # 重置所有状态
        global documents, doc_id_counter, doc_tfidf, doc_length, total_docs
        documents = []
        doc_id_counter = 0
        doc_tfidf = defaultdict(dict)
        doc_length = []
        total_docs = 0
        queue = []
        all_urlset = set()
        used_urlset = set()
        count = 0
        #将种子url加入队列和all_url
        for url in input_urls:
            if url not in all_urlset:
                queue.append(url)
                all_urlset.add(url)

    # 加载停用词
    stopwords = load_stopwords()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while len(queue) > 0 and count < max_count:
                batch = []
                with lock:
                    while queue and len(batch) < max_workers:
                        batch.append(queue.pop(0))
                        count += 1
                if not batch:
                    break

                # 进行爬取（返回（docid, 词频））
                futures = [executor.submit(crawl_one, url, headers, stopwords) for url in batch]
                for f in as_completed(futures):
                    pass  # 爬取结果已在crawl_one中处理，此处仅等待完成

                # 打印进度
                print(f"已爬取 {count} 页 | 队列剩余 {len(queue)} 页 | 总链接数 {len(all_urlset)}")

                # 定期询问是否继续
                if count % 10 == 0 and count < max_count and len(queue) > 0:
                    continue_crawl = input("是否继续？(y/n，默认y): ").lower()
                    if continue_crawl.startswith('n'):
                        build_inverted_index(stopwords)  # 先更新索引
                        compute_tfidf()  # 再计算TF-IDF
                        save_state(queue, all_urlset, used_urlset, count, state_file)
                        print("已暂停，状态保存完成")
                        return

                time.sleep(wait_time)

        # 爬取完成：构建索引+计算TF-IDF+保存状态
        build_inverted_index(stopwords)
        compute_tfidf()
        save_state(queue, all_urlset, used_urlset, count, state_file)
        print(f"\n爬取完成：")
        print(f"- 总爬取页面：{len(used_urlset)}")
        print(f"- 有效文档数：{total_docs}")
        print(f"- 倒排索引词数：{len(inverted_index)}")
        print(f"- TF-IDF向量已计算，支持相关性排序")

    except KeyboardInterrupt:
        build_inverted_index(stopwords)
        compute_tfidf()
        save_state(queue, all_urlset, used_urlset, count, state_file)
        print("\n手动中断，状态已保存")

#根据文档id查询文档内容
def get_document_via_id(target_id):
    """根据文档ID查询URL和内容，处理ID不存在的情况"""
    for doc in documents:
        if doc["doc_id"] == target_id:
            return (doc["url"], doc["content"])
    return (None, f"文档ID {target_id} 不存在（当前最大ID：{len(documents)-1}）")

#主程序
def main():
    main_crawl()  
    search_interface()  


main()
