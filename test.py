import pickle
import os
import re
import jieba
import jieba.posseg as pseg
import numpy as np  
from urllib.request import urljoin
from bs4 import BeautifulSoup
import requests
import time
from url_normalize import url_normalize
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, Counter  

# -------------------------- 全局变量（修复BM25相关变量） --------------------------
documents = []  # 存储文档：{"doc_id": id, "url": url, "content": 文本, "word_freq": 词频}
doc_id_counter = 0  # 文档唯一ID计数器
inverted_index = defaultdict(list)  # 改进倒排索引：{词: [(docid, tf), ...]}
doc_bm25 = defaultdict(dict)  # BM25权重（替代原doc_tfidf）
doc_length = []  # BM25向量模长（用于余弦归一化）
total_docs = 0  # 总文档数
idf_dict = {}  # 缓存IDF值（BM25共用）
# 线程锁
lock = threading.Lock()
doc_lock = threading.Lock()
index_lock = threading.Lock()
input_url = ''


# -------------------------- 基础工具函数（无修改） --------------------------
def load_stopwords(filename='中文停用词表.txt'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f.readlines()]
        return set(stopwords)
    except FileNotFoundError:
        print("警告：停用词表文件未找到，将不进行停用词过滤")
        return set()

def extract_text(html_doc):
    soup = BeautifulSoup(html_doc, 'lxml')
    raw_text = soup.get_text()
    text_with_space = raw_text.replace('\t', ' ').replace('\n', ' ')
    clean_text = re.sub(r'\s+', ' ', text_with_space).strip()
    return clean_text

def get_html(uri, headers={}, timeout=10):
    try:
        r = requests.get(uri, headers=headers, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except Exception as e:
        print(f"获取 {uri} 失败: {str(e)}")
        return None

def crawl_all_urls(html_doc, url):
    global input_url
    all_links = set()
    soup = BeautifulSoup(html_doc, 'lxml')
    for anchor in soup.find_all('a'):
        href = anchor.attrs.get("href")
        if not href or href == "":
            continue
        href = urljoin(url, href)
        normalized_href = url_normalize(href)
        # 过滤目标链接（科研网+学生处）
        if ('http://keyan' in normalized_href and '.htm' in normalized_href) or \
           ('http://xsc' in normalized_href and '.htm' in normalized_href):
            all_links.add(normalized_href)
    return all_links


# -------------------------- 爬虫核心函数（无修改） --------------------------
def crawl_one(url, headers, stopwords):
    global doc_id_counter, documents
    html_doc = get_html(url, headers=headers)
    if html_doc is None:
        with lock:
            used_urlset.add(url)
        return None

    text_content = extract_text(html_doc)
    if len(text_content) < 2:
        print(f"文档 {url} 文本过短（<2字），跳过")
        with lock:
            used_urlset.add(url)
        return None

    # 统计有效词频（BM25需基于有效词计算）
    words = jieba.lcut(text_content)
    filtered_words = [w for w in words if w not in stopwords and len(w) >= 2]
    word_freq = Counter(filtered_words)

    # 存储文档信息（含有效词频）
    with doc_lock:
        doc_id = doc_id_counter
        documents.append({
            "doc_id": doc_id,
            "url": url,
            "content": text_content,
            "word_freq": word_freq
        })
        doc_id_counter += 1

    # 提取链接并补充到队列
    url_sets = crawl_all_urls(html_doc, url)
    print(f"从 {url} 找到 {len(url_sets)} 个链接（文档ID：{doc_id}）")
    with lock:
        for new_url in url_sets:
            if new_url not in all_urlset:
                queue.append(new_url)
                all_urlset.add(new_url)

    with lock:
        used_urlset.add(url)
    return doc_id, word_freq


# -------------------------- 索引与BM25计算（核心修复） --------------------------
def build_inverted_index(stopwords):
    global inverted_index, total_docs
    total_docs = len(documents)
    inverted_index.clear()

    # 复用爬取时的词频构建倒排索引
    for doc in documents:
        doc_id = doc["doc_id"]
        word_freq = doc["word_freq"]
        with index_lock:
            for word, tf in word_freq.items():
                inverted_index[word].append((doc_id, tf))

    # 去重同一文档的重复词（保证DF准确性）
    with index_lock:
        for word in inverted_index:
            doc_set = set()
            unique_postings = []
            for (docid, tf) in inverted_index[word]:
                if docid not in doc_set:
                    doc_set.add(docid)
                    unique_postings.append((docid, tf))
            inverted_index[word] = unique_postings

    print(f"倒排索引构建完成：{len(inverted_index)} 个词，总文档数：{total_docs}")

def compute_bm25(k1=1.2, b=0.75):
    """
    修复点1：正确计算BM25权重+向量模长
    - 用有效词数（len(word_freq)）作为文档长度（更符合BM25定义）
    - 计算BM25向量模长存入doc_length（替代原TF-IDF模长）
    """
    global doc_bm25, total_docs, idf_dict, doc_length
    doc_bm25.clear()
    doc_length.clear()
    idf_dict.clear()

    # 1. 计算平均文档长度（有效词数的平均值）
    doc_len_list = [len(doc["word_freq"]) for doc in documents]
    avgdl = np.mean(doc_len_list) if doc_len_list else 0.0

    # 2. 计算IDF（BM25平滑公式）
    all_words = list(inverted_index.keys())
    for word in all_words:
        df = len(inverted_index[word])
        # 平滑避免DF=0时的极端值
        idf = np.log10((total_docs - df + 0.5) / (df + 0.5)) if df > 0 else 0.0
        idf_dict[word] = idf

    # 3. 计算每个文档的BM25权重+向量模长
    for doc in documents:
        doc_id = doc["doc_id"]
        word_freq = doc["word_freq"]
        doc_len = len(word_freq)  # 文档长度=有效词数（修复：原用len(content)不准确）
        bm25_vec = {}

        # 计算当前文档的BM25权重
        for word, tf in word_freq.items():
            if word not in idf_dict:
                continue
            # BM25核心公式
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
            bm25_vec[word] = idf_dict[word] * (numerator / denominator)

        # 保存BM25权重
        doc_bm25[doc_id] = bm25_vec

        # 计算BM25向量模长（用于后续余弦归一化）
        vec_values = np.array(list(bm25_vec.values()))
        vec_norm = np.sqrt(np.sum(vec_values ** 2)) if len(vec_values) > 0 else 0.0
        doc_length.append(vec_norm)

    print(f"BM25计算完成：{len(doc_bm25)} 个文档，k1={k1}, b={b}")


# -------------------------- 排序检索函数（核心修复） --------------------------
def cosine_search(query, top_k=20):
    """
    修复点2：适配BM25权重，修复逻辑结构
    - 计算查询的BM25权重（替代TF-IDF）
    - 从doc_bm25获取文档权重（替代doc_tfidf）
    - 修复短语匹配逻辑位置，避免结构错乱
    """
    # 替换原有分词：保留词性
    query_words = pseg.lcut(query)  # 返回[(词, 词性), ...]
    # 过滤+词性加权
    filtered_q_words = []
    for word, flag in query_words:
        if word in stopwords or len(word) < 2:
            continue
        # 词性权重：名词1.0，动词0.8，其他0.5
        weight = 1.0 if flag.startswith('n') else 0.8 if flag.startswith('v') else 0.5
        filtered_q_words.extend([word] * int(weight * 10))  # 用重复次数模拟权重
    q_word_freq = Counter(filtered_q_words)

    # 2. 计算查询的BM25权重（与文档端公式一致）
    q_bm25 = {}
    k1, b = 1.2, 0.75
    query_len = len(filtered_q_words)  # 查询长度=有效词数
    # 平均文档长度复用全局计算结果（从doc_len_list推导，避免重复计算）
    avgdl = np.mean([len(doc["word_freq"]) for doc in documents]) if documents else 0.0
    query_length_norm = query_len / avgdl if avgdl > 0 else 0.0

    for word, q_tf in q_word_freq.items():
        if word not in idf_dict:
            continue
        # 查询BM25权重公式（与文档端完全一致）
        numerator = q_tf * (k1 + 1)
        denominator = q_tf + k1 * (1 - b + b * query_length_norm)
        q_bm25[word] = idf_dict[word] * (numerator / denominator)

    # 3. 计算查询向量模长
    q_vec_values = np.array(list(q_bm25.values()))
    q_vec_norm = np.sqrt(np.sum(q_vec_values ** 2)) if len(q_vec_values) > 0 else 1.0

    # 4. 累加文档原始得分（基于BM25内积）
    doc_scores = defaultdict(float)
    for word, q_weight in q_bm25.items():
        postings = inverted_index.get(word, [])
        for (docid, _) in postings:
            # 从doc_bm25获取文档权重（修复：原用doc_tfidf）
            d_weight = doc_bm25[docid].get(word, 0.0)
            doc_scores[docid] += d_weight * q_weight

    # 5. 短语匹配加分（修复：移到得分累加后，避免重复计算）
    query_phrase = " ".join(filtered_q_words)
    phrase_boost = 0.2
    for docid in doc_scores:
        # 避免docid不存在的异常（修复：增加判断）
        doc_info = next((d for d in documents if d["doc_id"] == docid), None)
        if doc_info and query_phrase in doc_info["content"]:
            doc_scores[docid] *= (1 + phrase_boost)

    # 6. 计算余弦相似度并整理结果（修复：逻辑顺序正确）
    sorted_docs = []
    for docid, raw_score in doc_scores.items():
        # 避免文档长度越界或为0
        if docid >= len(doc_length) or doc_length[docid] == 0.0:
            continue
        # 余弦相似度公式（基于BM25权重）
        cos_score = raw_score / (q_vec_norm * doc_length[docid])
        # 获取文档信息
        doc_info = next((d for d in documents if d["doc_id"] == docid), None)
        if doc_info:
            preview = doc_info["content"][:150] + "..." if len(doc_info["content"]) > 150 else doc_info["content"]
            sorted_docs.append((cos_score, docid, doc_info["url"], preview))

    # 按相似度降序排序
    sorted_docs.sort(reverse=True, key=lambda x: x[0])
    return sorted_docs[:top_k]


# -------------------------- 状态保存与加载（核心修复） --------------------------
def save_state(queue, all_urlset, used_urlset, count, filename="crawler_state.pkl"):
    """
    修复点3：保存BM25相关数据（替代原TF-IDF）
    - 保存doc_bm25（替代doc_tfidf）
    - 保存doc_length（BM25向量模长）
    """
    global inverted_index, doc_bm25, doc_length, total_docs, idf_dict
    state = {
        'queue': queue,
        'all_urlset': all_urlset,
        'used_urlset': used_urlset,
        'count': count,
        'index': inverted_index,
        'documents': documents,
        'doc_id_counter': doc_id_counter,
        'doc_bm25': doc_bm25,  # 修复：替换原doc_tfidf
        'doc_length': doc_length,  # 保存BM25向量模长
        'total_docs': total_docs,
        'idf_dict': idf_dict
    }
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    print(f"状态已保存（含BM25数据）：{filename}")

def load_state(filename="crawler_state.pkl"):
    """
    修复点4：加载BM25相关数据
    - 恢复doc_bm25（替代原doc_tfidf）
    - 恢复doc_length（BM25向量模长）
    """
    global documents, doc_id_counter, inverted_index, doc_bm25, doc_length, total_docs, idf_dict
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        # 恢复核心数据（含BM25）
        documents = state['documents']
        doc_id_counter = state['doc_id_counter']
        inverted_index = state['index']
        doc_bm25 = state['doc_bm25']  # 修复：替换原doc_tfidf
        doc_length = state['doc_length']  # 恢复BM25向量模长
        total_docs = state['total_docs']
        idf_dict = state['idf_dict']
        print(f"已加载状态：总文档数={total_docs}，索引词数={len(inverted_index)}，BM25权重数={len(doc_bm25)}")
        return state
    except Exception as e:
        print(f"加载失败：{str(e)}，将从头开始")
        return None


# -------------------------- 交互与主函数（修复BM25调用） --------------------------
def search_interface():
    global total_docs
    state_file = "crawler_state.pkl"
    state = load_state(state_file)
    if not state or total_docs == 0:
        print("无爬取数据，请先运行爬虫")
        return

    print("\n" + "="*60)
    print("           基于BM25的排序检索系统")
    print("操作：输入查询关键词，输入'q'退出")
    print("="*60)

    while True:
        query = input("\n请输入查询关键词：").strip()
        if query.lower() == 'q':
            print("退出搜索")
            break
        if not query:
            print("请输入有效关键词")
            continue

        top_k = 20  # 固定返回前20（符合需求）
        results = cosine_search(query, top_k)

        if not results:
            print(f"未找到与「{query}」相关的文档")
            continue
        print(f"\n找到 {len(results)} 个相关文档（按相似度降序）：")
        for i, (score, docid, url, preview) in enumerate(results, 1):
            print(f"\n{i}. 相似度：{score:.4f} | 文档ID：{docid}")
            print(f"   URL：{url}")
            print(f"   预览：{preview}")

def main_crawl():
    input_urls = ['https://xsc.ruc.edu.cn/', 'http://keyan.ruc.edu.cn']
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    wait_time = 0.1  # 注意：间隔过短可能被反爬，建议调整为0.5s以上
    max_count = 10000
    max_workers = 5
    global queue, all_urlset, used_urlset, stopwords

    # 初始化/恢复状态
    resume = input("是否从上次状态继续爬取？(y/n): ").lower().startswith('y')
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
        # 重置所有状态（含BM25相关）
        global documents, doc_id_counter, doc_bm25, doc_length, total_docs, idf_dict
        documents = []
        doc_id_counter = 0
        doc_bm25 = defaultdict(dict)
        doc_length = []
        total_docs = 0
        idf_dict = {}
        queue = []
        all_urlset = set()
        used_urlset = set()
        count = 0
        for url in input_urls:
            if url not in all_urlset:
                queue.append(url)
                all_urlset.add(url)

    stopwords = load_stopwords()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while len(queue) > 0 and count < max_count:
                # 批量取URL
                batch = []
                with lock:
                    while queue and len(batch) < max_workers:
                        batch.append(queue.pop(0))
                        count += 1
                if not batch:
                    break

                # 提交爬取任务
                futures = [executor.submit(crawl_one, url, headers, stopwords) for url in batch]
                for f in as_completed(futures):
                    pass

                # 打印进度
                print(f"已爬取 {count} 页 | 队列剩余 {len(queue)} 页 | 总链接数 {len(all_urlset)}")
                time.sleep(wait_time)

        # 修复点5：爬取完成后调用compute_bm25（原漏调用，导致无BM25权重）
        build_inverted_index(stopwords)
        compute_bm25()  # 关键：计算BM25权重
        save_state(queue, all_urlset, used_urlset, count, state_file)
        print(f"\n爬取完成：")
        print(f"- 总爬取页面：{len(used_urlset)}")
        print(f"- 有效文档数：{total_docs}")
        print(f"- 倒排索引词数：{len(inverted_index)}")
        print(f"- BM25权重：{len(doc_bm25)} 个文档")

    except KeyboardInterrupt:
        build_inverted_index(stopwords)
        compute_bm25()
        save_state(queue, all_urlset, used_urlset, count, state_file)
        print("\n手动中断，已保存当前状态（含BM25数据）")

def evaluate(query, state_file="crawler_state.pkl"):
    """修复点6：适配BM25权重的评估函数"""
    global documents, inverted_index, doc_bm25, doc_length, total_docs, idf_dict, stopwords
    
    # 加载本地BM25数据
    with open(state_file, 'rb') as f:
        state = pickle.load(f)
    
    # 恢复BM25相关核心数据
    documents = state['documents']
    inverted_index = state['index']
    doc_bm25 = state['doc_bm25']  # 修复：替换原doc_tfidf
    doc_length = state['doc_length']
    total_docs = state['total_docs']
    idf_dict = state['idf_dict']
    stopwords = load_stopwords()
    
    # 执行BM25检索
    top_20_results = cosine_search(query, top_k=20)
    # 提取URL列表
    url_list = [result[2] for result in top_20_results]
    return url_list


if __name__ == "__main__":
    main_crawl()  # 爬取+构建索引+计算BM25
    search_interface()  # 检索交互
    # 示例：评估"数字孪生"并打印前20 URL
    print("\n评估查询「数字孪生」的前20 URL：")
    top_20_urls = evaluate('数字孪生')
    for i, url in enumerate(top_20_urls, 1):
        print(f"{i}. {url}")
