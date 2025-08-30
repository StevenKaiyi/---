import pickle
import os
import jieba
from urllib.request import urljoin
from bs4 import BeautifulSoup
import requests
import time
from url_normalize import url_normalize
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

# 全局变量用于存储文档信息
documents = []  # 存储格式: {"doc_id": id, "url": url, "content": 文本内容}
doc_id_counter = 0  # 文档ID计数器

# 线程锁
lock = threading.Lock()
doc_lock = threading.Lock()

# 保存爬虫和索引状态
def save_state(queue, all_urlset, used_urlset, count, index, filename="crawler_state.pkl"):
    state = {
        'queue': queue,
        'all_urlset': all_urlset,
        'used_urlset': used_urlset,
        'count': count,
        'index': index,
        'documents': documents,
        'doc_id_counter': doc_id_counter
    }
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    print(f"状态已保存到 {filename}")

# 加载爬虫和索引状态
def load_state(filename="crawler_state.pkl"):
    global documents, doc_id_counter
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        documents = state['documents']
        doc_id_counter = state['doc_id_counter']
        print(f"已从 {filename} 加载状态")
        return state
    except:
        print(f"加载 {filename} 失败，将从头开始")
        return None

# 加载停用词
def load_stopwords(filename='中文停用词表.txt'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f.readlines()]
        return set(stopwords)
    except FileNotFoundError:
        print("警告：停用词表文件未找到，将不进行停用词过滤")
        return set()

# 提取网页中的文本内容
def extract_text(html_doc):
    soup = BeautifulSoup(html_doc, 'lxml') 
    raw_text = soup.get_text()
    text_with_space = raw_text.replace('\t', ' ').replace('\n', ' ')
    clean_text= re.sub(r'\s+', ' ', text_with_space)
    return clean_text

# 爬取链接
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

# 获取网页内容
def get_html(uri, headers={}, timeout=10):
    try:
        r = requests.get(uri, headers=headers, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding 
        return r.text
    except Exception as e:
        print(f"获取 {uri} 失败: {str(e)}")
        return None

# 处理单个网页
def crawl_one(url, headers, stopwords):
    global doc_id_counter, documents
    
    html_doc = get_html(url, headers=headers)
    if html_doc is None:
        with lock:
            used_urlset.add(url)
        return
    
    # 提取文本内容
    text_content = extract_text(html_doc)
    
    # 保存文档信息
    with doc_lock:
        doc_id = doc_id_counter
        documents.append({
            "doc_id": doc_id,
            "url": url,
            "content": text_content
        })
        doc_id_counter += 1
    

    # 提取链接
    url_sets = crawl_all_urls(html_doc, url)
    print(f"从 {url} 找到 {len(url_sets)} 个链接")

    # 添加新链接到队列
    with lock:
        for new_url in url_sets:
            # 这里可以根据需要修改过滤条件
            if (new_url not in all_urlset and 
                'https://xsc' in new_url and 
                '.htm' in new_url):
                queue.append(new_url)
                all_urlset.add(new_url)

    # 标记为已处理
    with lock:
        used_urlset.add(url)
    
    return doc_id, text_content

# 构建倒排索引
def build_inverted_index(documents, stopwords):
    inverted_index = defaultdict(list)
    for doc in documents:
        doc_id = doc["doc_id"]
        content = doc["content"]
        
        # 分词
        words = jieba.lcut(content)
        # 过滤并添加到索引
        seen_words = set()  # 避免同一文档中重复索引同一词
        for word in words:
            if (word not in stopwords and 
                len(word) >= 2 and 
                word not in seen_words):
                inverted_index[word].append(doc_id)
                seen_words.add(word)
    
    return inverted_index

# 主爬虫函数
def main_crawl():
    # 初始URL
    input_urls = ['https://xsc.ruc.edu.cn/']
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'} 
    wait_time = 0.5  # 爬取间隔，礼貌爬虫
    max_count = 100000  # 最大爬取数量，可根据需要调整
    max_workers = 10   # 线程数
    
    # 全局变量
    global queue, all_urlset, used_urlset, inverted_index
    inverted_index = defaultdict(list)
    
    # 询问是否从上次状态继续
    resume = input("是否要从上次的状态继续爬取？(y/n): ").lower().startswith('y')
    state_file = "crawler_state.pkl"

    # 初始化或恢复状态
    if resume:
        state = load_state(state_file)
        if state:
            queue = state['queue']
            all_urlset = state['all_urlset']
            used_urlset = state['used_urlset']
            count = state['count']
            inverted_index = state['index']
        else:
            queue = []
            all_urlset = set()
            for url in input_urls:
                if url not in all_urlset:
                    queue.append(url)
                    all_urlset.add(url)
            used_urlset = set()
            count = 0
    else:
        # 重置状态
        global documents, doc_id_counter
        documents = []
        doc_id_counter = 0
        queue = []
        all_urlset = set()
        for url in input_urls:
            if url not in all_urlset:
                queue.append(url)
                all_urlset.add(url)
        used_urlset = set()
        count = 0
        inverted_index = defaultdict(list)
    
    # 加载停用词
    stopwords = load_stopwords()
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while len(queue) > 0 and count < max_count:
                # 每次取一批URL进行爬取
                batch = []
                with lock:
                    while queue and len(batch) < max_workers:
                        batch.append(queue.pop(0))
                        count += 1

                if not batch:
                    break

                # 提交爬取任务
                futures = [executor.submit(crawl_one, url, headers, stopwords) for url in batch]
                
                # 处理爬取结果并更新索引
                for f in as_completed(futures):
                    result = f.result()
                    if result:
                        doc_id, text_content = result
                        # 实时更新倒排索引
                        words = jieba.lcut(text_content)
                        seen_words = set()
                        for word in words:
                            if (word not in stopwords and 
                                len(word) >= 2 and 
                                word not in seen_words):
                                inverted_index[word].append(doc_id)
                                seen_words.add(word)

                print(f"已爬取 {count} 页，队列剩余：{len(queue)}，总链接数：{len(all_urlset)}")

                # 定期询问是否继续
                if count < max_count and len(queue) > 0 and count % 10 == 0:
                    continue_crawl = input("是否继续爬取？(y/n，默认y): ").lower()
                    if continue_crawl.startswith('n'):
                        save_state(queue, all_urlset, used_urlset, count, inverted_index, state_file)
                        print("已暂停爬取")
                        return

                # 礼貌等待
                if wait_time > 0 and len(queue) > 0:
                    time.sleep(wait_time)

        # 爬取完成，保存状态
        save_state(queue, all_urlset, used_urlset, count, inverted_index, state_file)
        print("\n爬取完成")
        print(f"总共爬取了 {len(used_urlset)} 个页面")
        print(f"发现了 {len(all_urlset)} 个独特URL")
        print(f"建立了包含 {len(inverted_index)} 个词的索引")

    except KeyboardInterrupt:
        save_state(queue, all_urlset, used_urlset, count, inverted_index, state_file)
        print("\n用户中断，已保存状态")

#判断And
def Intersection(word1,word2):
    answer=[]
    p1,p2=iter(word1),iter(word2)
    try:
        docid1,docid2=next(p1),next(p2)
        while True:
            if docid1==docid2:
                answer.append(docid1)
                docid1,docid2=next(p1),next(p2)
            elif docid1<docid2:
                docid1=next(p1)
            else:
                docid2=next(p2)
    except StopIteration:
        pass
    return answer

#搜索 AND
def And_query(inverted_index,word1,word2):
    docs1 = inverted_index.get(word1, [-1])[1:]
    docs2 = inverted_index.get(word2, [-1])[1:]
    return Intersection(docs1,docs2)

#搜索Or
def Or_query(inverted_index,word1,word2):
    docs1=inverted_index.get(word1,[-1])[1:]
    docs2=inverted_index.get(word2,[-1])[1:]
    answer=set(docs1)
    for doc in docs2:
        answer.add(doc)
    return answer

# 搜索交互函数
def search():
    global inverted_index
    state_file = "crawler_state.pkl"
    
    # 加载状态
    state = load_state(state_file)
    if not state:
        print("没有找到爬取数据，请先运行爬虫")
        return
    
    inverted_index = state['index']
    
    while True:
        operation=input("请输入搜索类型：and/or")
        word1 = input("\n请输入第一个搜索关键词")
        word2=input("\n请输入第二个关键词：")
        if operation=='and':
            results = And_query(inverted_index, word1,word2)
        else:
            results= Or_query(inverted_index,word1,word2)
        
        print(results)

def get_document_via_id(id,documents):
    for doc in documents:
        if id==doc["doc_id"]:
            return (doc['url'],doc['content'])
        
def main():
    main_crawl()
    #search()
    print(get_document_via_id(3000,documents))

main()
