def web_search(query, state_file="crawler_state.pkl"):
    global documents, inverted_index, doc_bm25, doc_length, total_docs, idf_dict, stopwords
    
    # 加载状态数据
    with open(state_file, 'rb') as f:
        state = pickle.load(f)
    
    # 恢复全局变量
    documents = state['documents']
    inverted_index = state['index']
    doc_bm25 = state['doc_bm25']
    doc_length = state['doc_length']
    total_docs = state['total_docs']
    idf_dict = state['idf_dict']
    stopwords = load_stopwords()
    
    # 执行检索（cosine_search返回格式：(cos_score, docid, url, preview)）
    top_20_results = cosine_search(query, top_k=20)
    
    # 构造前端需要的结果格式
    formatted_results = []
    for score, docid, url, preview in top_20_results:
        # 1. 获取文档原始数据
        doc_info = documents[docid]  # 简化取值，避免重复索引
        title = doc_info.get('title', '无标题')  # 用get避免键不存在报错
        raw_keywords = doc_info.get('keywords', set())  # 原始关键词（set类型）
        word_freq = doc_info.get('word_freq', Counter())  # 词频统计（Counter类型）
        
        # 2. 处理关键词：无效时用正文词频Top10替代，有效时转列表
        if not raw_keywords or any(kw.lower() == 'null' for kw in raw_keywords):
            # 无效关键词场景：空集合 / 包含null（不区分大小写）
            # 取词频最高的前10个词（用most_common确保按频率排序）
            top10_words = [word for word, _ in word_freq.most_common(10)]
            processed_keywords = top10_words if top10_words else ['无关键词']
        else:
            # 有效关键词场景：set转列表（前端遍历更友好），并去重（set已去重，直接转）
            processed_keywords = list(raw_keywords)
        
        # 3. 格式化结果（关键词转为字符串更易前端显示，可选列表）
        # 若前端需要逗号分隔的字符串，可替换为：','.join(processed_keywords)
        formatted_results.append({
            'url': url,
            'title': title,
            'abstract': preview,
            'keywords': processed_keywords  # 处理后的关键词（列表类型，前端可直接遍历）
        })
    return formatted_results
