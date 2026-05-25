import json
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import silhouette_score


def filter_redundancy_by_centroid(articles, X, kmeans, num_clusters, top_k_per_cluster=1):
    """
    Tìm các bài báo gần tâm cụm nhất để lọc bỏ trùng lặp và giữ lại ý chính của từng tiểu chủ đề.
    """
    # kmeans.transform(X) trả về khoảng cách từ mỗi bài báo tới tất cả các tâm cụm
    distances = kmeans.transform(X)
    labels = kmeans.labels_

    filtered_articles = []

    for c in range(num_clusters):
        # Lấy chỉ số của các bài báo thuộc cụm c
        cluster_indices = np.where(labels == c)[0]
        if len(cluster_indices) == 0:
            continue

        # Lấy khoảng cách tới tâm cụm c của các bài báo trong cụm đó
        cluster_distances = distances[cluster_indices, c]

        # Tìm các bài báo có khoảng cách nhỏ nhất (gần tâm nhất)
        closest_cluster_indices = np.argsort(cluster_distances)[:top_k_per_cluster]
        global_indices = cluster_indices[closest_cluster_indices]

        for idx in global_indices:
            filtered_articles.append(articles[idx])

    return filtered_articles

def find_optimal_k(X, max_k=10):
    num_samples = X.shape[0]
    if num_samples < 3:
        return 1

    if hasattr(X, "toarray"):
        X_dense = X.toarray()
    else:
        X_dense = X

    # Giới hạn max_k hợp lý: tối đa 15, tối thiểu 2
    actual_max_k = min(max_k, num_samples - 1, 15)
    actual_max_k = max(2, actual_max_k)

    # --- ELBOW METHOD dựa trên inertia ---
    inertias = []
    k_range = range(2, actual_max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_dense if len(X_dense) <= 500 else X_dense[
            np.random.default_rng(42).choice(len(X_dense), 500, replace=False)
        ])
        inertias.append(km.inertia_)

    # Tìm "khuỷu tay" = điểm giảm tốc độ giảm nhiều nhất
    # Dùng tỉ lệ cải thiện: nếu thêm 1 cụm không giảm >5% thì dừng
    best_k = 2
    for i in range(1, len(inertias)):
        improvement = (inertias[i-1] - inertias[i]) / inertias[i-1]
        prev_improvement = (inertias[i-2] - inertias[i-1]) / inertias[i-2] if i > 1 else 1.0
        # Elbow: tốc độ cải thiện giảm mạnh (xuống dưới 40% của lần trước)
        if improvement < prev_improvement * 0.4:
            best_k = list(k_range)[i-1]
            break
        best_k = list(k_range)[i]

    # Giới hạn thực tế: với báo chí đơn chủ đề không nên quá 10 cụm
    best_k = min(best_k, 10)

    print(f"    Elbow Method chọn K = {best_k} (inertia range: {inertias[0]:.0f} → {inertias[-1]:.0f})")
    return best_k

def run_clustering():
    input_file = "us_iran_news_processed.json"

    KEYWORDS = ['mỹ', 'iran', 'tehran', 'washington', 'đàm phán',
                'hạt nhân', 'hormuz', 'trump', 'nuclear']
    def is_relevant(article):
        text = (article.get('title','') + ' ' + article.get('nội_dung_gốc','')).lower()
        return any(kw in text for kw in KEYWORDS)


    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except FileNotFoundError:
        print("Không tìm thấy file processed.")
        return

    if not articles:
        print("❌ Không có bài nào trong file processed. Hãy chạy lại preprocessing.")
        return

    # Kiểm tra nội_dung_tfidf không rỗng
    corpus_check = [a.get('nội_dung_tfidf', '') for a in articles]
    if not any(corpus_check):
        print("❌ Tất cả nội_dung_tfidf đều rỗng. Hãy chạy lại preprocessing.")
        return

    # --- NHÁNH 1: TF-IDF VÀ LỌC TRÙNG ---
    print("\n--- CHẠY PHÂN CỤM VÀ LỌC TRÙNG VỚI TF-IDF ---")
    corpus_tfidf = [a['nội_dung_tfidf'] for a in articles]
    vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = vectorizer.fit_transform(corpus_tfidf)

    # hàm tối ưu K
    optimal_k_tfidf = find_optimal_k(X_tfidf.toarray() if hasattr(X_tfidf, "toarray") else X_tfidf, max_k=10)
    print(f"[*] Thuật toán tự động chọn K = {optimal_k_tfidf} cho TF-IDF")

    kmeans_tfidf = KMeans(n_clusters=optimal_k_tfidf, random_state=42, n_init=10)
    kmeans_tfidf.fit(X_tfidf)

    filtered_tfidf = filter_redundancy_by_centroid(
                        articles, X_tfidf, kmeans_tfidf, optimal_k_tfidf,
                        top_k_per_cluster=3  # lấy 3 bài/cụm thay vì 1
                    )

    filtered_tfidf = [a for a in filtered_tfidf if is_relevant(a)]

    # 🌟 ĐỘC LẬP HÓA DỮ LIỆU: Chỉ giữ lại các trường liên quan đến TF-IDF
    clean_tfidf_data = []
    for item in filtered_tfidf:
        clean_tfidf_data.append({
            "article_id": item.get("article_id"),
            "publish_date": item.get("publish_date"),
            "title": item.get("title"),
            "nội_dung_gốc": item.get("nội_dung_gốc"),
            "nội_dung_tfidf": item.get("nội_dung_tfidf")  # Giữ lại trường này
        })

    with open("us_iran_filtered_tfidf.json", "w", encoding="utf-8") as f:
        json.dump(clean_tfidf_data, f, ensure_ascii=False, indent=4)
    print(f"✅ TF-IDF đã lưu {len(clean_tfidf_data)} bài báo đại diện.")

    # --- NHÁNH 2: BERT VÀ LỌC TRÙNG ---
    print("\n--- CHẠY PHÂN CỤM VÀ LỌC TRÙNG VỚI viBERT4news ---")
    tokenizer = AutoTokenizer.from_pretrained("FPTAI/vibert-base-cased")
    model = AutoModel.from_pretrained("FPTAI/vibert-base-cased")

    embeddings = []
    corpus_bert = [a['nội_dung_bert'] for a in articles]

    for text in corpus_bert:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        sentence_emb = torch.mean(outputs.last_hidden_state.squeeze(0), dim=0).numpy()
        embeddings.append(sentence_emb)

    X_bert = np.array(embeddings)

    # hàm tối ưu K
    optimal_k_bert = find_optimal_k(X_bert, max_k=10)
    print(f"[*] Thuật toán tự động chọn K = {optimal_k_bert} cho viBERT4news")

    kmeans_bert = KMeans(n_clusters=optimal_k_bert, random_state=42, n_init=10)
    kmeans_bert.fit(X_bert)

    filtered_bert = filter_redundancy_by_centroid(
                    articles, X_bert, kmeans_bert, optimal_k_bert,
                    top_k_per_cluster=3
                    )

    filtered_bert = [a for a in filtered_bert if is_relevant(a)]

    # 🌟 ĐỘC LẬP HÓA DỮ LIỆU: Chỉ giữ lại các trường liên quan đến BERT
    clean_bert_data = []
    for item in filtered_bert:
        clean_bert_data.append({
            "article_id": item.get("article_id"),
            "publish_date": item.get("publish_date"),
            "title": item.get("title"),
            "nội_dung_gốc": item.get("nội_dung_gốc"),
            "nội_dung_bert": item.get("nội_dung_bert")  # Giữ lại trường này
        })

    with open("us_iran_filtered_bert.json", "w", encoding="utf-8") as f:
        json.dump(clean_bert_data, f, ensure_ascii=False, indent=4)
    print(f"✅ BERT đã lưu {len(clean_bert_data)} bài báo đại diện.")