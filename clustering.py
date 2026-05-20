import json
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel


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


def run_clustering():
    input_file = "us_iran_news_processed.json"
    num_clusters = 8

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except FileNotFoundError:
        print("Không tìm thấy file processed. Hãy chạy preprocessing trước.")
        return

    # --- NHÁNH 1: TF-IDF VÀ LỌC TRÙNG ---
    print("\n--- CHẠY PHÂN CỤM VÀ LỌC TRÙNG VỚI TF-IDF ---")
    corpus_tfidf = [a['nội_dung_tfidf'] for a in articles]
    vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = vectorizer.fit_transform(corpus_tfidf)

    kmeans_tfidf = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans_tfidf.fit(X_tfidf)

    filtered_tfidf = filter_redundancy_by_centroid(articles, X_tfidf, kmeans_tfidf, num_clusters, top_k_per_cluster=1)

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
    kmeans_bert = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans_bert.fit(X_bert)

    filtered_bert = filter_redundancy_by_centroid(articles, X_bert, kmeans_bert, num_clusters, top_k_per_cluster=1)

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