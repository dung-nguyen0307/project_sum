# 📰 Vietnamese Multi-Document Summarization Pipeline

Hệ thống tóm tắt đa văn bản tiếng Việt tự động, áp dụng cho bộ dữ liệu báo chí chủ đề **Quan hệ Mỹ – Iran (2/2026 – 5/2026)**. Pipeline so sánh hai hướng tiếp cận biểu diễn văn bản: **TF-IDF** và **viBERT4news**, kết hợp phân cụm K-Means và mô hình tóm tắt **ViT5**.

---

## 🏗️ Kiến trúc Pipeline

```
Crawl (VnExpress / Tuổi Trẻ / Thanh Niên)
    ↓
Preprocessing (Làm sạch · Chuẩn hóa thực thể · Word Segmentation · Stopwords)
    ↓
         ┌─────────────────────┐         ┌─────────────────────────┐
         │  Nhánh 1: TF-IDF    │         │   Nhánh 2: viBERT4news  │
         │  TF-IDF Vectorizer  │         │   Sentence Embedding    │
         │  K-Means Clustering │         │   K-Means Clustering    │
         │  Centroid Filtering │         │   Centroid Filtering    │
         └──────────┬──────────┘         └────────────┬────────────┘
                    ↓                                  ↓
              ViT5 Summarization               ViT5 Summarization
                    ↓                                  ↓
              final_summary_tfidf.json     final_summary_bert.json
                    ↓                                  ↓
                    └──────────┬───────────────────────┘
                               ↓
                    ROUGE Evaluation + HTML Visualization
```

---

## 📁 Cấu trúc thư mục

```
project/
├── crawler.py                  # Bước 1: Thu thập dữ liệu từ 3 tờ báo
├── preprocessing.py            # Bước 2: Tiền xử lý & dedup
├── clustering.py               # Bước 3: Phân cụm K-Means (Elbow) + lọc bài đại diện
├── summarization.py            # Bước 4: Tóm tắt với ViT5
├── evaluation.py               # Bước 5: Đánh giá ROUGE + sinh HTML visualize
├── main.py                     # Entry point chạy toàn bộ pipeline
├── ground_truth.txt            # Bản tóm tắt chuẩn để đánh giá ROUGE
├── us_iran_news.json           # Dữ liệu crawl thô (sinh ra khi chạy crawler)
├── us_iran_news_processed.json # Dữ liệu sau preprocessing (sinh ra khi chạy)
├── us_iran_filtered_tfidf.json # Bài đại diện nhánh TF-IDF (sinh ra khi chạy)
├── us_iran_filtered_bert.json  # Bài đại diện nhánh BERT (sinh ra khi chạy)
├── final_summary_tfidf.json    # Kết quả tóm tắt TF-IDF (sinh ra khi chạy)
├── final_summary_bert.json     # Kết quả tóm tắt BERT (sinh ra khi chạy)
├── evaluation_report.json      # Báo cáo điểm ROUGE (sinh ra khi chạy)
└── evaluation_visualization.html  # Dashboard HTML (sinh ra khi chạy)
```

---

## ⚙️ Cài đặt

### Yêu cầu
- Python 3.10+
- GPU khuyến nghị cho bước embedding BERT và tóm tắt ViT5 (CPU vẫn chạy được, chậm hơn)

### Cài thư viện

```bash
pip install torch transformers
pip install scikit-learn underthesea
pip install rouge-score beautifulsoup4 requests
```

---

## 🚀 Cách chạy

### Chạy toàn bộ pipeline (bỏ qua crawler nếu đã có dữ liệu)

```python
# Trong main.py, bật/tắt từng bước tùy nhu cầu:
# run_crawler()       # Bỏ comment nếu muốn crawl lại
run_preprocessing()
run_clustering()
run_summarization()
run_evaluation()
```

```bash
python main.py
```

### Chạy riêng từng bước

```bash
# Chỉ crawl
python -c "from crawler import run_crawler; run_crawler()"

# Chỉ đánh giá & visualize (nếu đã có các file JSON)
python -c "from evaluation import run_evaluation; run_evaluation()"

# Chỉ sinh lại HTML từ evaluation_report.json có sẵn
python -c "
import json
from evaluation import visualize_results
report = json.load(open('evaluation_report.json', encoding='utf-8'))
visualize_results(report)
"
```

---

## 📊 Kết quả

| Metric    | TF-IDF + K-Means + ViT5 | viBERT + K-Means + ViT5 |
|-----------|-----------|-------------------------|
| ROUGE-1 F1 | 0.6071    | **0.6179**              |
| ROUGE-2 F1 | 0.2972    | **0.3144**              |
| ROUGE-L F1 | 0.2888    | **0.3003**              |

> Sau khi chạy xong, mở file `evaluation_visualization.html` bằng trình duyệt để xem dashboard đầy đủ.

---

## 🔍 Giải thích các bước chính

### Bước 2 — Preprocessing
- Chuẩn hóa thực thể (Tehran/Tê-hê-ran → Tehran, Hoa Kỳ → Mỹ...)
- Word segmentation dùng `underthesea`
- Dedup theo title để loại bài trùng lặp do crawl nhiều lần

### Bước 3 — Clustering
- **Nhánh TF-IDF**: vector hóa bằng `TfidfVectorizer(max_features=1000)`, phân cụm K-Means
- **Nhánh BERT**: embedding câu bằng `FPTAI/vibert-base-cased`, phân cụm K-Means
- Tìm K tối ưu bằng **Elbow Method** (dựa trên inertia), giới hạn `max_k=10`
- Mỗi cụm lấy `top_k=3` bài gần tâm nhất, lọc thêm bài lạc chủ đề bằng keyword filter

### Bước 4 — Summarization
- Mô hình: `VietAI/vit5-base-vietnews-summarization`
- Input: ghép các bài đại diện, truncate mỗi bài còn 300 từ trước khi nối
- Tham số sinh: `max_length=500`, `min_length=320`, `num_beams=6`, `length_penalty=2.0`
- Lưu ý: `min/max_length` tính bằng **token** (≈ 2 token/từ tiếng Việt với SentencePiece)

### Bước 5 — Evaluation
- Tách từ tiếng Việt trước khi tính ROUGE (dùng `underthesea`)
- Báo cáo đầy đủ F1, Precision, Recall cho ROUGE-1, ROUGE-2, ROUGE-L
- Sinh file `evaluation_visualization.html` với biểu đồ Chart.js

---

## 📝 Ghi chú kỹ thuật

| Vấn đề | Giải pháp |
|--------|-----------|
| Silhouette Score không phù hợp cho TF-IDF sparse | Thay bằng Elbow Method (inertia) |
| `min_length` token ≠ số từ tiếng Việt | Nhân 2x so với số từ mong muốn |
| viBERT embedding chọn bài lạc chủ đề | Thêm keyword filter sau clustering |
| Dữ liệu crawl bị duplicate | Dedup theo title trong cả crawler và preprocessing |
| ROUGE = 0 khi ground truth là tên file | Đọc nội dung file bằng `open().read().strip()` |

---

## 🗃️ Nguồn dữ liệu

| Báo | URL tìm kiếm |
|-----|-------------|
| VnExpress | `https://timkiem.vnexpress.net/?q=Mỹ+Iran` |
| Tuổi Trẻ | `https://tuoitre.vn/tim-kiem.htm?keywords=Mỹ+Iran` |
| Thanh Niên | `https://thanhnien.vn/tim-kiem.htm?keywords=Mỹ+Iran` |

Khoảng thời gian thu thập: **02/2026 – 05/2026**

---

## 📦 Models sử dụng

| Model | Nguồn | Mục đích |
|-------|-------|---------|
| `FPTAI/vibert-base-cased` | HuggingFace | Sentence embedding tiếng Việt |
| `VietAI/vit5-base-vietnews-summarization` | HuggingFace | Tóm tắt abstractive tiếng Việt |
