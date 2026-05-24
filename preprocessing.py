import json
import re
from underthesea import word_tokenize

# ==========================================
# CẤU HÌNH TỪ ĐIỂN VÀ STOPWORDS
# ==========================================

# 1. Chuẩn hóa thực thể: Báo chí thường viết sai khác nhau (Tê-hê-ran vs Tehran) [cite: 119, 120]
ENTITY_DICTIONARY = {
    "tê-hê-ran": "Tehran",
    "tê hê ran": "Tehran",
    "teheran": "Tehran",
    "oa-sinh-tơn": "Washington",
    "washington dc": "Washington",
    "huê kỳ": "Mỹ",
    "hoa kỳ": "Mỹ",
    "u.s": "Mỹ",
    "us": "Mỹ"
}

# 2. Stopwords cơ bản (Bạn có thể tải file vietnamese-stopwords.txt chuẩn trên mạng để thêm vào)
STOPWORDS = {"và", "là", "của", "các", "những", "đã", "đang", "sẽ", "để", "thì", "mà", "như", "một", "có", "với"}


# ==========================================
# CÁC HÀM XỬ LÝ DỮ LIỆU CỐT LÕI
# ==========================================

def normalize_entities(text):
    """Đồng nhất tên gọi các thực thể địa lý, chính trị"""
    # Dùng regex ignore case để thay thế
    for variant, standard in ENTITY_DICTIONARY.items():
        pattern = re.compile(re.escape(variant), re.IGNORECASE)
        text = pattern.sub(standard, text)
    return text


def clean_basic(text):
    """Làm sạch khoảng trắng và các ký tự rác cơ bản [cite: 21]"""
    text = re.sub(r'\s+', ' ', text)  # Xóa khoảng trắng thừa
    text = text.replace('"', '').replace("'", "")  # Bỏ ngoặc kép nhiễu
    return text.strip()


# --- TRƯỜNG HỢP 1: PREPROCESSING CHO TF-IDF ---
def preprocess_for_tfidf(text):
    text = clean_basic(text)
    text = normalize_entities(text)
    text = text.lower()  # Chữ thường hoàn toàn

    # Xóa dấu câu, chỉ giữ chữ cái, số và khoảng trắng
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tách từ (Segmentation) [cite: 86]
    segmented_text = word_tokenize(text, format="text")

    # Xóa Stopwords
    words = segmented_text.split()
    filtered_words = [w for w in words if w.replace('_', ' ') not in STOPWORDS]

    return " ".join(filtered_words)


# --- TRƯỜNG HỢP 2: PREPROCESSING CHO viBERT4news ---
def preprocess_for_bert(text):
    text = clean_basic(text)
    text = normalize_entities(text)

    # KHÔNG lower case, KHÔNG xóa dấu câu, KHÔNG xóa stopwords
    # Chỉ thực hiện Word Segmentation [cite: 88, 106]
    segmented_text = word_tokenize(text, format="text")

    return segmented_text


# ==========================================
# HÀM CHẠY CHÍNH (MAIN PROCESS)
# ==========================================
def run_preprocessing():
    input_file = "us_iran_news.json"
    output_file = "us_iran_news_processed.json"

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except FileNotFoundError:
        print(f"Không tìm thấy {input_file}. Hãy chạy Crawler trước.")
        return

    processed_articles = []
    print(f"Đang tiến hành tiền xử lý {len(articles)} bài báo...")

    for idx, article in enumerate(articles):
        # Đã cập nhật key lấy từ file crawl mới: 'content' thay vì 'nội dung'
        raw_content = article.get('content', '')

        if not raw_content:
            continue

        tfidf_content = preprocess_for_tfidf(raw_content)
        bert_content = preprocess_for_bert(raw_content)

        processed_articles.append({
            "article_id": article.get('article_id', ''),
            "publish_date": article.get('publish_date', ''),
            "title": article.get('title', ''),
            "nội_dung_gốc": raw_content,
            "nội_dung_tfidf": tfidf_content,
            "nội_dung_bert": bert_content
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_articles, f, ensure_ascii=False, indent=4)

    print(f"✅ HOÀN THÀNH TIỀN XỬ LÝ! Lưu tại: {output_file}")
    seen_titles = set()
    unique_articles = []
    for a in processed_articles:
        if a['title'] not in seen_titles:
            seen_titles.add(a['title'])
            unique_articles.append(a)

    print(f"Sau dedup: {len(unique_articles)}/{len(processed_articles)} bài unique")
    processed_articles = unique_articles