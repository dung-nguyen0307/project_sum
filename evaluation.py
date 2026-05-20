import json
import os
from rouge_score import rouge_scorer
from underthesea import word_tokenize
import google.generativeai as genai

# ==========================================
# CẤU HÌNH API LLM (Thay bằng API Key của bạn nếu muốn tự động tạo Ground Truth)
# Bạn có thể lấy key miễn phí tại: https://aistudio.google.com/
# ==========================================
API_KEY = "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY"  # Ví dụ: "AIzaSy..."


def create_ground_truth_via_llm(input_file="us_iran_news_processed.json", gt_file="ground_truth.txt"):
    """Sử dụng LLM để đọc toàn bộ dữ liệu gốc và sinh ra bản tóm tắt tiêu chuẩn (Ground Truth)"""
    if os.path.exists(gt_file):
        print(f"Đã tìm thấy file {gt_file}. Bỏ qua bước gọi LLM.")
        with open(gt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()

    print("\n--- ĐANG GỌI LLM ĐỂ TẠO GROUND TRUTH ---")
    if API_KEY == "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY":
        print(
            "CẢNH BÁO: Chưa cấu hình API_KEY. Bạn hãy tự copy các bài báo nhờ ChatGPT tóm tắt và lưu vào file 'ground_truth.txt' nhé!")
        return None

    # Load dữ liệu gốc
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    all_texts = "\n\n".join([f"Bài {i + 1}: " + a['nội_dung_gốc'] for i, a in enumerate(articles)])

    # Prompt yêu cầu LLM làm Ground Truth
    prompt = f"""
    Bạn là một nhà báo và biên tập viên chuyên nghiệp. Dưới đây là danh sách các bài báo thu thập được về chủ đề 'Xung đột Mỹ - Iran'.
    Nhiệm vụ của bạn là tổng hợp và viết MỘT bản tóm tắt duy nhất (Ground Truth), khái quát đầy đủ các sự kiện, nguyên nhân và kết quả chính.
    Độ dài yêu cầu: Khoảng 150 - 250 từ. Văn phong: Khách quan, trung lập báo chí.

    Dữ liệu:
    {all_texts}
    """

    # Gọi API Gemini (hoặc bạn có thể dùng thư viện openai tương tự)
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    ground_truth = response.text.strip()

    # Lưu lại để dùng cho các lần sau
    with open(gt_file, 'w', encoding='utf-8') as f:
        f.write(ground_truth)
    print("✅ Đã tạo xong Ground Truth từ LLM!")

    return ground_truth


def calculate_rouge(hypothesis, reference):
    """Tính toán ROUGE Score. Chú ý: Phải tách từ (Segment) trước khi tính để đảm bảo ROUGE chạy đúng cho tiếng Việt"""
    # Tách từ bằng '_' (ví dụ: "tàu_sân_bay") để ROUGE xem nó là 1 từ duy nhất (1 gram)
    hyp_segmented = word_tokenize(hypothesis, format="text")
    ref_segmented = word_tokenize(reference, format="text")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = scorer.score(ref_segmented, hyp_segmented)
    return scores


def run_evaluation():
    print("\n====== ĐÁNH GIÁ MÔ HÌNH VỚI ROUGE SCORE ======")

    # 1. Lấy Ground Truth
    ground_truth = create_ground_truth_via_llm()
    if not ground_truth:
        return

    print("\n[GROUND TRUTH TỪ LLM]:")
    print(ground_truth)
    print("-" * 50)

    # 2. Đọc file tóm tắt của 2 nhánh
    results = {}

    try:
        with open("final_summary_tfidf.json", 'r', encoding='utf-8') as f:
            summary_tfidf = json.load(f)["bản_tóm_tắt_duy_nhất"]

        with open("final_summary_bert.json", 'r', encoding='utf-8') as f:
            summary_bert = json.load(f)["bản_tóm_tắt_duy_nhất"]
    except FileNotFoundError:
        print("Không tìm thấy file kết quả tóm tắt. Vui lòng chạy bước summarization trước.")
        return

    # 3. Tính điểm ROUGE
    print("\nĐang tính toán độ đo ROUGE...")
    rouge_tfidf = calculate_rouge(summary_tfidf, ground_truth)
    rouge_bert = calculate_rouge(summary_bert, ground_truth)

    # Hàm in kết quả đẹp
    def print_scores(name, scores):
        print(f"\n[Kết quả cho luồng: {name}]")
        print(
            f" - ROUGE-1 (Độ trùng lặp từ đơn) : F1 = {scores['rouge1'].fmeasure:.4f} | Precision = {scores['rouge1'].precision:.4f} | Recall = {scores['rouge1'].recall:.4f}")
        print(
            f" - ROUGE-2 (Độ trùng lặp cặp từ) : F1 = {scores['rouge2'].fmeasure:.4f} | Precision = {scores['rouge2'].precision:.4f} | Recall = {scores['rouge2'].recall:.4f}")
        print(
            f" - ROUGE-L (Mạch lạc cấu trúc)  : F1 = {scores['rougeL'].fmeasure:.4f} | Precision = {scores['rougeL'].precision:.4f} | Recall = {scores['rougeL'].recall:.4f}")

    print_scores("TF-IDF + K-Means + ViT5", rouge_tfidf)
    print_scores("viBERT + K-Means + ViT5", rouge_bert)

    # 4. Lưu báo cáo đánh giá
    evaluation_report = {
        "TF-IDF": {
            "ROUGE-1": rouge_tfidf['rouge1'].fmeasure,
            "ROUGE-2": rouge_tfidf['rouge2'].fmeasure,
            "ROUGE-L": rouge_tfidf['rougeL'].fmeasure
        },
        "BERT": {
            "ROUGE-1": rouge_bert['rouge1'].fmeasure,
            "ROUGE-2": rouge_bert['rouge2'].fmeasure,
            "ROUGE-L": rouge_bert['rougeL'].fmeasure
        }
    }
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=4)

    print("\n✅ Đã lưu báo cáo so sánh vào evaluation_report.json")