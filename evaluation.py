import json
import os
from rouge_score import rouge_scorer
from underthesea import word_tokenize
from openai import OpenAI  # <-- Sử dụng thư viện openai thay vì gemini

# ==========================================
# CẤU HÌNH API LLM (BÊN THỨ 3 TƯƠNG THÍCH OPENAI)
# ==========================================
API_KEY = "sk-57IfkGiaNziGyb9BgPOXPF5yZPrjAV73rCA2HtQ0tmfcd2Hh"
BASE_URL = "https://llm.wokushop.com/v1"
MODEL_NAME = "gpt-5-nano"


def create_ground_truth_via_llm(input_file="us_iran_news_processed.json", gt_file="ground_truth.txt"):
    """Sử dụng LLM để đọc toàn bộ dữ liệu gốc và sinh ra bản tóm tắt tiêu chuẩn (Ground Truth)"""
    if os.path.exists(gt_file):
        print(f"Đã tìm thấy file {gt_file}. Bỏ qua bước gọi LLM.")
        with open(gt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()

    print("\n--- ĐANG GỌI LLM ĐỂ TẠO GROUND TRUTH ---")
    if API_KEY == "sk-57IfkGiaNziGyb9BgPOXPF5yZPrjAV73rCA2HtQ0tmfcd2Hh":
        print(
            "CẢNH BÁO: Chưa cấu hình API_KEY. Bạn hãy tự copy các bài báo nhờ ChatGPT tóm tắt và lưu vào file 'ground_truth.txt' nhé!")
        return None

    # Load dữ liệu gốc
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except FileNotFoundError:
        print(f"Không tìm thấy file {input_file}. Vui lòng chạy các bước trước đó.")
        return None

    all_texts = "\n\n".join([f"Bài {i + 1}: " + a['nội_dung_gốc'] for i, a in enumerate(articles)])

    # Khởi tạo OpenAI Client với Custom Base URL
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # Phân chia Prompt theo chuẩn OpenAI (System và User)
    system_prompt = """
    Bạn là một nhà báo và biên tập viên chuyên nghiệp. 
    Nhiệm vụ của bạn là tổng hợp và viết MỘT bản tóm tắt duy nhất (Ground Truth), khái quát đầy đủ các sự kiện, nguyên nhân và kết quả chính.
    Độ dài yêu cầu: Khoảng 150 - 250 từ. Văn phong: Khách quan, trung lập báo chí.
    """

    user_prompt = f"Dưới đây là danh sách các bài báo thu thập được về chủ đề 'Xung đột Mỹ - Iran'. Hãy tóm tắt chúng:\n\n{all_texts}"

    try:
        # Gọi API ChatCompletion
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Giảm sáng tạo để LLM bám sát sự thật báo chí
            max_tokens=500
        )

        # Trích xuất văn bản từ response
        ground_truth = response.choices[0].message.content.strip()

        # Lưu lại để dùng cho các lần sau
        with open(gt_file, 'w', encoding='utf-8') as f:
            f.write(ground_truth)
        print("✅ Đã tạo xong Ground Truth từ LLM bên thứ 3!")

        return ground_truth

    except Exception as e:
        print(f"❌ Lỗi khi gọi API: {e}")
        return None


def calculate_rouge(hypothesis, reference):
    """Tính toán ROUGE Score. Chú ý: Phải tách từ (Segment) trước khi tính để đảm bảo ROUGE chạy đúng cho tiếng Việt"""
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
    try:
        with open("final_summary_tfidf.json", 'r', encoding='utf-8') as f:
            summary_tfidf = json.load(f)["bản_tóm_tắt"]  # Cập nhật theo file summarization hiện tại của bạn

        with open("final_summary_bert.json", 'r', encoding='utf-8') as f:
            summary_bert = json.load(f)["bản_tóm_tắt"]
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


if __name__ == "__main__":
    run_evaluation()