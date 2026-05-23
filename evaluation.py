import json
import os
from rouge_score import rouge_scorer
from underthesea import word_tokenize

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
    gt_file = "ground_truth.txt"
    if not os.path.exists(gt_file):
        print(f"❌ Không tìm thấy {gt_file}. Vui lòng tạo file này trước.")
        return

    with open(gt_file, 'r', encoding='utf-8') as f:
        ground_truth = f.read().strip()  # <-- ĐỌC NỘI DUNG THỰC SỰ

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