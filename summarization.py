import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def generate_single_summary(filtered_file_path, output_file_path, pipeline_name):
    """
    Đọc các bài báo đã được lọc trùng từ K-means, gom chúng lại và tiến hành tóm tắt ra 1 bản duy nhất.
    """
    print(f"\n--- TIẾN HÀNH TÓM TẮT DUY NHẤT CHO HƯỚNG: {pipeline_name} ---")

    with open(filtered_file_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    # Tổng hợp nội dung từ các bài báo cốt lõi đã được lọc trùng
    MAX_WORDS_PER_ARTICLE = 300
    truncated_texts = [
        " ".join(a['nội_dung_gốc'].split()[:MAX_WORDS_PER_ARTICLE])
        for a in articles
    ]
    combined_text = " ".join(truncated_texts)

    # Cấu hình mô hình Transformer ViT5
    model_name = "VietAI/vit5-base-vietnews-summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tạo đầu vào theo đúng định dạng VietNews
    input_text = "vietnews: " + combined_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=224,  # 1. Tăng giới hạn trần ~230 từ
            min_length=148,  # 2. Tăng giới hạn sàn ~190 từ
            num_beams=6,  # 3. Tăng số lượng chùm tìm kiếm để mở rộng không gian chọn từ
            length_penalty=2,  # 4. Đặt length_penalty >= 1.0 để phạt các chuỗi ngắn, khuyến khích câu dài
            no_repeat_ngram_size=3,  # 5. Chặn lặp cụm 3 từ liên tiếp khi ép mô hình viết dài
            early_stopping=False
        )

    """
    min_length=150: Tạo ra một bộ lọc bắt buộc (hard constraint). Mô hình sẽ không được phép phân phối xác suất cho token kết thúc (<eos>) 
    cho đến khi chuỗi giải mã đạt tối thiểu 150 tokens.

    length_penalty=2.5: Điểm số của chuỗi giải mã được tính bằng công thức tỉ lệ thuận với độ dài câu lũy thừa hóa. Khi giá trị này lớn hơn 1.0, 
    các chuỗi có độ dài lớn hơn sẽ có lợi thế về điểm số xác suất tích lũy trong quá trình duyệt cây Beam Search.

    no_repeat_ngram_size=3: Khi ép các mô hình Transformer viết dài, một lỗi hệ thống phổ biến là hiện tượng "vòng lặp vô hạn" 
    (nhắc đi nhắc lại một cụm từ). Việc cấu hình tham số này bằng 3 sẽ triệt tiêu xác suất của bất kỳ từ nào 
    nếu nó chuẩn bị tạo thành một cụm 3 từ đã từng xuất hiện trước đó trong sequence, đảm bảo văn bản sinh ra phong phú và mạch lạc hơn.
    """

    master_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = {
        "hướng_tiếp_cận": pipeline_name,
        "bản_tóm_tắt": master_summary
    }

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"✅ Đã xuất bản tóm tắt tổng duy nhất vào: {output_file_path}")


def run_summarization():
    # Chạy tóm tắt duy nhất cho nhánh TF-IDF từ tập tài liệu đã loại trùng
    generate_single_summary(
        filtered_file_path="us_iran_filtered_tfidf.json",
        output_file_path="final_summary_tfidf.json",
        pipeline_name="Crawl -> Preprocessing -> TF-IDF -> K-means (Centroid Filtering) -> ViT5"
    )

    # Chạy tóm tắt duy nhất cho nhánh BERT từ tập tài liệu đã loại trùng
    generate_single_summary(
        filtered_file_path="us_iran_filtered_bert.json",
        output_file_path="final_summary_bert.json",
        pipeline_name="Crawl -> Preprocessing -> viBERT4news -> K-means (Centroid Filtering) -> ViT5"
    )
