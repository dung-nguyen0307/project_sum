from crawler import run_crawler
from preprocessing import run_preprocessing
from clustering import run_clustering
from summarization import run_summarization
# from evaluation import run_evaluation

if __name__ == "__main__":
    print("=== KÍCH HOẠT PIPELINE TÓM TẮT ĐA VĂN BẢN TIẾNG VIỆT ===")
    # run_crawler()       # Bước 1: Crawl data báo
    run_preprocessing() # Bước 2: Tiền xử lý
    run_clustering()  # Bước 3: Phân cụm & Lọc trùng tại tâm cụm
    run_summarization()  # Bước 4: Tổng hợp ra 1 bản tóm tắt duy nhất cho mỗi luồng
    # run_evaluation()  # Bước 5: Chấm điểm ROUGE với Ground Truth
    print("\n🎉 HOÀN THÀNH TOÀN BỘ LUỒNG PIPELINE!")