import requests
import json
from bs4 import BeautifulSoup
import time
from datetime import datetime
import re
import hashlib
import random
MAX_RETRIES = 3  # Số lần thử lại tối đa khi gặp lỗi mạng
from requests.exceptions import RequestException


# Cấu hình User-Agent
# Cấu hình User-Agent đa dạng hơn để đánh lừa hệ thống chống Bot
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
]

# ==========================================
# CÁC HÀM TIỆN ÍCH (UTILITIES)
# ==========================================

def get_html(url):
    """
        Hàm gọi HTTP an toàn, có tính năng thử lại (retry) khi bị timeout hoặc lỗi mạng.
        """
    for attempt in range(MAX_RETRIES):
        try:
            # Chọn ngẫu nhiên một User-Agent
            headers = {"User-Agent": random.choice(USER_AGENTS)}

            # Tăng timeout lên 15 giây
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                return response.text
            else:
                print(f"    [!] Cảnh báo: HTTP Status {response.status_code} tại {url}")

        except RequestException as e:
            print(f"    [!] Lỗi kết nối (Lần thử {attempt + 1}/{MAX_RETRIES}) tại {url}: {type(e).__name__}")
            if attempt < MAX_RETRIES - 1:
                # Tạm nghỉ 2-3 giây trước khi thử lại
                sleep_time = random.uniform(2, 4)
                print(f"        -> Đang chờ {sleep_time:.1f}s để thử lại...")
                time.sleep(sleep_time)
            else:
                print(f"    [X] Bỏ qua URL sau {MAX_RETRIES} lần thử thất bại.")
                return None
    return None

def generate_article_id(title, date_str):
    """Tạo ID duy nhất cho bài báo dựa trên hash của tiêu đề và ngày đăng"""
    raw_id = f"{title}_{date_str}".encode('utf-8')
    return hashlib.md5(raw_id).hexdigest()[:12]  # Lấy 12 ký tự đầu của mã băm


def extract_date(date_string):
    """Trích xuất ngày tháng theo định dạng dd/mm/yyyy từ chuỗi lộn xộn"""
    # Tìm kiếm chuỗi có dạng dd/mm/yyyy hoặc d/m/yyyy
    match = re.search(r'\d{1,2}/\d{1,2}/\d{4}', date_string)
    if match:
        return match.group(0)
    return None


def is_within_time_range(date_str, start_month, start_year, end_month, end_year):
    """Kiểm tra xem ngày đăng có nằm trong khoảng thời gian cho phép không"""
    clean_date_str = extract_date(date_str)
    if not clean_date_str:
        return False  # Bỏ qua nếu không lấy được ngày

    try:
        # Chuyển chuỗi ngày thành đối tượng datetime
        article_date = datetime.strptime(clean_date_str, "%d/%m/%Y")

        # Tạo datetime cho ngày bắt đầu (ngày 1 của tháng bắt đầu)
        start_date = datetime(start_year, start_month, 1)

        # Tạo datetime cho ngày kết thúc (ngày cuối cùng của tháng kết thúc, tính gần đúng)
        if end_month == 12:
            end_date = datetime(end_year + 1, 1, 1)  # Đầu năm sau
        else:
            end_date = datetime(end_year, end_month + 1, 1)  # Đầu tháng sau

        # Kiểm tra điều kiện (tính đến sát trước ngày mùng 1 tháng kế tiếp)
        return start_date <= article_date < end_date

    except ValueError:
        return False


# ==========================================
# CÁC HÀM CRAWL TỪNG BÁO
# ==========================================

def parse_vnexpress_article(url):
    html = get_html(url)
    if not html: return None
    soup = BeautifulSoup(html, 'html.parser')

    try:
        title = soup.find('h1', class_='title-detail').text.strip()
        raw_date = soup.find('span', class_='date').text.strip()
        clean_date = extract_date(raw_date)

        paragraphs = soup.find_all('p', class_='Normal')
        content = " ".join([p.text.strip() for p in paragraphs])

        if title and content and clean_date:
            article_id = generate_article_id(title, clean_date)
            return {
                "article_id": f"VNE_{article_id}",
                "title": title,
                "content": content,
                "publish_date": clean_date
            }
    except Exception:
        pass
    return None


def crawl_vnexpress(keyword, start_m, start_y, end_m, end_y, max_pages=1):
    articles = []
    for page in range(1, max_pages + 1):
        # VnExpress search pagination: &p=2
        search_url = f"https://timkiem.vnexpress.net/?q={keyword}&p={page}"
        html = get_html(search_url)
        if not html: continue

        soup = BeautifulSoup(html, 'html.parser')
        title_tags = soup.find_all('h3', class_='title-news')

        for tag in title_tags:
            a_tag = tag.find('a')
            if a_tag and 'href' in a_tag.attrs:
                article_url = a_tag['href']
                article_data = parse_vnexpress_article(article_url)

                # Kiểm tra bộ lọc thời gian
                if article_data and is_within_time_range(article_data['publish_date'], start_m, start_y, end_m, end_y):
                    articles.append(article_data)
                    print(f"  [+] Đã lấy: {article_data['title'][:50]}... ({article_data['publish_date']})")

                time.sleep(random.uniform(1.5, 3.5))  # Tránh bị chặn
    return articles


def parse_tuoitre_article(url):
    html = get_html(url)
    if not html: return None
    soup = BeautifulSoup(html, 'html.parser')

    try:
        title = soup.find('h1', class_='detail-title').text.strip()
        raw_date = soup.find('div', class_='detail-time').text.strip()
        clean_date = extract_date(raw_date)

        paragraphs = soup.find('div', class_='detail-cmain').find_all('p')
        content = " ".join([p.text.strip() for p in paragraphs])

        if title and content and clean_date:
            article_id = generate_article_id(title, clean_date)
            return {
                "article_id": f"TTO_{article_id}",
                "title": title,
                "content": content,
                "publish_date": clean_date
            }
    except Exception:
        pass
    return None


def crawl_tuoitre(keyword, start_m, start_y, end_m, end_y, max_pages=1):
    articles = []
    for page in range(1, max_pages + 1):
        search_url = f"https://tuoitre.vn/tim-kiem.htm?keywords={keyword}&page={page}"
        html = get_html(search_url)
        if not html: continue

        soup = BeautifulSoup(html, 'html.parser')
        title_tags = soup.find_all('h3', class_='box-title-text')

        for tag in title_tags:
            a_tag = tag.find('a')
            if a_tag and 'href' in a_tag.attrs:
                href = a_tag['href']
                article_url = "https://tuoitre.vn" + href if not href.startswith('http') else href
                article_data = parse_tuoitre_article(article_url)

                if article_data and is_within_time_range(article_data['publish_date'], start_m, start_y, end_m, end_y):
                    articles.append(article_data)
                    print(f"  [+] Đã lấy: {article_data['title'][:50]}... ({article_data['publish_date']})")

                time.sleep(random.uniform(1.5, 3.5))
    return articles


def parse_thanhnien_article(url):
    html = get_html(url)
    if not html: return None
    soup = BeautifulSoup(html, 'html.parser')

    try:
        title = soup.find('h1', class_='detail-title').text.strip()
        raw_date = soup.find('div', class_='detail-time').text.strip()
        clean_date = extract_date(raw_date)

        paragraphs = soup.find('div', class_='detail-cmain').find_all('p')
        content = " ".join([p.text.strip() for p in paragraphs])

        if title and content and clean_date:
            article_id = generate_article_id(title, clean_date)
            return {
                "article_id": f"TNO_{article_id}",
                "title": title,
                "content": content,
                "publish_date": clean_date
            }
    except Exception:
        pass
    return None


def crawl_thanhnien(keyword, start_m, start_y, end_m, end_y, max_pages=1):
    articles = []
    for page in range(1, max_pages + 1):
        search_url = f"https://thanhnien.vn/tim-kiem.htm?keywords={keyword}&page={page}"
        html = get_html(search_url)
        if not html: continue

        soup = BeautifulSoup(html, 'html.parser')
        title_tags = soup.find_all('h3', class_='box-title-text')

        for tag in title_tags:
            a_tag = tag.find('a')
            if a_tag and 'href' in a_tag.attrs:
                href = a_tag['href']
                article_url = "https://thanhnien.vn" + href if not href.startswith('http') else href
                article_data = parse_thanhnien_article(article_url)

                if article_data and is_within_time_range(article_data['publish_date'], start_m, start_y, end_m, end_y):
                    articles.append(article_data)
                    print(f"  [+] Đã lấy: {article_data['title'][:50]}... ({article_data['publish_date']})")

                time.sleep(random.uniform(1.5, 3.5))
    return articles


# ==========================================
# HÀM CHÍNH ĐƯỢC GỌI TỪ MAIN.PY
# ==========================================
def run_crawler():
    keyword = "Mỹ Iran"
    START_MONTH = 2
    START_YEAR = 2026
    END_MONTH = 5
    END_YEAR = 2026
    MAX_PAGES_TO_SEARCH = 10

    print(f"\n=== BẮT ĐẦU CRAWL TỪ {START_MONTH}/{START_YEAR} ĐẾN {END_MONTH}/{END_YEAR} ===")
    print(f"Từ khóa: '{keyword}'")

    all_data = []

    print("\n[1] Đang crawl VnExpress...")
    vne_data = crawl_vnexpress(keyword, START_MONTH, START_YEAR, END_MONTH, END_YEAR, max_pages=MAX_PAGES_TO_SEARCH)
    all_data.extend(vne_data)

    print("\n[2] Đang crawl Tuổi Trẻ...")
    tto_data = crawl_tuoitre(keyword, START_MONTH, START_YEAR, END_MONTH, END_YEAR, max_pages=MAX_PAGES_TO_SEARCH)
    all_data.extend(tto_data)

    print("\n[3] Đang crawl Thanh Niên...")
    tno_data = crawl_thanhnien(keyword, START_MONTH, START_YEAR, END_MONTH, END_YEAR, max_pages=MAX_PAGES_TO_SEARCH)
    all_data.extend(tno_data)

    # Lưu dữ liệu ra file JSON
    output_file = "us_iran_news.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 HOÀN THÀNH! Đã lọc và crawl được tổng cộng {len(all_data)} bài viết thỏa mãn điều kiện thời gian.")
    print(f"Dữ liệu đã được lưu chuẩn form vào file: {output_file}")