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

    # 1. Lấy Ground Truth — ĐỌC NỘI DUNG FILE
    gt_file = "ground_truth.txt"
    if not os.path.exists(gt_file):
        print(f"❌ Không tìm thấy {gt_file}.")
        return
    with open(gt_file, "r", encoding="utf-8") as f:
        ground_truth = f.read().strip()

    print("\n[GROUND TRUTH]:")
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

    # 4. Lưu báo cáo đánh giá (đầy đủ F1 + Precision + Recall)
    evaluation_report = {
        "TF-IDF": {
            "ROUGE-1": {"F1": rouge_tfidf['rouge1'].fmeasure, "Precision": rouge_tfidf['rouge1'].precision, "Recall": rouge_tfidf['rouge1'].recall},
            "ROUGE-2": {"F1": rouge_tfidf['rouge2'].fmeasure, "Precision": rouge_tfidf['rouge2'].precision, "Recall": rouge_tfidf['rouge2'].recall},
            "ROUGE-L": {"F1": rouge_tfidf['rougeL'].fmeasure, "Precision": rouge_tfidf['rougeL'].precision, "Recall": rouge_tfidf['rougeL'].recall},
        },
        "BERT": {
            "ROUGE-1": {"F1": rouge_bert['rouge1'].fmeasure, "Precision": rouge_bert['rouge1'].precision, "Recall": rouge_bert['rouge1'].recall},
            "ROUGE-2": {"F1": rouge_bert['rouge2'].fmeasure, "Precision": rouge_bert['rouge2'].precision, "Recall": rouge_bert['rouge2'].recall},
            "ROUGE-L": {"F1": rouge_bert['rougeL'].fmeasure, "Precision": rouge_bert['rougeL'].precision, "Recall": rouge_bert['rougeL'].recall},
        },
        "summary_tfidf": summary_tfidf,
        "summary_bert": summary_bert,
    }
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=4)

    print("\n✅ Đã lưu báo cáo so sánh vào evaluation_report.json")

    # 5. Sinh file visualize HTML
    visualize_results(evaluation_report)


def visualize_results(report, output_html="evaluation_visualization.html"):
    """
    Đọc evaluation_report và sinh file HTML trực quan với biểu đồ Chart.js.
    Có thể gọi độc lập: visualize_results(json.load(open('evaluation_report.json')))
    """
    tfidf = report["TF-IDF"]
    bert  = report["BERT"]

    # Hàm tiện ích: lấy giá trị an toàn (hỗ trợ cả format cũ lẫn mới)
    def get(block, metric, key):
        val = block[metric]
        if isinstance(val, dict):
            return val.get(key, 0.0)
        return val if key == "F1" else 0.0

    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]

    # Dữ liệu cho từng biểu đồ
    f1_tfidf  = [get(tfidf, m, "F1")        for m in metrics]
    f1_bert   = [get(bert,  m, "F1")        for m in metrics]
    pr_tfidf  = [get(tfidf, m, "Precision") for m in metrics]
    re_tfidf  = [get(tfidf, m, "Recall")    for m in metrics]
    pr_bert   = [get(bert,  m, "Precision") for m in metrics]
    re_bert   = [get(bert,  m, "Recall")    for m in metrics]

    winner = "TF-IDF" if sum(f1_tfidf) >= sum(f1_bert) else "viBERT"
    best_r1 = max(get(tfidf, "ROUGE-1", "F1"), get(bert, "ROUGE-1", "F1"))
    best_r2 = max(get(tfidf, "ROUGE-2", "F1"), get(bert, "ROUGE-2", "F1"))
    best_rl = max(get(tfidf, "ROUGE-L", "F1"), get(bert, "ROUGE-L", "F1"))

    summary_tfidf = report.get("summary_tfidf", "—")
    summary_bert  = report.get("summary_bert",  "—")

    def fmt(v): return f"{v:.4f}"
    def pct(v): return f"{v*100:.1f}%"

    def score_row(metric, tfidf_b, bert_b):
        f1_t = get(tfidf_b, metric, "F1");  pr_t = get(tfidf_b, metric, "Precision"); re_t = get(tfidf_b, metric, "Recall")
        f1_b = get(bert_b,  metric, "F1");  pr_b = get(bert_b,  metric, "Precision"); re_b = get(bert_b,  metric, "Recall")
        win_t = "color:#1D9E75;font-weight:500" if f1_t >= f1_b else ""
        win_b = "color:#1D9E75;font-weight:500" if f1_b >  f1_t else ""
        return f"""
        <tr>
          <td style="padding:10px 12px;font-weight:500;color:#444">{metric}</td>
          <td style="padding:10px 12px;text-align:center;{win_t}">{fmt(f1_t)}</td>
          <td style="padding:10px 12px;text-align:center">{fmt(pr_t)}</td>
          <td style="padding:10px 12px;text-align:center">{fmt(re_t)}</td>
          <td style="padding:10px 12px;text-align:center;border-left:1px solid #eee;{win_b}">{fmt(f1_b)}</td>
          <td style="padding:10px 12px;text-align:center">{fmt(pr_b)}</td>
          <td style="padding:10px 12px;text-align:center">{fmt(re_b)}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ROUGE Evaluation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f7;color:#1d1d1f;padding:32px 24px}}
  h1{{font-size:22px;font-weight:600;margin-bottom:4px}}
  .sub{{color:#6e6e73;font-size:14px;margin-bottom:28px}}
  .cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:28px}}
  .card{{background:#fff;border-radius:12px;padding:18px 20px;border:1px solid #e5e5ea}}
  .card .label{{font-size:12px;color:#6e6e73;margin-bottom:6px}}
  .card .value{{font-size:26px;font-weight:600;color:#1d1d1f}}
  .card .tag{{font-size:11px;color:#1D9E75;margin-top:4px}}
  .section{{background:#fff;border-radius:12px;padding:22px 24px;margin-bottom:20px;border:1px solid #e5e5ea}}
  .section h2{{font-size:15px;font-weight:600;margin-bottom:16px;color:#1d1d1f}}
  .charts{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  thead tr{{background:#f5f5f7}}
  thead th{{padding:10px 12px;text-align:center;font-weight:500;color:#6e6e73;font-size:12px}}
  thead th:first-child{{text-align:left}}
  tbody tr{{border-top:1px solid #f0f0f0}}
  tbody tr:hover{{background:#fafafa}}
  .winner-badge{{display:inline-block;background:#e1f5ee;color:#0F6E56;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:500;margin-left:8px}}
  .summary-box{{background:#f9f9f9;border:1px solid #e5e5ea;border-radius:8px;padding:14px 16px;font-size:13px;line-height:1.7;color:#3a3a3c}}
  .pipeline-label{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#6e6e73;margin-bottom:6px}}
  .gap{{height:12px}}
  .legend{{display:flex;gap:20px;font-size:12px;color:#6e6e73;margin-bottom:14px;align-items:center}}
  .dot{{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:5px}}
  .pr-re-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:4px}}
  .pr-re-card{{background:#f9f9f9;border-radius:8px;padding:12px 14px}}
  .pr-re-card h3{{font-size:12px;font-weight:600;color:#6e6e73;margin-bottom:8px}}
  .pr-re-row{{display:flex;justify-content:space-between;font-size:12px;padding:3px 0;border-bottom:1px solid #eee}}
  .pr-re-row:last-child{{border-bottom:none}}
  .high{{color:#1D9E75;font-weight:500}} .low{{color:#D85A30;font-weight:500}}
</style>
</head>
<body>
<h1>ROUGE Evaluation Report</h1>
<p class="sub">Pipeline: Crawl → Preprocessing → Clustering (K-Means) → ViT5 Summarization</p>

<div class="cards">
  <div class="card">
    <div class="label">ROUGE-1 F1 tốt nhất</div>
    <div class="value">{fmt(best_r1)}</div>
    <div class="tag">▲ Trùng lặp từ đơn — pipeline thắng: {winner}</div>
  </div>
  <div class="card">
    <div class="label">ROUGE-2 F1 tốt nhất</div>
    <div class="value">{fmt(best_r2)}</div>
    <div class="tag">▲ Trùng lặp cặp từ — pipeline thắng: {winner}</div>
  </div>
  <div class="card">
    <div class="label">ROUGE-L F1 tốt nhất</div>
    <div class="value">{fmt(best_rl)}</div>
    <div class="tag">▲ Mạch lạc cấu trúc — pipeline thắng: {winner}</div>
  </div>
</div>

<div class="charts">
  <div class="section">
    <h2>So sánh F1 Score</h2>
    <div class="legend">
      <span><span class="dot" style="background:#185FA5"></span>TF-IDF + K-Means + ViT5</span>
      <span><span class="dot" style="background:#1D9E75"></span>viBERT + K-Means + ViT5</span>
    </div>
    <div style="position:relative;height:260px;width:100%"><canvas id="f1Chart" role="img" aria-label="So sánh F1 ROUGE giữa TF-IDF và viBERT"></canvas></div>
  </div>
  <div class="section">
    <h2>Precision vs Recall theo metric</h2>
    <div class="legend">
      <span><span class="dot" style="background:#185FA5"></span>Precision TF-IDF</span>
      <span><span class="dot" style="background:#9FE1CB"></span>Recall TF-IDF</span>
      <span><span class="dot" style="background:#1D9E75"></span>Precision viBERT</span>
      <span><span class="dot" style="background:#B5D4F4"></span>Recall viBERT</span>
    </div>
    <div style="position:relative;height:260px;width:100%"><canvas id="prChart" role="img" aria-label="Precision và Recall của 2 pipeline"></canvas></div>
  </div>
</div>

<div class="section">
  <h2>Bảng chi tiết đầy đủ</h2>
  <table>
    <thead>
      <tr>
        <th style="text-align:left">Metric</th>
        <th colspan="3" style="border-right:1px solid #ddd">TF-IDF + K-Means + ViT5 <span class="winner-badge">{'★ Tốt hơn' if winner == 'TF-IDF' else ''}</span></th>
        <th colspan="3">viBERT + K-Means + ViT5 <span class="winner-badge">{'★ Tốt hơn' if winner == 'viBERT' else ''}</span></th>
      </tr>
      <tr>
        <th></th>
        <th>F1</th><th>Precision</th><th>Recall</th>
        <th style="border-left:1px solid #eee">F1</th><th>Precision</th><th>Recall</th>
      </tr>
    </thead>
    <tbody>
      {''.join(score_row(m, tfidf, bert) for m in metrics)}
    </tbody>
  </table>
</div>

<div class="section">
  <h2>Phân tích Precision / Recall</h2>
  <div class="pr-re-grid">
    <div class="pr-re-card">
      <h3>TF-IDF + K-Means + ViT5</h3>
      {''.join(f'<div class="pr-re-row"><span>{m}</span><span>P=<b class="{"high" if get(tfidf,m,"Precision")>=0.5 else "low"}">{pct(get(tfidf,m,"Precision"))}</b> &nbsp; R=<b class="{"high" if get(tfidf,m,"Recall")>=0.5 else "low"}">{pct(get(tfidf,m,"Recall"))}</b></span></div>' for m in metrics)}
      <div style="margin-top:10px;font-size:12px;color:#6e6e73">
        {'⚠ Precision > Recall: bản tóm tắt chính xác nhưng chưa bao phủ đủ ý' if get(tfidf,'ROUGE-1','Precision') > get(tfidf,'ROUGE-1','Recall') else '✓ Cân bằng Precision và Recall'}
      </div>
    </div>
    <div class="pr-re-card">
      <h3>viBERT + K-Means + ViT5</h3>
      {''.join(f'<div class="pr-re-row"><span>{m}</span><span>P=<b class="{"high" if get(bert,m,"Precision")>=0.5 else "low"}">{pct(get(bert,m,"Precision"))}</b> &nbsp; R=<b class="{"high" if get(bert,m,"Recall")>=0.5 else "low"}">{pct(get(bert,m,"Recall"))}</b></span></div>' for m in metrics)}
      <div style="margin-top:10px;font-size:12px;color:#6e6e73">
        {'⚠ Precision > Recall: bản tóm tắt chính xác nhưng chưa bao phủ đủ ý' if get(bert,'ROUGE-1','Precision') > get(bert,'ROUGE-1','Recall') else '✓ Cân bằng Precision và Recall'}
      </div>
    </div>
  </div>
</div>

<div class="section">
  <h2>Bản tóm tắt sinh ra</h2>
  <div class="pipeline-label">TF-IDF + K-Means + ViT5</div>
  <div class="summary-box">{summary_tfidf}</div>
  <div class="gap"></div>
  <div class="pipeline-label">viBERT + K-Means + ViT5</div>
  <div class="summary-box">{summary_bert}</div>
</div>

<script>
const labels = ['ROUGE-1','ROUGE-2','ROUGE-L'];
new Chart(document.getElementById('f1Chart'),{{
  type:'bar',
  data:{{
    labels,
    datasets:[
      {{label:'TF-IDF',data:[{','.join(fmt(v) for v in f1_tfidf)}],backgroundColor:'#185FA5',borderRadius:5}},
      {{label:'viBERT',data:[{','.join(fmt(v) for v in f1_bert)}],backgroundColor:'#1D9E75',borderRadius:5}},
      {{label:'Ngưỡng tốt',data:[0.45,0.20,0.40],type:'line',borderColor:'#aaa',borderDash:[5,5],pointRadius:0,borderWidth:1.5,fill:false}}
    ]
  }},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>c.dataset.label+': '+parseFloat(c.parsed.y).toFixed(4)}}}}}},scales:{{y:{{min:0,max:1,ticks:{{callback:v=>v.toFixed(1)}}}},x:{{ticks:{{autoSkip:false}}}}}}}}
}});

new Chart(document.getElementById('prChart'),{{
  type:'bar',
  data:{{
    labels,
    datasets:[
      {{label:'P TF-IDF',data:[{','.join(fmt(v) for v in pr_tfidf)}],backgroundColor:'#185FA5',borderRadius:4}},
      {{label:'R TF-IDF',data:[{','.join(fmt(v) for v in re_tfidf)}],backgroundColor:'#9FE1CB',borderRadius:4}},
      {{label:'P viBERT',data:[{','.join(fmt(v) for v in pr_bert)}],backgroundColor:'#1D9E75',borderRadius:4}},
      {{label:'R viBERT',data:[{','.join(fmt(v) for v in re_bert)}],backgroundColor:'#B5D4F4',borderRadius:4}}
    ]
  }},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>c.dataset.label+': '+parseFloat(c.parsed.y).toFixed(4)}}}}}},scales:{{y:{{min:0,max:1,ticks:{{callback:v=>v.toFixed(1)}}}},x:{{ticks:{{autoSkip:false}}}}}}}}
}});
</script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Đã sinh file visualize: {output_html} — mở bằng trình duyệt để xem!")


if __name__ == "__main__":
    run_evaluation()