import json
import os
from rouge_score import rouge_scorer
from underthesea import word_tokenize


def calculate_rouge(hypothesis, reference):
    hyp_segmented = word_tokenize(hypothesis, format="text")
    ref_segmented = word_tokenize(reference, format="text")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    return scorer.score(ref_segmented, hyp_segmented)


def run_evaluation():
    print("\n====== ĐÁNH GIÁ MÔ HÌNH VỚI ROUGE SCORE ======")

    # 1. Đọc Ground Truth từ file
    gt_file = "ground_truth.txt"
    if not os.path.exists(gt_file):
        print(f"❌ Không tìm thấy {gt_file}.")
        return
    with open(gt_file, 'r', encoding='utf-8') as f:
        ground_truth = f.read().strip()

    print("\n[GROUND TRUTH]:")
    print(ground_truth)
    print("-" * 50)

    # 2. Đọc file tóm tắt của 2 nhánh
    try:
        with open("final_summary_tfidf.json", 'r', encoding='utf-8') as f:
            summary_tfidf = json.load(f)["bản_tóm_tắt"]
        with open("final_summary_bert.json", 'r', encoding='utf-8') as f:
            summary_bert = json.load(f)["bản_tóm_tắt"]
    except FileNotFoundError:
        print("Không tìm thấy file kết quả tóm tắt. Vui lòng chạy bước summarization trước.")
        return

    # 3. Tính điểm ROUGE
    print("\nĐang tính toán độ đo ROUGE...")
    rouge_tfidf = calculate_rouge(summary_tfidf, ground_truth)
    rouge_bert  = calculate_rouge(summary_bert,  ground_truth)

    def print_scores(name, scores):
        print(f"\n[Kết quả cho luồng: {name}]")
        print(f" - ROUGE-1 : F1={scores['rouge1'].fmeasure:.4f} | P={scores['rouge1'].precision:.4f} | R={scores['rouge1'].recall:.4f}")
        print(f" - ROUGE-2 : F1={scores['rouge2'].fmeasure:.4f} | P={scores['rouge2'].precision:.4f} | R={scores['rouge2'].recall:.4f}")
        print(f" - ROUGE-L : F1={scores['rougeL'].fmeasure:.4f} | P={scores['rougeL'].precision:.4f} | R={scores['rougeL'].recall:.4f}")

    print_scores("TF-IDF + K-Means + ViT5", rouge_tfidf)
    print_scores("viBERT + K-Means + ViT5",  rouge_bert)

    # 4. Lưu báo cáo đầy đủ
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
        "summary_bert":  summary_bert,
        "ground_truth":  ground_truth,
    }
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=4)
    print("\n✅ Đã lưu báo cáo vào evaluation_report.json")

    # 5. Sinh HTML visualize
    visualize_results(evaluation_report)


def visualize_results(report, output_html="evaluation_visualization.html"):
    """
    Sinh file HTML visualize từ evaluation_report.
    Gọi độc lập: visualize_results(json.load(open('evaluation_report.json')))
    """
    tfidf = report["TF-IDF"]
    bert  = report["BERT"]

    def get(block, metric, key):
        val = block[metric]
        if isinstance(val, dict):
            return val.get(key, 0.0)
        return val if key == "F1" else 0.0

    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]

    f1_tfidf = [get(tfidf, m, "F1")        for m in metrics]
    f1_bert  = [get(bert,  m, "F1")        for m in metrics]
    pr_tfidf = [get(tfidf, m, "Precision") for m in metrics]
    re_tfidf = [get(tfidf, m, "Recall")    for m in metrics]
    pr_bert  = [get(bert,  m, "Precision") for m in metrics]
    re_bert  = [get(bert,  m, "Recall")    for m in metrics]

    winner   = "TF-IDF" if sum(f1_tfidf) >= sum(f1_bert) else "viBERT"
    best_r1  = max(get(tfidf, "ROUGE-1", "F1"), get(bert, "ROUGE-1", "F1"))
    best_r2  = max(get(tfidf, "ROUGE-2", "F1"), get(bert, "ROUGE-2", "F1"))
    best_rl  = max(get(tfidf, "ROUGE-L", "F1"), get(bert, "ROUGE-L", "F1"))

    summary_tfidf     = report.get("summary_tfidf", "—")
    summary_bert      = report.get("summary_bert",  "—")
    # Ưu tiên đọc thẳng từ file để luôn có nội dung mới nhất
    gt_file = "ground_truth.txt"
    if os.path.exists(gt_file):
        with open(gt_file, 'r', encoding='utf-8') as f:
            ground_truth_text = f.read().strip()
    else:
        ground_truth_text = report.get("ground_truth", "—")

    def fmt(v): return f"{v:.4f}"
    def pct(v): return f"{v*100:.1f}%"
    def js_arr(lst): return "[" + ",".join(fmt(v) for v in lst) + "]"

    def score_row(metric):
        f1_t = get(tfidf, metric, "F1");  pr_t = get(tfidf, metric, "Precision"); re_t = get(tfidf, metric, "Recall")
        f1_b = get(bert,  metric, "F1");  pr_b = get(bert,  metric, "Precision"); re_b = get(bert,  metric, "Recall")
        st = "color:#1D9E75;font-weight:600" if f1_t >= f1_b else "color:#444"
        sb = "color:#1D9E75;font-weight:600" if f1_b >  f1_t else "color:#444"
        return (f'<tr>'
                f'<td class="metric-name">{metric}</td>'
                f'<td style="{st}">{fmt(f1_t)}</td><td>{fmt(pr_t)}</td><td>{fmt(re_t)}</td>'
                f'<td class="divider" style="{sb}">{fmt(f1_b)}</td><td>{fmt(pr_b)}</td><td>{fmt(re_b)}</td>'
                f'</tr>')

    def pr_rows(block):
        rows = ""
        for m in metrics:
            p = get(block, m, "Precision"); r = get(block, m, "Recall")
            cp = "high" if p >= 0.5 else "low"; cr = "high" if r >= 0.5 else "low"
            rows += f'<div class="pr-row"><span>{m}</span><span>P=<b class="{cp}">{pct(p)}</b>&nbsp; R=<b class="{cr}">{pct(r)}</b></span></div>'
        note = "⚠ Precision > Recall: tóm tắt chính xác nhưng chưa bao phủ đủ ý" \
               if get(block,"ROUGE-1","Precision") > get(block,"ROUGE-1","Recall") \
               else "✓ Cân bằng Precision và Recall"
        return rows + f'<div class="pr-note">{note}</div>'

    score_rows_html = "".join(score_row(m) for m in metrics)
    winner_tfidf = "★ Tốt hơn" if winner == "TF-IDF" else ""
    winner_bert  = "★ Tốt hơn" if winner == "viBERT"  else ""

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ROUGE Evaluation Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f2f5;color:#1d1d1f;padding:28px 20px}}
h1{{font-size:21px;font-weight:700;margin-bottom:4px}}
.sub{{color:#6e6e73;font-size:13px;margin-bottom:24px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px}}
.card{{background:#fff;border-radius:12px;padding:16px 20px;border:1px solid #e0e0e5}}
.card .lbl{{font-size:11px;color:#888;margin-bottom:6px;text-transform:uppercase;letter-spacing:.4px}}
.card .val{{font-size:28px;font-weight:700}}
.card .tag{{font-size:11px;color:#1D9E75;margin-top:5px}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
.section{{background:#fff;border-radius:12px;padding:20px 22px;margin-bottom:16px;border:1px solid #e0e0e5}}
.section h2{{font-size:14px;font-weight:700;margin-bottom:14px}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;font-size:11px;color:#666;margin-bottom:12px}}
.dot{{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:4px;vertical-align:middle}}
/* Canvas wrapper — chiều cao cố định để Chart.js không chạy vô hạn */
.chart-wrap{{position:relative;height:260px;width:100%}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
thead tr:first-child th{{background:#f5f5f7;padding:8px 10px;font-size:11px;font-weight:600;color:#555}}
thead tr:last-child th{{padding:7px 10px;font-size:11px;color:#888;font-weight:500;background:#fafafa}}
td{{padding:10px 10px;border-top:1px solid #f0f0f2}}
.metric-name{{font-weight:600;color:#333}}
.divider{{border-left:2px solid #e0e0e5}}
tbody tr:hover{{background:#fafafe}}
.badge{{display:inline-block;background:#e1f5ee;color:#0F6E56;padding:2px 7px;border-radius:20px;font-size:10px;font-weight:600;margin-left:6px}}
.pr-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.pr-card{{background:#f8f8fa;border-radius:8px;padding:12px 14px}}
.pr-card h3{{font-size:12px;font-weight:700;margin-bottom:8px;color:#444}}
.pr-row{{display:flex;justify-content:space-between;font-size:12px;padding:4px 0;border-bottom:1px solid #eee}}
.pr-row:last-of-type{{border-bottom:none}}
.pr-note{{margin-top:8px;font-size:11px;color:#888}}
.high{{color:#1D9E75;font-weight:600}} .low{{color:#D85A30;font-weight:600}}
.summary-box{{border-radius:8px;padding:14px 16px;font-size:13px;line-height:1.75;color:#2c2c2e;white-space:pre-wrap;word-break:break-word}}
.plabel{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px}}
.gap{{height:14px}}
</style>
</head>
<body>
<h1>ROUGE Evaluation Report</h1>
<p class="sub">Pipeline: Crawl → Preprocessing → K-Means Clustering → ViT5 Summarization</p>

<div class="cards">
  <div class="card"><div class="lbl">ROUGE-1 F1 tốt nhất</div><div class="val">{fmt(best_r1)}</div><div class="tag">▲ Từ đơn — {winner}</div></div>
  <div class="card"><div class="lbl">ROUGE-2 F1 tốt nhất</div><div class="val">{fmt(best_r2)}</div><div class="tag">▲ Cặp từ — {winner}</div></div>
  <div class="card"><div class="lbl">ROUGE-L F1 tốt nhất</div><div class="val">{fmt(best_rl)}</div><div class="tag">▲ Mạch lạc — {winner}</div></div>
</div>

<div class="charts">
  <div class="section">
    <h2>So sánh F1 Score</h2>
    <div class="legend">
      <span><span class="dot" style="background:#185FA5"></span>TF-IDF + K-Means + ViT5</span>
      <span><span class="dot" style="background:#1D9E75"></span>viBERT + K-Means + ViT5</span>
      <span><span class="dot" style="background:#bbb;border:1px dashed #888"></span>Ngưỡng tốt</span>
    </div>
    <div class="chart-wrap"><canvas id="f1Chart"></canvas></div>
  </div>
  <div class="section">
    <h2>Precision vs Recall theo metric</h2>
    <div class="legend">
      <span><span class="dot" style="background:#185FA5"></span>P TF-IDF</span>
      <span><span class="dot" style="background:#93C5FD"></span>R TF-IDF</span>
      <span><span class="dot" style="background:#1D9E75"></span>P viBERT</span>
      <span><span class="dot" style="background:#6EE7B7"></span>R viBERT</span>
    </div>
    <div class="chart-wrap"><canvas id="prChart"></canvas></div>
  </div>
</div>

<div class="section">
  <h2>Bảng chi tiết đầy đủ</h2>
  <table>
    <thead>
      <tr>
        <th style="text-align:left">Metric</th>
        <th colspan="3">TF-IDF + K-Means + ViT5 <span class="badge">{winner_tfidf}</span></th>
        <th colspan="3" class="divider">viBERT + K-Means + ViT5 <span class="badge">{winner_bert}</span></th>
      </tr>
      <tr>
        <th style="text-align:left"></th>
        <th>F1</th><th>Precision</th><th>Recall</th>
        <th class="divider">F1</th><th>Precision</th><th>Recall</th>
      </tr>
    </thead>
    <tbody>{score_rows_html}</tbody>
  </table>
</div>

<div class="section">
  <h2>Phân tích Precision / Recall</h2>
  <div class="pr-grid">
    <div class="pr-card"><h3>TF-IDF + K-Means + ViT5</h3>{pr_rows(tfidf)}</div>
    <div class="pr-card"><h3>viBERT + K-Means + ViT5</h3>{pr_rows(bert)}</div>
  </div>
</div>

<div class="section">
  <h2>Bản tóm tắt sinh ra &amp; Ground Truth</h2>
  <div class="plabel" style="color:#185FA5">TF-IDF + K-Means + ViT5</div>
  <div class="summary-box" style="background:#EFF6FF;border-left:3px solid #185FA5">{summary_tfidf}</div>
  <div class="gap"></div>
  <div class="plabel" style="color:#1D9E75">viBERT + K-Means + ViT5</div>
  <div class="summary-box" style="background:#ECFDF5;border-left:3px solid #1D9E75">{summary_bert}</div>
  <div class="gap"></div>
  <div class="plabel" style="color:#7C3AED">Ground Truth (tham chiếu)</div>
  <div class="summary-box" style="background:#F5F3FF;border-left:3px solid #7C3AED">{ground_truth_text}</div>
</div>

<script>
(function() {{
  // Nhúng Chart.js từ CDN, render sau khi tải xong
  var s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js';
  s.onload = function() {{
    var labels = ['ROUGE-1','ROUGE-2','ROUGE-L'];
    var opts = function(extra) {{
      return Object.assign({{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{display: false}},
          tooltip: {{callbacks: {{label: function(c) {{ return c.dataset.label+': '+parseFloat(c.parsed.y).toFixed(4); }}}}}}
        }},
        scales: {{
          y: {{min:0, max:1, ticks:{{callback: function(v){{return v.toFixed(1);}} }}, grid:{{color:'rgba(0,0,0,0.06)'}}}},
          x: {{ticks:{{autoSkip:false}}, grid:{{display:false}}}}
        }}
      }}, extra||{{}});
    }};

    new Chart(document.getElementById('f1Chart'), {{
      data: {{
        labels: labels,
        datasets: [
          {{type:'bar', label:'TF-IDF', data:{js_arr(f1_tfidf)}, backgroundColor:'#185FA5', borderRadius:5}},
          {{type:'bar', label:'viBERT', data:{js_arr(f1_bert)},  backgroundColor:'#1D9E75', borderRadius:5}},
          {{type:'line', label:'Ngưỡng tốt', data:[0.45,0.20,0.40], borderColor:'#aaa', borderDash:[5,5], pointRadius:0, borderWidth:1.5, fill:false}}
        ]
      }},
      options: opts()
    }});

    new Chart(document.getElementById('prChart'), {{
      type: 'bar',
      data: {{
        labels: labels,
        datasets: [
          {{label:'P TF-IDF', data:{js_arr(pr_tfidf)}, backgroundColor:'#185FA5', borderRadius:4}},
          {{label:'R TF-IDF', data:{js_arr(re_tfidf)}, backgroundColor:'#93C5FD', borderRadius:4}},
          {{label:'P viBERT', data:{js_arr(pr_bert)},  backgroundColor:'#1D9E75', borderRadius:4}},
          {{label:'R viBERT', data:{js_arr(re_bert)},  backgroundColor:'#6EE7B7', borderRadius:4}}
        ]
      }},
      options: opts()
    }});
  }};
  s.onerror = function() {{
    document.querySelectorAll('.chart-wrap').forEach(function(el) {{
      el.innerHTML = '<p style="color:#888;padding:20px;text-align:center">⚠ Không tải được Chart.js (cần kết nối internet)</p>';
    }});
  }};
  document.head.appendChild(s);
}})();
</script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Đã tạo file visualize: {output_html}")


if __name__ == "__main__":
    run_evaluation()