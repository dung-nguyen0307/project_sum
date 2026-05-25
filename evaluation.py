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
               else ("⚠ Recall >> Precision: tóm tắt bao phủ rộng nhưng sinh nhiều từ thừa"
                     if get(block,"ROUGE-1","Recall") - get(block,"ROUGE-1","Precision") > 0.15
                     else "✓ Cân bằng Precision và Recall")
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
  :root{{
    --color-background-primary:#fff;--color-background-secondary:#f5f5f7;--color-background-tertiary:#f0f2f5;
    --color-background-info:#e6f1fb;--color-background-success:#eaf3de;--color-background-warning:#faeeda;--color-background-danger:#fcebeb;
    --color-text-primary:#1d1d1f;--color-text-secondary:#6e6e73;--color-text-info:#185FA5;--color-text-success:#3B6D11;--color-text-warning:#854F0B;--color-text-danger:#A32D2D;
    --color-border-tertiary:rgba(0,0,0,0.12);--color-border-secondary:rgba(0,0,0,0.2);--color-border-primary:rgba(0,0,0,0.3);
    --border-radius-md:8px;--border-radius-lg:12px;
  }}
  @media(prefers-color-scheme:dark){{
    :root{{
      --color-background-primary:#1c1c1e;--color-background-secondary:#2c2c2e;--color-background-tertiary:#3a3a3c;
      --color-background-info:#0c3b6e;--color-background-success:#1a3d08;--color-background-warning:#4a2d06;--color-background-danger:#3d1010;
      --color-text-primary:#f5f5f7;--color-text-secondary:#aeaeb2;--color-text-info:#5ba3f5;--color-text-success:#7ec94e;--color-text-warning:#f5c542;--color-text-danger:#f47575;
      --color-border-tertiary:rgba(255,255,255,0.12);--color-border-secondary:rgba(255,255,255,0.2);--color-border-primary:rgba(255,255,255,0.3);
    }}
  }}
body{{font-family:'Noto Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f2f5;color:#1d1d1f;padding:28px 20px}}
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
.summary-box{{border-radius:8px;padding:14px 16px;font-size:13px;line-height:1.75;color:#1a1a1a !important;white-space:pre-wrap;word-break:break-word}}
.qual-table{{width:100%;border-collapse:collapse;font-size:12px;min-width:600px}}
.qual-table thead tr th{{background:var(--color-background-secondary);padding:10px 12px;font-weight:500;font-size:12px;color:var(--color-text-secondary);border-bottom:1px solid var(--color-border-tertiary)}}
.qual-table thead tr th:first-child{{text-align:left}}
.qual-table thead tr th:not(:first-child){{text-align:left;border-left:2px solid var(--color-border-tertiary)}}
.qual-table tbody tr{{border-top:1px solid var(--color-border-tertiary)}}
.qual-table tbody tr:hover{{background:var(--color-background-secondary)}}
.aspect-label{{padding:12px;vertical-align:top;white-space:nowrap}}
.aspect-badge{{display:inline-block;padding:3px 9px;border-radius:20px;font-size:11px;font-weight:500}}
.cell-tfidf{{padding:12px 14px;vertical-align:top;border-left:2px solid #185FA5;background:#f8fbff}}
.cell-bert{{padding:12px 14px;vertical-align:top;border-left:2px solid #1D9E75;background:#f6fdf9}}
.cell-point{{padding:4px 0;line-height:1.6;color:#1a1a1a}}
.cell-point.warn{{color:#7a3800;font-weight:500}}
.cell-point.good{{color:#1a5c2e;font-weight:500}}
.plabel{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px}}
.gap{{height:14px}}
/* Roadmap cards */
.roadmap-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:4px}}
.roadmap-col{{border-radius:10px;overflow:hidden;border:1px solid #e0e0e5}}
.roadmap-col-header{{padding:14px 18px;font-size:14px;font-weight:700;color:#fff}}
.roadmap-item{{padding:12px 18px;border-top:1px solid rgba(0,0,0,0.07);background:#fff}}
.roadmap-item-title{{font-size:13px;font-weight:600;color:#1a1a1a;margin-bottom:2px}}
.roadmap-item-desc{{font-size:12px;color:#555}}
.roadmap-state{{background:#f5f5f7;border-radius:10px;padding:12px 20px;text-align:center;margin-bottom:16px;border:1px solid #e0e0e5}}
.roadmap-state-title{{font-size:14px;font-weight:700;color:#1a1a1a}}
.roadmap-state-sub{{font-size:12px;color:#666;margin-top:3px}}
.roadmap-arrow{{text-align:center;font-size:20px;color:#999;margin:8px 0}}
.roadmap-timeline{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:16px}}
.roadmap-tl-card{{border-radius:8px;padding:10px 14px;border:1px solid}}
.roadmap-tl-badge{{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:12px;margin-bottom:6px}}
.roadmap-tl-items{{font-size:12px;color:#444;line-height:1.7}}
.roadmap-dest{{border-radius:10px;padding:14px 20px;text-align:center;margin-top:16px;background:linear-gradient(135deg,#0d4f3c,#1D9E75);color:#fff}}
.roadmap-dest-title{{font-size:14px;font-weight:700}}
.roadmap-dest-sub{{font-size:12px;opacity:.85;margin-top:3px}}
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

<div class="section">
  <h2>Phân tích định tính</h2>
  <p style="font-size:13px;color:var(--color-text-secondary);margin-bottom:16px">So sánh chi tiết theo 3 khía cạnh ngôn ngữ học giữa 2 pipeline</p>
  <div style="overflow-x:auto">
  <table class="qual-table">
    <thead>
      <tr>
        <th style="width:180px;text-align:left">Khía cạnh</th>
        <th>TF-IDF + K-Means + ViT5</th>
        <th>viBERT + K-Means + ViT5</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="aspect-label">
          <span class="aspect-badge" style="background:var(--color-background-info);color:var(--color-text-info)">Từ ngữ &amp; Cú pháp</span>
          <span style="font-size:11px;display:block;margin-top:4px;color:var(--color-text-secondary)">Lexicon &amp; Syntax</span>
        </td>
        <td class="cell-tfidf">
          <div class="cell-point">✦ ROUGE-1 F1=0.607, Recall=0.831 — bao phủ từ đơn tốt nhờ cơ chế đếm tần suất TF-IDF; các từ khóa chủ đạo (Mỹ, Iran, Hormuz, Trump) được lưu giữ.</div>
          <div class="cell-point warn">⚠ Precision chỉ đạt 0.478 — sinh ra nhiều từ/cụm thừa không có trong ground truth, kéo dài bản tóm tắt vô nghĩa.</div>
          <div class="cell-point warn">⚠ Lỗi cú pháp: ký tự lạ (ÓỚ...Ạ.Ù), mốc thời gian mâu thuẫn (22/4, 22.5, 6/5) và câu bị cắt đứt ở cuối.</div>
        </td>
        <td class="cell-bert">
          <div class="cell-point good">✓ ROUGE-1 F1=0.618, Recall=0.863 — bao phủ từ đơn cao hơn TF-IDF nhờ Contextual Embedding nắm được các từ đồng nghĩa và biến thể.</div>
          <div class="cell-point warn">⚠ Precision=0.481, kém hơn không đáng kể so với TF-IDF — vẫn sinh ra từ thừa nhưng ít hơn.</div>
          <div class="cell-point warn">⚠ Lỗi cú pháp: nhiễu ký tự đặc biệt (ÓÓÓỚ...Ạ.Ù), từ lạc ngữ cảnh (thả heo, tuần tra tự do, f f f. 10-11).</div>
        </td>
      </tr>
      <tr>
        <td class="aspect-label">
          <span class="aspect-badge" style="background:var(--color-background-warning);color:var(--color-text-warning)">Ngữ nghĩa &amp; Sự kiện</span>
          <span style="font-size:11px;display:block;margin-top:4px;color:var(--color-text-secondary)">Semantics &amp; Events</span>
        </td>
        <td class="cell-tfidf">
          <div class="cell-point warn">⚠ ROUGE-2 F1=0.297, ROUGE-L F1=0.289 — chuỗi từ liên tiếp và mạch lạc câu rất thấp, cho thấy các sự kiện bị sắp xếp sai trình tự.</div>
          <div class="cell-point warn">⚠ Hallucination: đưa vào thông tin kinh tế không có trong ground truth (giá dầu 160 USD/ounce, giá xăng 10 USD) — phân cụm TF-IDF gom nhầm bài lạc chủ đề.</div>
          <div class="cell-point warn">⚠ Bỏ sót hoàn toàn các sự kiện ngoại giao cốt lõi: đàm phán Islamabad, vai trò JD Vance, Pakistan làm trung gian.</div>
        </td>
        <td class="cell-bert">
          <div class="cell-point good">✓ ROUGE-2 F1=0.314, ROUGE-L F1=0.300 — chuỗi từ liên tiếp và mạch lạc tốt hơn TF-IDF ~5.7%, phản ánh embedding nắm ngữ nghĩa sâu hơn.</div>
          <div class="cell-point warn">⚠ Nhòe sự kiện (Semantic Blurring): nhắc đến "đàm phán" và "eo biển Hormuz" nhưng chi tiết bị trộn lẫn mốc thời gian sai (23-5, 21-5, 1-6 xen kẽ không logic).</div>
          <div class="cell-point warn">⚠ Bỏ sót vai trò Pakistan làm trung gian và điểm nghẽn hạt nhân — hai nội dung quan trọng nhất trong ground truth.</div>
        </td>
      </tr>
      <tr>
        <td class="aspect-label">
          <span class="aspect-badge" style="background:var(--color-background-success);color:var(--color-text-success)">Bám sát nguồn</span>
          <span style="font-size:11px;display:block;margin-top:4px;color:var(--color-text-secondary)">Source Alignment</span>
        </td>
        <td class="cell-tfidf">
          <div class="cell-point good">✓ Recall cao (R1=0.831, R2=0.407, RL=0.395) — bao quát được nhiều từ và cụm từ từ ground truth hơn mức trung bình.</div>
          <div class="cell-point warn">⚠ Precision thấp hơn Recall rõ rệt ở mọi metric (chênh ~35 điểm %) — tóm tắt quá dài, loãng, chứa nhiều thông tin thừa không liên quan.</div>
          <div class="cell-point warn">⚠ Không đề cập đến: lệnh ngừng bắn tại Hormuz ngày 7-8/5, giá dầu 104 USD/thùng (ground truth), vai trò Thổ Nhĩ Kỳ-Ai Cập-Pakistan.</div>
        </td>
        <td class="cell-bert">
          <div class="cell-point good">✓ Recall cao nhất trong cả 2 pipeline ở mọi metric (R1=0.863, R2=0.439, RL=0.419) — embedding ngữ nghĩa giúp bao phủ rộng hơn.</div>
          <div class="cell-point good">✓ Đề cập đúng ngữ cảnh: căng thẳng Mỹ-Iran, đàm phán, eo biển Hormuz, vấn đề hạt nhân — bám sát chủ đề ground truth tốt hơn TF-IDF.</div>
          <div class="cell-point warn">⚠ Precision=0.481 — tương đương TF-IDF, vẫn sinh nhiều câu thừa; chênh lệch Recall–Precision ~38 điểm % cho thấy bản tóm tắt bị "phình" quá mức.</div>
        </td>
      </tr>
    </tbody>
  </table>
  </div>

  <div style="margin-top:16px;padding:12px 16px;background:var(--color-background-secondary);border-radius:8px;font-size:12px;color:var(--color-text-secondary);line-height:1.7">
    <b style="color:var(--color-text-primary);font-weight:500">Kết luận:</b>
    viBERT nhỉnh hơn TF-IDF ở <i>mọi</i> metric ROUGE (R1: +1.1%, R2: +1.7%, RL: +1.1%), xác nhận embedding ngữ nghĩa giúp bao phủ nội dung tốt hơn.
    Tuy nhiên, <b style="font-weight:500">cả hai pipeline đều có Recall &gt;&gt; Precision (~35–38 điểm %)</b> — bản tóm tắt quá dài và loãng, chứa nhiều thông tin thừa do bước clustering chọn nhầm bài lạc chủ đề.
    Bottleneck chính nằm ở <b style="font-weight:500">bước lọc và ranking bài đại diện</b>, không phải mô hình ViT5: cần cải thiện deduplication và lọc bài theo độ liên quan chủ đề trước khi đưa vào sinh tóm tắt.
  </div>
</div>


<div class="section">
  <h2>Định hướng phát triển</h2>
  <p style="font-size:13px;color:#555;margin-bottom:16px">4 hướng tối ưu và mở rộng pipeline theo thứ tự ưu tiên</p>

  <div class="roadmap-state">
    <div class="roadmap-state-title">📍 Trạng thái hiện tại</div>
    <div class="roadmap-state-sub">1 chủ đề &nbsp;·&nbsp; 3 nguồn báo &nbsp;·&nbsp; TF-IDF + viBERT &nbsp;·&nbsp; ViT5</div>
  </div>
  <div class="roadmap-arrow">↓</div>

  <div class="roadmap-grid">
    <div class="roadmap-col">
      <div class="roadmap-col-header" style="background:linear-gradient(135deg,#3C34A0,#5B52D9)">🧠 1. Nâng chất lượng mô hình</div>
      <div class="roadmap-item"><div class="roadmap-item-title">Fine-tune ViT5 trên domain</div><div class="roadmap-item-desc">Dùng VietNews + bộ báo Mỹ-Iran để fine-tune cho tiếng Việt báo chí</div></div>
      <div class="roadmap-item"><div class="roadmap-item-title">Thay ViT5 bằng Gemini / GPT API</div><div class="roadmap-item-desc">Abstractive tốt hơn, không cần GPU, dễ triển khai</div></div>
      <div class="roadmap-item"><div class="roadmap-item-title">Phân cụm phân cấp (HDBSCAN)</div><div class="roadmap-item-desc">Tự động xác định số cụm K, giảm lỗi chọn nhầm bài đại diện</div></div>
      <div class="roadmap-item"><div class="roadmap-item-title">PhoBERT thay viBERT</div><div class="roadmap-item-desc">Embedding tiếng Việt mạnh hơn, huấn luyện trên corpus lớn hơn</div></div>
    </div>
    <div class="roadmap-col">
      <div class="roadmap-col-header" style="background:linear-gradient(135deg,#085041,#1D9E75)">📦 2. Mở rộng dữ liệu</div>
      <div class="roadmap-item"><div class="roadmap-item-title">Đa chủ đề (multi-topic)</div><div class="roadmap-item-desc">Kinh tế, thể thao, công nghệ — đánh giá khả năng tổng quát hóa</div></div>
      <div class="roadmap-item"><div class="roadmap-item-title">Đa ngôn ngữ (multilingual)</div><div class="roadmap-item-desc">EN → VI cross-lingual summarization</div></div>
      <div class="roadmap-item"><div class="roadmap-item-title">Đa phương tiện (multimodal)</div><div class="roadmap-item-desc">Tóm tắt kèm ảnh, biểu đồ từ bài báo gốc</div></div>
      <div class="roadmap-item"><div class="roadmap-item-title">Thêm nguồn báo</div><div class="roadmap-item-desc">Dân Trí, Zing, BBC Tiếng Việt, VOV, VnExpress…</div></div>
    </div>
    <div class="roadmap-col">
      <div class="roadmap-col-header" style="background:linear-gradient(135deg,#0C447C,#2E7DC5)">🚀 3. Hạ tầng &amp; Sản phẩm</div>
      <div class="roadmap-item"><div class="roadmap-item-title">REST API (FastAPI)</div><div class="roadmap-item-desc">Nhận URL → trả bản tóm tắt JSON, dễ tích hợp vào ứng dụng khác</div></div>
      <div class="roadmap-item"><div class="roadmap-item-title">Dashboard web (Streamlit)</div><div class="roadmap-item-desc">Nhập từ khóa → xem kết quả và biểu đồ ROUGE trực tiếp</div></div>
      <div class="roadmap-item"><div class="roadmap-item-title">Tóm tắt theo thời gian thực</div><div class="roadmap-item-desc">Scheduler cron tự động crawl + push alert khi có tin mới</div></div>
      <div class="roadmap-item"><div class="roadmap-item-title">Docker + CI/CD pipeline</div><div class="roadmap-item-desc">Deploy lên cloud, reproducible, dễ bảo trì</div></div>
    </div>
    <div class="roadmap-col">
      <div class="roadmap-col-header" style="background:linear-gradient(135deg,#6B3A06,#D4840A)">📊 4. Đánh giá nâng cao</div>
      <div class="roadmap-item"><div class="roadmap-item-title">BERTScore + MoverScore</div><div class="roadmap-item-desc">Đánh giá ngữ nghĩa sâu, không chỉ trùng khớp từ khóa bề mặt</div></div>
      <div class="roadmap-item"><div class="roadmap-item-title">Nhiều ground truth reference</div><div class="roadmap-item-desc">3–5 người viết tóm tắt → tính ROUGE trung bình, giảm bias</div></div>
      <div class="roadmap-item"><div class="roadmap-item-title">Human evaluation</div><div class="roadmap-item-desc">Đánh giá Fluency · Coherence · Relevance theo thang Likert 1-5</div></div>
      <div class="roadmap-item"><div class="roadmap-item-title">Ablation study</div><div class="roadmap-item-desc">Tắt từng module (clustering, dedup, filter…) và đo độ ảnh hưởng</div></div>
    </div>
  </div>

  <div class="roadmap-arrow" style="margin-top:16px">↓</div>
  <div class="roadmap-dest">
    <div class="roadmap-dest-title">🎯 Đích cuối: Production-ready Vietnamese News Summarizer</div>
    <div class="roadmap-dest-sub">Đa chủ đề &nbsp;·&nbsp; Đa ngôn ngữ &nbsp;·&nbsp; REST API &nbsp;·&nbsp; Realtime &nbsp;·&nbsp; Đánh giá toàn diện</div>
  </div>

  <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
    <div class="roadmap-tl-card" style="background:#fff5f0;border-color:#f4a37a">
      <div class="roadmap-tl-badge" style="background:#f4a37a;color:#7a2b00">⚡ Ngắn hạn</div>
      <div class="roadmap-tl-items">PhoBERT · Gemini API<br>Dedup cải tiến · BERTScore</div>
    </div>
    <div class="roadmap-tl-card" style="background:#fffbf0;border-color:#e8a820">
      <div class="roadmap-tl-badge" style="background:#e8a820;color:#5a3d00">🕐 Trung hạn</div>
      <div class="roadmap-tl-items">FastAPI · Streamlit<br>Đa chủ đề · Human eval</div>
    </div>
    <div class="roadmap-tl-card" style="background:#f0fdf7;border-color:#5DC9A0">
      <div class="roadmap-tl-badge" style="background:#5DC9A0;color:#083d26">🔭 Dài hạn</div>
      <div class="roadmap-tl-items">Multimodal · Cross-lingual<br>Realtime · Fine-tune ViT5</div>
    </div>
  </div>
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