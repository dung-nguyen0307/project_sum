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
.summary-box{{border-radius:8px;padding:14px 16px;font-size:13px;line-height:1.75;color:var(--color-text-primary);white-space:pre-wrap;word-break:break-word}}
.qual-table{{width:100%;border-collapse:collapse;font-size:12px;min-width:600px}}
.qual-table thead tr th{{background:var(--color-background-secondary);padding:10px 12px;font-weight:500;font-size:12px;color:var(--color-text-secondary);border-bottom:1px solid var(--color-border-tertiary)}}
.qual-table thead tr th:first-child{{text-align:left}}
.qual-table thead tr th:not(:first-child){{text-align:left;border-left:2px solid var(--color-border-tertiary)}}
.qual-table tbody tr{{border-top:1px solid var(--color-border-tertiary)}}
.qual-table tbody tr:hover{{background:var(--color-background-secondary)}}
.aspect-label{{padding:12px;vertical-align:top;white-space:nowrap}}
.aspect-badge{{display:inline-block;padding:3px 9px;border-radius:20px;font-size:11px;font-weight:500}}
.cell-tfidf{{padding:12px 14px;vertical-align:top;border-left:2px solid #185FA5}}
.cell-bert{{padding:12px 14px;vertical-align:top;border-left:2px solid #1D9E75}}
.cell-point{{padding:4px 0;line-height:1.6;color:var(--color-text-primary)}}
.cell-point.warn{{color:var(--color-text-warning)}}
.cell-point.good{{color:var(--color-text-success)}}
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
          <div class="cell-point">✦ Trùng khớp từ đơn rất cao — từ vựng đầu vào (Mỹ, Iran, Hormuz) được lưu giữ nhờ cơ chế đếm tần suất TF-IDF.</div>
          <div class="cell-point warn">⚠ Lỗi cú pháp: ký tự lạ và cấu trúc ngắt câu sai ở đoạn cuối.</div>
          <div class="cell-point warn">⚠ Sai lệch sự kiện: gọi nhân vật lịch sử cũ (Tập Cận Bình, Kim Jong-un) thay vì sự kiện 2026 (JD Vance, Islamabad, Pakistan).</div>
        </td>
        <td class="cell-bert">
          <div class="cell-point warn">⚠ Bỏ qua hệ thuật ngữ ngoại giao-quân sự trong file gốc — từ vựng lệch sang tài chính và dầu khí (giá xăng dầu, nhà máy lọc dầu, FED).</div>
          <div class="cell-point warn">⚠ Lỗi cú pháp: nhiễu loạn chuỗi số và mốc thời gian, sinh ra từ ngữ phi logic (1-1-1900, 112,15 USD).</div>
        </td>
      </tr>
      <tr>
        <td class="aspect-label">
          <span class="aspect-badge" style="background:var(--color-background-warning);color:var(--color-text-warning)">Ngữ nghĩa &amp; Sự kiện</span>
          <span style="font-size:11px;display:block;margin-top:4px;color:var(--color-text-secondary)">Semantics &amp; Events</span>
        </td>
        <td class="cell-tfidf">
          <div class="cell-point warn">⚠ Ảo giác (Hallucination): tái hiện sai hoàn toàn cốt truyện — dữ liệu nói xung đột 2026 nhưng mô hình kéo sự kiện về 2018 (quan hệ Mỹ-Trung-Triều).</div>
          <div class="cell-point warn">⚠ Nguyên nhân: phân cụm TF-IDF bị đánh lừa bởi độ trùng lặp từ khóa bề mặt, gom nhầm bài báo cũ → ngữ cảnh "mớm" sai cho ViT5.</div>
        </td>
        <td class="cell-bert">
          <div class="cell-point warn">⚠ Nhòe sự kiện (Semantic Blurring): làm lu mờ sự kiện đàm phán Islamabad, trộn lẫn sự kiện lịch sử (1999, 2019).</div>
          <div class="cell-point good">✓ Nắm được ngữ nghĩa sâu hơn nhờ Contextual Embedding — bắt được hệ quả kinh tế (giá dầu) từ một vài bài trong file dữ liệu.</div>
        </td>
      </tr>
      <tr>
        <td class="aspect-label">
          <span class="aspect-badge" style="background:var(--color-background-success);color:var(--color-text-success)">Bám sát nguồn</span>
          <span style="font-size:11px;display:block;margin-top:4px;color:var(--color-text-secondary)">Source Alignment</span>
        </td>
        <td class="cell-tfidf">
          <div class="cell-point good">✓ Trích xuất thành công từ khóa chủ đạo từ file JSON.</div>
          <div class="cell-point warn">⚠ Ghép nối sai ngữ cảnh hoàn toàn — tạo ra câu chuyện lịch sử không tồn tại trong bộ dữ liệu gốc.</div>
        </td>
        <td class="cell-bert">
          <div class="cell-point good">✓ Bám sát được một phần ngữ cảnh hệ quả (khủng hoảng năng lượng).</div>
          <div class="cell-point warn">⚠ Bỏ sót ~80% cốt truyện chính: vai trò Pakistan làm trung gian, tình hình eo biển Hormuz, vấn đề hạt nhân.</div>
        </td>
      </tr>
    </tbody>
  </table>
  </div>

  <div style="margin-top:16px;padding:12px 16px;background:var(--color-background-secondary);border-radius:8px;font-size:12px;color:var(--color-text-secondary);line-height:1.7">
    <b style="color:var(--color-text-primary);font-weight:500">Kết luận:</b>
    TF-IDF thắng về ROUGE nhờ trùng khớp từ khóa bề mặt nhưng mắc lỗi hallucination nghiêm trọng.
    viBERT nắm ngữ nghĩa tốt hơn nhưng bị nhiễu dữ liệu lạc chủ đề do embedding không phân biệt được ngữ cảnh báo chí.
    Cả 2 pipeline đều cho thấy bottleneck chính nằm ở <b style="font-weight:500">bước clustering</b> (chọn bài đại diện) chứ không phải mô hình ViT5.
  </div>
</div>


<div class="section">
  <h2>Định hướng phát triển</h2>
  <p style="font-size:13px;color:var(--color-text-secondary);margin-bottom:20px">4 hướng tối ưu và mở rộng pipeline theo thứ tự ưu tiên</p>
  <div style="overflow-x:auto">
<svg width="100%" viewBox="0 0 680 920" role="img" style="" xmlns="http://www.w3.org/2000/svg">
<title style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">Roadmap phát triển pipeline tóm tắt đa văn bản tiếng Việt</title>
<desc style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">Sơ đồ 4 hướng phát triển: chất lượng mô hình, mở rộng dữ liệu, hạ tầng sản phẩm, và đánh giá nâng cao</desc>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>

<!-- Tiêu đề trạng thái hiện tại -->
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="240" y="20" width="200" height="38" rx="8" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="340" y="44" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Trạng thái hiện tại</text>
</g>
<text x="340" y="74" text-anchor="middle" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:auto">1 chủ đề · 3 nguồn · TF-IDF + viBERT · ViT5</text>

<!-- Mũi tên xuống -->
<line x1="340" y1="84" x2="340" y2="108" marker-end="url(#arrow)" stroke="var(--color-text-secondary)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<!-- === HƯỚNG 1: CHẤT LƯỢNG MÔ HÌNH === -->
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="20" y="112" width="300" height="40" rx="8" stroke-width="0.5" style="fill:rgb(60, 52, 137);stroke:rgb(175, 169, 236);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="170" y="132" text-anchor="middle" dominant-baseline="central" style="fill:rgb(206, 203, 246);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">1. Nâng chất lượng mô hình</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="30" y="164" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="170" y="178" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Fine-tune ViT5 trên domain</text>
  <text x="170" y="193" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Dùng VietNews + bộ báo Mỹ-Iran</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="30" y="210" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="170" y="224" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Thay ViT5 bằng Gemini/GPT API</text>
  <text x="170" y="239" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Abstractive tốt hơn, không cần GPU</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="30" y="256" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="170" y="270" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Phân cụm phân cấp (HDBSCAN)</text>
  <text x="170" y="285" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Không cần xác định K trước</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="30" y="302" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="170" y="316" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">PhoBERT thay viBERT</text>
  <text x="170" y="331" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Embedding tiếng Việt mạnh hơn</text>
</g>

<!-- === HƯỚNG 2: MỞ RỘNG DỮ LIỆU === -->
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="360" y="112" width="300" height="40" rx="8" stroke-width="0.5" style="fill:rgb(8, 80, 65);stroke:rgb(93, 202, 165);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="510" y="132" text-anchor="middle" dominant-baseline="central" style="fill:rgb(159, 225, 203);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">2. Mở rộng dữ liệu</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="370" y="164" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="510" y="178" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Đa chủ đề (multi-topic)</text>
  <text x="510" y="193" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Kinh tế, thể thao, công nghệ…</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="370" y="210" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="510" y="224" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Đa ngôn ngữ (multilingual)</text>
  <text x="510" y="239" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">EN → VI cross-lingual summary</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="370" y="256" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="510" y="270" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Đa phương tiện (multimodal)</text>
  <text x="510" y="285" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Tóm tắt kèm ảnh, biểu đồ</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="370" y="302" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="510" y="316" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Thêm nguồn báo</text>
  <text x="510" y="331" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Dân Trí, Zing, BBC Vi, VOV…</text>
</g>

<!-- Đường kẻ phân cách -->
<line x1="20" y1="356" x2="660" y2="356" stroke="var(--color-border-tertiary)" stroke-width="0.5" stroke-dasharray="4 4" style="fill:rgb(0, 0, 0);stroke:rgba(222, 220, 209, 0.15);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-dasharray:4px, 4px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<!-- === HƯỚNG 3: HẠ TẦNG & SẢN PHẨM === -->
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="20" y="370" width="300" height="40" rx="8" stroke-width="0.5" style="fill:rgb(12, 68, 124);stroke:rgb(133, 183, 235);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="170" y="390" text-anchor="middle" dominant-baseline="central" style="fill:rgb(181, 212, 244);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">3. Hạ tầng &amp; sản phẩm</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="30" y="422" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="170" y="436" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">REST API (FastAPI)</text>
  <text x="170" y="451" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Nhận URL → trả bản tóm tắt JSON</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="30" y="468" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="170" y="482" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Dashboard web (Streamlit)</text>
  <text x="170" y="497" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Nhập từ khóa → xem kết quả trực tiếp</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="30" y="514" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="170" y="528" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Tóm tắt theo thời gian thực</text>
  <text x="170" y="543" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Scheduler cron, push alert</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="30" y="560" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="170" y="574" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Docker + CI/CD pipeline</text>
  <text x="170" y="589" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Deploy lên cloud, reproducible</text>
</g>

<!-- === HƯỚNG 4: ĐÁNH GIÁ NÂNG CAO === -->
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="360" y="370" width="300" height="40" rx="8" stroke-width="0.5" style="fill:rgb(99, 56, 6);stroke:rgb(239, 159, 39);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="510" y="390" text-anchor="middle" dominant-baseline="central" style="fill:rgb(250, 199, 117);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">4. Đánh giá nâng cao</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="370" y="422" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="510" y="436" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">BERTScore + MoverScore</text>
  <text x="510" y="451" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Đánh giá ngữ nghĩa, không chỉ từ khóa</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="370" y="468" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="510" y="482" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Nhiều ground truth reference</text>
  <text x="510" y="497" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">3-5 người viết → ROUGE trung bình</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="370" y="514" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="510" y="528" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Human evaluation</text>
  <text x="510" y="543" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Fluency · Coherence · Relevance</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="370" y="560" width="280" height="36" rx="6" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="510" y="574" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Ablation study</text>
  <text x="510" y="589" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Tắt từng module, đo độ ảnh hưởng</text>
</g>

<!-- === ĐÍCH CUỐI === -->
<line x1="340" y1="606" x2="340" y2="632" marker-end="url(#arrow)" stroke="var(--color-text-secondary)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="120" y="636" width="440" height="56" rx="10" stroke-width="0.5" style="fill:rgb(8, 80, 65);stroke:rgb(93, 202, 165);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="340" y="656" text-anchor="middle" dominant-baseline="central" style="fill:rgb(159, 225, 203);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Đích cuối: Production-ready summarizer</text>
  <text x="340" y="676" text-anchor="middle" dominant-baseline="central" style="fill:rgb(93, 202, 165);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Đa chủ đề · Đa ngôn ngữ · API · Realtime · Đánh giá toàn diện</text>
</g>

<!-- Ghi chú ưu tiên -->
<rect x="20" y="712" width="640" height="190" rx="10" fill="var(--color-background-secondary)" stroke="var(--color-border-tertiary)" stroke-width="0.5" style="fill:rgb(38, 38, 36);stroke:rgba(222, 220, 209, 0.15);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="40" y="736" style="fill:rgb(250, 249, 245);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:start;dominant-baseline:auto">Thứ tự ưu tiên gợi ý</text>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"><rect x="40" y="748" width="60" height="22" rx="4" stroke-width="0.5" style="fill:rgb(113, 43, 19);stroke:rgb(240, 153, 123);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="70" y="763" text-anchor="middle" dominant-baseline="central" style="fill:rgb(240, 153, 123);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Ngắn hạn</text></g>
<text x="112" y="763" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:auto">PhoBERT · Gemini API · Dedup cải tiến · BERTScore</text>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"><rect x="40" y="782" width="70" height="22" rx="4" stroke-width="0.5" style="fill:rgb(99, 56, 6);stroke:rgb(239, 159, 39);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="75" y="797" text-anchor="middle" dominant-baseline="central" style="fill:rgb(239, 159, 39);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Trung hạn</text></g>
<text x="122" y="797" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:auto">FastAPI · Streamlit · Đa chủ đề · Human eval</text>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"><rect x="40" y="816" width="60" height="22" rx="4" stroke-width="0.5" style="fill:rgb(8, 80, 65);stroke:rgb(93, 202, 165);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="70" y="831" text-anchor="middle" dominant-baseline="central" style="fill:rgb(93, 202, 165);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Dài hạn</text></g>
<text x="112" y="831" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:auto">Multimodal · Cross-lingual · Realtime · Fine-tune ViT5</text>

<text x="40" y="870" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:auto">Lý do: các mục ngắn hạn sửa điểm yếu hiện tại (embedding, metric), trung hạn</text>
<text x="40" y="888" style="fill:rgb(194, 192, 182);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:auto">mở rộng thành sản phẩm thực, dài hạn mở rộng phạm vi nghiên cứu.</text>
</svg>
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