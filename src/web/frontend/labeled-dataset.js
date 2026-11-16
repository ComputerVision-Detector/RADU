// Check OCR against the team's hand-labelled note pages (Hugging Face images + label snapshot).
// Loads a page as the source image through the normal file input, then scores the analysed
// document on the server and shows label boxes beside the OCR output.
(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const select = $("#labeled-page"), loadButton = $("#labeled-load"),
    evaluateButton = $("#labeled-evaluate"), summary = $("#labeled-summary");
  if (!select) return;
  const splitNames = { test: "test", validation: "validation", train: "train · 한국어 LoRA 학습에 사용" };
  let loadedPage = null, sessionId = null, lastResult = null, programmatic = false;
  const esc = (value) => String(value == null ? "" : value).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const percent = (value) => value == null ? "–" : (value * 100).toFixed(1) + "%";
  const say = (message, error = false) => {
    summary.textContent = message;
    summary.classList.toggle("labeled-error", error);
  };
  async function api(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
      let detail = response.statusText;
      try { detail = (await response.json()).detail || detail; } catch (e) { /* not JSON */ }
      throw new Error(detail);
    }
    return response;
  }

  async function listPages() {
    try {
      const data = await (await api("/api/labeled-dataset")).json();
      for (const split of ["test", "validation", "train"]) {
        const group = document.createElement("optgroup");
        group.label = splitNames[split];
        for (const page of data.pages.filter((item) => item.split === split)) {
          const option = document.createElement("option");
          option.value = page.id;
          option.textContent = page.id + " · 라벨 " + page.regions + "줄" +
            (page.formula_regions ? " (수식 " + page.formula_regions + ")" : "") + (page.on_hold ? " · 검수 보류" : "");
          group.append(option);
        }
        select.append(group);
      }
      select.title = data.dataset + " @ " + data.revision.slice(0, 8) + " · 라벨 " + data.labels_branch + " @ " + data.labels_commit.slice(0, 8);
      say(data.pages.length + "쪽 · " + data.dataset);
    } catch (error) {
      say("라벨 데이터셋을 불러오지 못했습니다: " + error.message, true);
    }
  }

  async function loadPage() {
    const page = select.value;
    if (!page) return;
    loadButton.disabled = true;
    say(page + " 원본 이미지를 불러오는 중…");
    try {
      const blob = await (await api("/api/labeled-dataset/" + page + "/image")).blob();
      const transfer = new DataTransfer();
      transfer.items.add(new File([blob], page + ".png", { type: "image/png" }));
      const input = $("#image-file");
      input.files = transfer.files;
      programmatic = true;
      input.dispatchEvent(new Event("change", { bubbles: true }));
      programmatic = false;
      loadedPage = page;
      sessionId = null;
      lastResult = null;
      evaluateButton.disabled = true;
      say(page + " 준비됨. OCR 모드를 고르고 '이미지 분석'을 누르면 정답과 대조합니다.");
    } catch (error) {
      say("이미지를 불러오지 못했습니다: " + error.message, true);
    } finally {
      loadButton.disabled = !select.value;
    }
  }

  async function evaluate(open) {
    if (!loadedPage || !sessionId) return;
    evaluateButton.disabled = true;
    say(loadedPage + " 정답 대조 중…");
    try {
      const url = "/api/sessions/" + sessionId + "/evaluate-labels?page=" + loadedPage;
      lastResult = await (await api(url, { method: "POST" })).json();
      const r = lastResult, d = r.detection;
      say(r.page.id + " (" + r.page.split + ") · 전체 영역 CER " + percent(r.all_regions.cer_micro) +
        " · 일치 영역 CER " + percent(r.matched_regions.cer_micro) + " · 검출 " + d.matched + "/" + d.references +
        " · 정답 밖 예측 " + (d.predictions - d.matched));
      if (open) showResult();
    } catch (error) {
      lastResult = null;
      say("정답 대조 실패: " + error.message, true);
    } finally {
      evaluateButton.disabled = false;
    }
  }

  function showResult() {
    const r = lastResult;
    if (!r) return;
    const { width, height } = r.image, d = r.detection;
    const rect = (box, kind, title) => '<rect class="labeled-box ' + kind + '" x="' + box[0] + '" y="' + box[1] + '" width="' +
      (box[2] - box[0]) + '" height="' + (box[3] - box[1]) + '" vector-effect="non-scaling-stroke"><title>' + esc(title) + "</title></rect>";
    const boxes = r.regions.map((g) => rect(g.bbox, g.held ? "held" : g.prediction == null ? "missed" : g.errors ? "wrong" : "exact",
      g.text + (g.prediction == null ? "" : " → " + g.prediction))).join("") +
      r.extra_predictions.map((p) => rect(p.bbox, "extra", "정답 밖 예측: " + p.text)).join("");
    const rows = r.regions.map((g) => '<tr class="' + (g.held ? "held" : g.prediction == null ? "missed" : g.errors ? "wrong" : "exact") + '"><td>' +
      esc(g.type) + "</td><td>" + esc(g.text) + "</td><td>" +
      (g.held ? "<i>검수 보류 · 채점 제외</i>" : g.prediction == null ? "<i>검출 안 됨</i>" : esc(g.prediction)) + "</td><td>" +
      (g.cer == null ? "–" : percent(g.cer)) + "</td></tr>").join("");
    const types = Object.entries(r.by_type).map(([kind, m]) => esc(kind) + " " + m.samples + "개 " + percent(m.cer_micro)).join(" · ");
    const notes = [];
    if (r.page.split === "train") notes.push("이 페이지는 한국어 LoRA 학습에 사용됐습니다. 점수를 일반화 성능으로 보지 마세요.");
    else notes.push("split은 필자 독립성이 확인되지 않은 임시 분할입니다.");
    if (!r.page.eligible || r.page.hold_reasons.length) notes.push("검수 보류 페이지입니다: " + (r.page.hold_reasons.join("; ") || "평가 대상 제외"));
    if (r.preprocess_steps.length) notes.push("전처리가 페이지를 보정해(" + r.preprocess_steps.join(", ") + ") 정답 박스를 같은 변환으로 옮겼습니다.");
    notes.push("라벨에 없는 글씨(미라벨 줄, 뒷면 비침)를 읽은 결과도 '정답 밖 예측'으로 세어 검출 정밀도와 페이지 CER을 낮춥니다.");
    notes.push("라벨은 텍스트 줄만 있습니다. 도식·표 구조와 수식 LaTeX는 정답이 없어 채점하지 않습니다.");
    $("#modal-root").innerHTML = '<div class="modal-backdrop"><div class="modal ocr-comparison labeled-result" role="dialog" aria-modal="true" aria-label="라벨 정답 대조">' +
      '<div class="modal-head"><h2>정답 대조 · ' + esc(r.page.id) + " · " + esc(r.page.split) + '</h2><button class="modal-close" aria-label="닫기">×</button></div>' +
      '<p class="hint">' + esc(r.dataset) + " · 라벨 " + esc(r.labels_commit.slice(0, 8)) + " · 라벨러 " + esc(r.page.annotator || "–") +
      " · OCR " + esc(r.ocr_engine) + " · 구조 " + esc(r.structure_mode) + "</p>" +
      '<div class="labeled-metrics"><div><b>' + percent(r.all_regions.cer_micro) + "</b><span>전체 영역 CER (미검출 포함)</span></div><div><b>" +
      percent(r.matched_regions.cer_micro) + "</b><span>일치 영역 CER (" + d.matched + "개)</span></div><div><b>" +
      percent(r.page_text ? r.page_text.cer_micro : null) + "</b><span>페이지 CER (순서·추가 예측 포함)</span></div><div><b>" +
      percent(d.precision) + " / " + percent(d.recall) + "</b><span>검출 정밀도 / 재현율 (IoU " + r.iou_threshold + ")</span></div><div><b>" +
      percent(r.matched_regions.exact_match_rate) + "</b><span>줄 완전 일치</span></div></div>" +
      '<p class="hint">정답 ' + d.references + "줄 · 예측 " + d.predictions + "줄 · 정답 밖 예측 " + (d.predictions - d.matched) +
      (d.held_references ? " · 보류 정답 " + d.held_references : "") + (types ? " · 유형별: " + types : "") + "</p>" +
      notes.map((note) => '<p class="hint">' + esc(note) + "</p>").join("") +
      '<div class="labeled-grid"><div><p class="labeled-legend"><span class="exact">일치</span><span class="wrong">글자 오류</span><span class="missed">검출 안 됨</span><span class="extra">정답 밖 예측</span><span class="held">보류</span></p>' +
      '<svg role="img" aria-label="정답 박스와 예측" viewBox="0 0 ' + width + " " + height + '"><image href="/api/sessions/' + sessionId +
      '/processed-image" width="' + width + '" height="' + height + '"/>' + boxes + "</svg></div>" +
      '<div class="labeled-table"><table><thead><tr><th>유형</th><th>정답</th><th>예측</th><th>CER</th></tr></thead><tbody>' + rows + "</tbody></table></div></div></div></div>";
    const close = () => { $("#modal-root").innerHTML = ""; };
    $(".modal-close").onclick = close;
    $(".modal-close").focus();
    $(".modal-backdrop").onclick = (event) => { if (event.target === event.currentTarget) close(); };
  }

  select.onchange = () => { loadButton.disabled = !select.value; };
  loadButton.onclick = loadPage;
  evaluateButton.onclick = () => lastResult ? showResult() : evaluate(true);
  // Any other image choice ends the dataset check.
  $("#image-file").addEventListener("change", () => {
    if (programmatic || !loadedPage) return;
    loadedPage = null; sessionId = null; lastResult = null;
    evaluateButton.disabled = true;
    say("다른 이미지를 선택해 정답 대조를 해제했습니다.");
  });
  document.addEventListener("radu:analyzed", (event) => {
    if (!loadedPage || event.detail.sourceFile !== loadedPage + ".png") return;
    sessionId = event.detail.sessionId;
    lastResult = null;
    if (event.detail.hasOcr) evaluate(false);
    else say("OCR 결과가 없어 정답과 대조할 수 없습니다. OCR 모드를 선택해 분석해 주세요.", true);
  });
  listPages();
})();
