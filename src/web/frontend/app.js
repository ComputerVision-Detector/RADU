(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) =>
    Array.from(root.querySelectorAll(selector));
  const hybrid = window.HYBRID_REVIEW || null;
  const draftKey = hybrid ? "hybrid.vector-draft." + hybrid.jobId : "radu.local-draft.v1";
  let tableSize = { rows: 3, cols: 3 };
  const state = {
    doc: null,
    docGeneration: 0,
    file: null,
    previewUrl: null,
    selected: null,
    selectedIds: [],
    tool: "select",
    connectFrom: null,
    tab: "properties",
    warningFilter: "all",
    relationFilter: "all",
    zoom: 1,
    fitMode: false,
    viewMode: "overlay",
    spacePan: false,
    pan: null,
    clipboard: [],
    gridSnap: false,
    marquee: null,
    guides: [],
    history: [],
    historyIndex: -1,
    savedDoc: null,
    editRevision: 0,
    autoSaving: false,
    manualSaveQueued: false,
    dirty: false,
    busy: false,
    drag: null,
    draw: null,
    shapeKind: "process",
    styleClipboard: null,
    pointerElementId: null,
    lastElementClick: null,
  };
  let editorTools;
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const safeColor = (value, fallback) =>
    typeof value === "string" && /^#(?:[\da-f]{3}|[\da-f]{4}|[\da-f]{6}|[\da-f]{8})$/i.test(value)
      ? value
      : fallback;
  const esc = (value) =>
    String(value == null ? "" : value).replace(
      /[&<>"']/g,
      (c) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[c]),
    );
  function toast(message, error) {
    const root = $("#toast-root");
    root.innerHTML = '<div class="toast ' + (error ? "error" : "") +
      '" role="alert">' + esc(message) + "</div>";
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => root.replaceChildren(), 3600);
  }
  function status(message, kind) {
    $("#status-text").textContent = message;
    $("#status-dot").className = "status-dot " + (kind || "ready");
  }
  function dirty(value) {
    state.dirty = value;
    $("#save-state").textContent = value
      ? "변경사항 저장 전"
      : state.doc && (state.doc.session_id || state.doc.updated_at)
      ? "저장됨"
      : state.doc
      ? "로컬 문서"
      : "저장되지 않은 문서";
  }
  function scheduleLocalDraft() {
    if (!state.doc) {
      clearTimeout(scheduleLocalDraft.timer);
      clearTimeout(scheduleLocalDraft.serverTimer);
      return;
    }
    clearTimeout(scheduleLocalDraft.timer);
    scheduleLocalDraft.timer = setTimeout(() => {
      try {
        localStorage.setItem(draftKey, JSON.stringify({
          updated_at: new Date().toISOString(),
          document: state.doc,
        }));
        $("#save-state").textContent = "자동 복구 저장됨 · 서버 저장 전";
      } catch (error) {
        console.warn("RADU local draft could not be saved", error);
      }
    }, 500);
    clearTimeout(scheduleLocalDraft.serverTimer);
    scheduleLocalDraft.serverTimer = setTimeout(() => {
      saveDocument({ silent: true, automatic: true });
    }, 1800);
  }
  function commit() {
    if (!state.doc) return;
    RaduModel.normalize(state.doc);
    state.history = state.history.slice(0, state.historyIndex + 1);
    state.history.push(clone(state.doc));
    if (state.history.length > 80) state.history.shift();
    state.historyIndex = state.history.length - 1;
    state.editRevision += 1;
    dirty(JSON.stringify(state.doc) !== JSON.stringify(state.savedDoc));
    scheduleLocalDraft();
    historyButtons();
  }
  function beginDocumentReplacement() {
    state.docGeneration += 1;
    clearTimeout(scheduleLocalDraft.timer);
    clearTimeout(scheduleLocalDraft.serverTimer);
  }
  function resetHistory() {
    state.history = state.doc ? [clone(state.doc)] : [];
    state.historyIndex = state.history.length - 1;
    state.savedDoc = state.doc ? clone(state.doc) : null;
    dirty(false);
    historyButtons();
  }
  function historyButtons() {
    $("#undo").disabled = state.historyIndex <= 0;
    $("#redo").disabled = state.historyIndex < 0 ||
      state.historyIndex >= state.history.length - 1;
  }
  function restore(index) {
    if (index < 0 || index >= state.history.length) return;
    state.historyIndex = index;
    const identity = {session_id: state.doc.session_id, updated_at: state.doc.updated_at, revision: state.doc.revision};
    state.doc = Object.assign(clone(state.history[index]), identity);
    state.editRevision += 1;
    scheduleLocalDraft();
    if (!state.doc.elements.some((e) => e.id === state.selected)) {
      state.selected = null;
    }
    state.selectedIds = state.selectedIds.filter((id) =>
      state.doc.elements.some((e) => e.id === id)
    );
    if (state.selected && !state.selectedIds.includes(state.selected)) {
      state.selectedIds = [state.selected];
    }
    dirty(JSON.stringify(state.doc) !== JSON.stringify(state.savedDoc));
    render();
    historyButtons();
  }
  const undo = () => restore(state.historyIndex - 1),
    redo = () => restore(state.historyIndex + 1);
  async function request(url, options) {
    if (hybrid) {
      const sessionPath = "/api/sessions/" + hybrid.jobId;
      if (url === sessionPath) url = "/api/jobs/" + hybrid.jobId + "/vector";
      else if (url.startsWith("/api/export/")) url = url.replace("/api/export/", "/api/jobs/" + hybrid.jobId + "/vector/export/");
    }
    let response;
    try {
      response = await fetch(url, options || {});
    } catch (e) {
      throw new Error(
        "서버에 연결할 수 없습니다. RADU 서버가 실행 중인지 확인하세요.",
      );
    }
    if (!response.ok) {
      let message = "요청 처리에 실패했습니다.";
      try {
        const data = await response.json();
        message = typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail || data);
      } catch (e) {
        try {
          message = await response.text();
        } catch (ignore) {}
      }
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return response;
  }
  function dimensions() {
    return state.doc && state.doc.image ||
      state.file && state.file.dimensions || { width: 900, height: 650 };
  }
  function setCanvasSize() {
    const size = dimensions(), card = $("#canvas-card");
    const compare = state.viewMode === "compare";
    $("#diagram").style.pointerEvents = state.viewMode === "ocr" ? "none" : "";
    card.classList.toggle("compare-mode", compare);
    card.style.width = size.width * (compare ? 2 : 1) + "px";
    card.style.height = size.height + "px";
    card.style.transform = "scale(" + state.zoom + ")";
    $("#source-image").style.width = compare ? "50%" : "100%";
    $("#diagram").style.left = compare ? "50%" : "0";
    $("#diagram").style.width = compare ? "50%" : "100%";
    $("#zoom-label").textContent = (state.fitMode ? "맞춤 " : "") + Math.round(state.zoom * 100) + "%";
    $("#dimension-label").textContent = size.width + " × " + size.height +
      " px";
  }
  function fitCanvas() {
    const size = dimensions(),
      area = $("#canvas-scroll"),
      fit = Math.min(
      (area.clientWidth - 42) / (size.width * (state.viewMode === "compare" ? 2 : 1)),
        (area.clientHeight - 42) / size.height,
        1,
      );
    state.zoom = Math.max(.12, fit || .45);
    state.fitMode = true;
    setCanvasSize();
    area.scrollTop = 0;
    area.scrollLeft = 0;
  }
  function pointAt(event) {
    const rect = $("#diagram").getBoundingClientRect(), size = dimensions();
    return {
      x: Math.max(
        0,
        Math.min(
          size.width - 1,
          (event.clientX - rect.left) * size.width / rect.width,
        ),
      ),
      y: Math.max(
        0,
        Math.min(
          size.height - 1,
          (event.clientY - rect.top) * size.height / rect.height,
        ),
      ),
    };
  }
  function center(element) {
    return {
      x: (element.bbox[0] + element.bbox[2]) / 2,
      y: (element.bbox[1] + element.bbox[3]) / 2,
    };
  }
  function point(value, fallback) {
    if (Array.isArray(value)) {
      return { x: Number(value[0]), y: Number(value[1]) };
    }
    if (value && typeof value === "object") {
      return { x: Number(value.x), y: Number(value.y) };
    }
    return fallback;
  }
  function kindName(element) {
    if (element.type === "Text") return "텍스트";
    if (element.type === "Arrow") return "화살표";
    if (element.type === "Image") return "이미지";
    if (element.type === "Stroke") return "펜 선";
    return ({
      process: "처리 도형", decision: "판단 도형",
      terminator: "시작·종료 도형", input_output: "입력·출력",
      document: "문서", database: "저장 데이터",
      predefined_process: "서브프로세스", manual_input: "수동 입력",
      preparation: "준비", connector: "연결자", offpage_connector: "페이지 간 연결자", delay: "지연",
    })[element.shape] || "처리 도형";
  }
  function selectTool(tool) {
    if (tool !== "style") state.styleClipboard = null;
    state.tool = tool;
    state.connectFrom = null;
    $$(".tool[data-tool]").forEach((b) =>
      b.classList.toggle("active", b.dataset.tool === tool)
    );
    $("#diagram").style.cursor = tool === "select"
      ? "default"
      : tool === "pan"
      ? "grab"
      : "crosshair";
    if (tool !== "select") {
      status(
        tool === "link"
          ? "연결할 시작 도형을 선택하세요."
          : tool === "pan"
          ? "드래그하여 캔버스를 이동하세요."
          : tool === "arrow"
          ? "화살표의 시작점과 끝점을 그리세요."
          : tool === "style"
          ? "서식을 복사할 원본 요소를 선택하세요."
          : "캔버스에서 요소를 그리세요.",
      );
    }
  }
  function render() {
    RaduModel.normalize(state.doc);
    setCanvasSize();
    renderDiagram();
    renderList();
    renderSummary();
    renderInspector();
    $("#analyze-relations").disabled = state.busy ||
      !state.doc || !state.doc.elements.some((element) =>
        element.type === "Box" || element.type === "Arrow"
      );
    $("#empty-stage").hidden = !!(state.doc || state.previewUrl);
    const image = state.doc && state.doc.image || {},
      source = state.viewMode === "original" || state.viewMode === "compare"
        ? (image.source_url || state.previewUrl || "")
        : state.viewMode === "restored"
        ? ""
        : (image.processed_source_url || image.source_url) ||
          state.previewUrl || "";
    $("#source-image").hidden = !source;
    $("#source-image").src = source;
    const hasOcr = !!(state.doc && state.doc.ocr);
    $("#ocr-view").disabled = !hasOcr;
    $("#radu-view").disabled = !hasOcr;
    $("#ocr-view").setAttribute("aria-pressed", String(state.viewMode === "ocr"));
    $("#radu-view").setAttribute("aria-pressed", String(state.viewMode === "radu"));
    $("#view-mode").textContent = state.viewMode === "compare"
      ? "나란히 비교"
      : state.viewMode === "original"
      ? "원본"
      : state.viewMode === "restored"
      ? "복원"
      : "오버레이";
    $("#upload-name").textContent = state.file && state.file.name ||
      state.doc && state.doc.image && state.doc.image.source_file ||
      "순서도 이미지를 선택하세요";
    $("#upload-hint").textContent = state.file
      ? (state.file.size / 1048576).toFixed(2) + " MB · " +
        (state.file.dimensions
          ? state.file.dimensions.width + " × " + state.file.dimensions.height
          : "미리보기는 분석 후 제공")
      : state.doc && state.doc.image.source_file
      ? "원본·전처리 이미지 비교 가능"
      : state.doc
      ? "빈 캔버스 · " + state.doc.image.width + " × " +
        state.doc.image.height + " px"
      : "";
    $("#project-name").value = state.doc && state.doc.name ||
      state.file && state.file.name.replace(/\.[^.]+$/, "") ||
      "새 순서도 프로젝트";
    editorTools?.onRender();
  }
  function wrapText(value, limit) {
    const result = [];
    for (const paragraph of String(value).split("\n")) {
      let line = "";
      for (const token of paragraph.split(/(\s+)/).filter(Boolean)) {
        if (line.length + token.length <= limit) {
          line += token;
          continue;
        }
        if (line.trim()) result.push(line.trimEnd());
        line = "";
        if (token.length > limit) {
          for (let offset = 0; offset < token.length; offset += limit) {
            const part = token.slice(offset, offset + limit);
            if (part.length === limit) {
              result.push(part);
            } else line = part;
          }
        } else if (!/^\s+$/.test(token)) line = token;
      }
      result.push(line.trimEnd());
    }
    return result.length ? result : [""];
  }
  function layoutText(value, width, height, requestedSize) {
    let fontSize = requestedSize;
    let lines = [""];
    for (;;) {
      const limit = Math.max(1, Math.floor((width - 12) / (fontSize * .9)));
      lines = wrapText(value, limit);
      if (lines.length * fontSize * 1.12 <= Math.max(fontSize, height - 8) || fontSize <= 8) break;
      fontSize -= 1;
    }
    return { fontSize, lines };
  }
  function renderBoxShape(shape, x, y, width, height, fill, stroke, sw) {
    const poly = (pairs) => pairs.map(([px, py]) => px + "," + py).join(" ");
    let body;
    if (shape === "decision") body = '<polygon points="' + poly([[x+width/2,y],[x+width,y+height/2],[x+width/2,y+height],[x,y+height/2]]) + '"/>';
    else if (shape === "terminator") body = '<rect x="' + x + '" y="' + y + '" width="' + width + '" height="' + height + '" rx="' + height/2 + '"/>';
    else if (shape === "input_output") { const k = Math.min(width*.16,height*.35); body = '<polygon points="' + poly([[x+k,y],[x+width,y],[x+width-k,y+height],[x,y+height]]) + '"/>'; }
    else if (shape === "document") body = '<path d="M'+x+' '+y+'H'+(x+width)+'V'+(y+height*.78)+' Q'+(x+width*.75)+' '+(y+height*1.08)+' '+(x+width/2)+' '+(y+height*.86)+' Q'+(x+width*.25)+' '+(y+height*.66)+' '+x+' '+(y+height*.9)+'Z"/>';
    else if (shape === "database") body = '<path d="M'+x+' '+(y+height*.14)+' C'+x+' '+(y-height*.08)+' '+(x+width)+' '+(y-height*.08)+' '+(x+width)+' '+(y+height*.14)+' V'+(y+height*.84)+' C'+(x+width)+' '+(y+height*1.08)+' '+x+' '+(y+height*1.08)+' '+x+' '+(y+height*.84)+'Z M'+x+' '+(y+height*.14)+' C'+x+' '+(y+height*.38)+' '+(x+width)+' '+(y+height*.38)+' '+(x+width)+' '+(y+height*.14)+' M'+x+' '+(y+height*.5)+' C'+x+' '+(y+height*.74)+' '+(x+width)+' '+(y+height*.74)+' '+(x+width)+' '+(y+height*.5)+'"/>';
    else if (shape === "predefined_process") body = '<rect x="'+x+'" y="'+y+'" width="'+width+'" height="'+height+'" rx="3"/><path d="M'+(x+width*.16)+' '+y+'V'+(y+height)+' M'+(x+width*.84)+' '+y+'V'+(y+height)+'"/>';
    else if (shape === "manual_input") body = '<polygon points="'+poly([[x,y+height*.2],[x+width,y],[x+width,y+height],[x,y+height]])+'"/>';
    else if (shape === "preparation") body = '<polygon points="'+poly([[x+width*.2,y],[x+width*.8,y],[x+width,y+height/2],[x+width*.8,y+height],[x+width*.2,y+height],[x,y+height/2]])+'"/>';
    else if (shape === "offpage_connector") body = '<polygon points="'+poly([[x,y],[x+width,y],[x+width,y+height*.6],[x+width/2,y+height],[x,y+height*.6]])+'"/>';
    else if (shape === "connector") body = '<ellipse cx="'+(x+width/2)+'" cy="'+(y+height/2)+'" rx="'+(width/2)+'" ry="'+(height/2)+'"/>';
    else if (shape === "delay") body = '<path d="M'+x+' '+y+'H'+(x+width/2)+' A'+(height/2)+' '+(height/2)+' 0 0 1 '+(x+width/2)+' '+(y+height)+' H'+x+'Z"/>';
    else body = '<rect x="'+x+'" y="'+y+'" width="'+width+'" height="'+height+'" rx="'+Math.max(4,height*.1)+'"/>';
    return '<g class="element-shape shape-'+shape+'" fill="'+fill+'" stroke="'+stroke+'" stroke-width="'+sw+'" stroke-linejoin="round">'+body+'</g>';
  }
  function shapeIcon(shape) {
    const markup = renderBoxShape(shape, 3, 3, 30, 20, "#eef0ff", "#5368e8", 1.8);
    return '<svg viewBox="0 0 36 26" aria-hidden="true" focusable="false">' + markup + "</svg>";
  }
  function renderDiagram() {
    const svg = $("#diagram"), doc = state.doc;
    if (!doc) {
      svg.innerHTML = "";
      return;
    }
    const w = doc.image.width, h = doc.image.height;
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);
    svg.setAttribute("width", w);
    svg.setAttribute("height", h);
    if (state.viewMode === "original") {
      svg.innerHTML = "";
      return;
    }
    let out =
        '<defs><marker id="radu-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0L9 4.5L0 9Z" fill="#5268e8"/></marker><marker id="radu-warn-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0L9 4.5L0 9Z" fill="#d2873d"/></marker></defs>',
      map = new Map(doc.elements.map((e) => [e.id, e]));
    (state.viewMode === "ocr" ? [] : doc.relations || []).forEach((rel) => {
      if (rel.type !== "flows_to") return;
      const arrow = map.get(rel.arrow_id);
      if (arrow && arrow.start && arrow.end) return;
      const a = map.get(rel.source), b = map.get(rel.target);
      if (!a || !b || !RaduModel.visible(doc,a) || !RaduModel.visible(doc,b)) return;
      const p = edgePoint(a, center(b)), q = edgePoint(b, center(a)), review = rel.status === "review";
      out += '<path d="M ' + p.x + " " + p.y + " L " + q.x + " " + q.y +
        '" fill="none" stroke="' + (review ? "#d2873d" : "#5268e8") +
        '" stroke-width="' + Math.max(2, w * .0015) + '" ' +
        (review ? 'stroke-dasharray="8 6"' : "") + ' marker-end="url(#' +
        (review ? "radu-warn-arrow" : "radu-arrow") + ')" opacity=".9"/>';
    });
    doc.elements.filter(e => RaduModel.visible(doc,e) &&
      (state.viewMode !== "ocr" ||
        e.type === "Text" && e.source === doc.ocr?.engine)).forEach((e) => {
      const b = e.bbox,
        x = b[0],
        y = b[1],
        ew = Math.max(2, b[2] - x),
        eh = Math.max(2, b[3] - y),
        style = e.style || {},
        selected = state.selectedIds.includes(e.id) || e.id === state.selected,
        stroke = selected
          ? "#5368e8"
          : safeColor(style.stroke, e.type === "Arrow"
          ? "#199574"
          : e.type === "Text"
          ? "#4b8bc5"
          : "#de8b48"),
        sw = Math.min(30, Math.max(1, Number(style.stroke_width) || 2)),
        fill = safeColor(style.fill, "#ffffff");
      out += '<g class="diagram-element" data-id="' + esc(e.id) +
        '" tabindex="' + (state.viewMode === "ocr" ? "-1" : "0") +
        '" role="' + (state.viewMode === "ocr" ? "img" : "button") + '" aria-label="' +
        esc(kindName(e) + " " + (e.text || e.id)) + '" style="cursor:' +
        (state.tool === "select" ? "move" : "pointer") + '" opacity="' + (style.opacity ?? 1) + '"' +
        (e.rotation ? ' transform="rotate(' + Number(e.rotation) + " " +
          (x + ew / 2) + " " + (y + eh / 2) + ')"' : "") + '>';
      if (e.type === "Image") {
        out += '<image href="'+esc(e.image_data)+'" x="'+x+'" y="'+y+'" width="'+ew+'" height="'+eh+'" preserveAspectRatio="none"/>';
      } else if(e.type === "Stroke") {
        out += '<polyline points="'+(e.points||[]).map(p=>p.x+','+p.y).join(' ')+'" fill="none" stroke="'+stroke+'" stroke-width="'+sw+'" stroke-linecap="round" stroke-linejoin="round"/>';
      } else if (e.type === "Arrow") {
        let a = point(e.start, { x: x, y: y }),
          q = point(e.end, { x: b[2], y: b[3] });
        const route = e.route || "straight",
          path = route === "orthogonal"
            ? "M " + a.x + " " + a.y + " H " + ((a.x+q.x)/2) + " V " + q.y + " H " + q.x
            : route === "curved"
            ? "M " + a.x + " " + a.y + " C " + ((a.x+q.x)/2) + " " + a.y + " " + ((a.x+q.x)/2) + " " + q.y + " " + q.x + " " + q.y
            : RaduModel.jumpPath({start:a,end:q},doc.elements.slice(0,doc.elements.indexOf(e)).filter(item=>item.type==="Arrow"&&RaduModel.visible(doc,item)),doc.canvas?.line_jump||"none"),
          dash = style.line_style === "dashed" ? ' stroke-dasharray="10 7"'
            : style.line_style === "dotted" ? ' stroke-dasharray="2 7"' : "";
        const markerKey="radu-marker-"+doc.elements.indexOf(e),
          marker=(kind,start)=>kind==="none" ? "" : '<marker id="'+markerKey+(start?"-start":"-end")+'" markerWidth="10" markerHeight="10" refX="'+(start?2:8)+'" refY="5" orient="'+(start?"auto-start-reverse":"auto")+'" markerUnits="userSpaceOnUse"><path d="'+(kind==="diamond"?"M1 5L5 1L9 5L5 9Z":"M1 1L9 5L1 9Z")+'" fill="'+stroke+'"/></marker>',
          startStyle=style.marker_start||"none",endStyle=style.marker_end||"arrow";
        out+='<defs>'+marker(startStyle,true)+marker(endStyle,false)+'</defs>';
        out += '<path class="element-shape" d="' + path +
          '" fill="none" stroke="' + stroke +
          '" stroke-width="' + (sw * 2) +
          '" stroke-linecap="round"' + dash + (startStyle!=="none"?' marker-start="url(#'+markerKey+'-start)"':'') + (endStyle!=="none"?' marker-end="url(#'+markerKey+'-end)"':'') + '/><path d="' +
          path +
          '" fill="none" stroke="transparent" stroke-width="' +
          Math.max(18, sw * 7) + '"/>';
        const rel = doc.relations.find((item) => item.arrow_id === e.id),
          label = e.text ?? (rel && rel.label);
        if (label) {
          const lx = (a.x + q.x) / 2, ly = (a.y + q.y) / 2 - 7,
            lines = wrapText(label, 24);
          out += '<text x="' + lx + '" y="' + (ly - (lines.length - 1) * 9) +
            '" text-anchor="middle" fill="#334057" font-size="16" pointer-events="none">';
          lines.forEach((line, index) => out += '<tspan x="' + lx + '" dy="' + (index ? 18 : 0) + '">' + esc(line) + '</tspan>');
          out += "</text>";
        }
      } else if (e.type === "Text") {
        out += '<rect class="element-shape" x="' + x + '" y="' + y +
          '" width="' + ew + '" height="' + eh + '" rx="3" fill="' +
          (selected ? "#eef2ff" : hybrid && state.viewMode === "overlay" ? "#fffffff2" : "#54a1d10c") + '" stroke="' + stroke +
          '" stroke-width="' + sw + '" ' +
          (selected ? "" : 'stroke-dasharray="5 4"') + "/>";
      } else {
        out += renderBoxShape(e.shape || "process", x, y, ew, eh,
          selected ? "#5268e817" : fill, stroke, sw);
      }
      if (e.type === "Box" || e.type === "Text") {
        const label = e.text ?? e.shape_label ?? (e.source ? "" : kindName(e)),
          textLayout = layoutText(label, ew, eh, Number(style.font_size) || Math.max(11, Math.min(20, eh * .22))),
          fontSize = textLayout.fontSize, lines = textLayout.lines,
          cx = style.text_align === "left" || e.type === "Text" && !style.text_align
            ? x + 6 : style.text_align === "right" ? x + ew - 6 : x + ew / 2,
          anchor = style.text_align === "left" || e.type === "Text" && !style.text_align
            ? "start" : style.text_align === "right" ? "end" : "middle",
          firstY = y + eh / 2 - (lines.length - 1) * fontSize * .56;
        out += '<text x="' + cx + '" y="' + firstY + '" text-anchor="' +
          anchor + '" fill="' + safeColor(style.text_color, "#334057") + '" font-size="' + fontSize +
          '" font-family="' + esc(style.font_family || "system-ui,sans-serif") +
          '" font-weight="'+(style.bold?"700":"400")+'" font-style="'+(style.italic?"italic":"normal")+'" pointer-events="none">';
        lines.forEach((line, i) => {
          out += '<tspan x="' + cx + '" y="' + (firstY + i * fontSize * 1.12) +
            '">' + esc(line) + "</tspan>";
        });
        out += "</text>";
      }
      if (selected && !RaduModel.locked(doc,e) && state.selectedIds.length <= 1 && e.type === "Arrow") {
        const a = point(e.start, { x: x, y: y }),
          q = point(e.end, { x: b[2], y: b[3] });
        [["start", a], ["end", q]].forEach(([name, p]) => {
          out += '<circle class="resize-handle arrow-handle" data-handle="' +
            name + '" cx="' + p.x + '" cy="' + p.y + '" r="' +
            Math.max(7, w * .004) + '" fill="white" stroke="#5368e8" stroke-width="' +
            Math.max(2, w * .0018) + '"/>';
        });
      } else if (selected && !RaduModel.locked(doc,e) && state.selectedIds.length <= 1) {
        const handles = {
          nw: [x, y],
          n: [x + ew / 2, y],
          ne: [x + ew, y],
          e: [x + ew, y + eh / 2],
          se: [x + ew, y + eh],
          s: [x + ew / 2, y + eh],
          sw: [x, y + eh],
          w: [x, y + eh / 2],
        };
        Object.entries(handles).forEach(([name, p]) => {
          out += '<circle class="resize-handle" data-handle="' + name +
            '" cx="' + p[0] + '" cy="' + p[1] + '" r="' +
            Math.max(6, w * .004) +
            '" fill="white" stroke="#5368e8" stroke-width="' +
            Math.max(2, w * .0018) + '"/>';
        });
      }
      if (e.type === "Box" && (state.tool === "arrow" || state.tool === "link")) {
        const ports = [[x + ew / 2, y], [x + ew, y + eh / 2], [x + ew / 2, y + eh], [x, y + eh / 2]];
        ports.forEach(([px, py]) => {
          out += '<circle class="connection-port" cx="' + px + '" cy="' + py + '" r="' + Math.max(5, w * .0035) + '" fill="#fff" stroke="#5368e8" stroke-width="' + Math.max(2, w * .0018) + '" pointer-events="none"/>';
        });
      }
      out += "</g>";
    });
    if(state.selectedIds.length>1 && state.tool==="select") {
      const items=state.doc.elements.filter(e=>state.selectedIds.includes(e.id)&&RaduModel.visible(doc,e)&&!RaduModel.locked(doc,e));
      if(items.length>1){
        const b=RaduModel.bounds(items),cx=(b[0]+b[2])/2,cy=(b[1]+b[3])/2;
        const handles={nw:[b[0],b[1]],n:[cx,b[1]],ne:[b[2],b[1]],e:[b[2],cy],se:[b[2],b[3]],s:[cx,b[3]],sw:[b[0],b[3]],w:[b[0],cy]};
        out+='<g class="diagram-element group-selection" data-id="'+esc(state.selected)+'"><rect x="'+b[0]+'" y="'+b[1]+'" width="'+(b[2]-b[0])+'" height="'+(b[3]-b[1])+'" fill="none" stroke="#5368e8" stroke-width="2" stroke-dasharray="7 5" pointer-events="none"/>';
        Object.entries(handles).forEach(([name,p])=>out+='<circle class="resize-handle" data-handle="'+name+'" cx="'+p[0]+'" cy="'+p[1]+'" r="'+Math.max(7,w*.004)+'" fill="white" stroke="#5368e8" stroke-width="2"/>');
        out+='</g>';
      }
    }
    (state.guides || []).forEach((guide) => {
      out += guide.axis === "x"
        ? '<line class="alignment-guide" x1="' + guide.value + '" y1="0" x2="' + guide.value + '" y2="' + h + '" pointer-events="none"/>'
        : '<line class="alignment-guide" x1="0" y1="' + guide.value + '" x2="' + w + '" y2="' + guide.value + '" pointer-events="none"/>';
    });
    if (state.draw) {
      const d = state.draw,
        x = Math.min(d.start.x, d.end.x),
        y = Math.min(d.start.y, d.end.y),
        dw = Math.abs(d.end.x - d.start.x),
        dh = Math.abs(d.end.y - d.start.y);
      out += d.tool === "pen" ? '<polyline points="'+d.points.map(p=>p.x+","+p.y).join(" ")+'" fill="none" stroke="#334057" stroke-width="2"/>' : d.tool === "arrow"
        ? '<path d="M ' + d.start.x + " " + d.start.y + " L " + d.end.x + " " +
          d.end.y +
          '" stroke="#5368e8" stroke-width="4" stroke-dasharray="8 5" marker-end="url(#radu-arrow)"/>'
        : '<rect x="' + x + '" y="' + y + '" width="' + Math.max(1, dw) +
          '" height="' + Math.max(1, dh) +
          '" fill="#5368e818" stroke="#5368e8" stroke-width="3" stroke-dasharray="8 5"/>';
    }
    if (state.marquee) {
      const m = state.marquee,
        x = Math.min(m.start.x, m.end.x),
        y = Math.min(m.start.y, m.end.y),
        width = Math.abs(m.end.x - m.start.x),
        height = Math.abs(m.end.y - m.start.y);
      out += '<rect x="' + x + '" y="' + y + '" width="' + width +
        '" height="' + height +
        '" fill="#5268e81a" stroke="#5268e8" stroke-width="2" stroke-dasharray="7 5" pointer-events="none"/>';
    }
    if (doc.canvas?.ruler && state.viewMode !== "ocr") {
      const step = Math.max(50, Math.ceil(50 / Math.max(.1, state.zoom) / 50) * 50);
      out += '<g class="canvas-rulers" pointer-events="none"><path d="M0 0H'+w+'V20H20V'+h+'H0Z" fill="#f1f3f8" stroke="#b8c0d0" stroke-width="1"/>';
      for (let x = step; x < w; x += step) out += '<path d="M'+x+' 20V11" stroke="#8290a6"/><text x="'+(x+3)+'" y="10" font-size="9" fill="#657187">'+x+'</text>';
      for (let y = step; y < h; y += step) out += '<path d="M20 '+y+'H11" stroke="#8290a6"/><text x="3" y="'+(y-3)+'" font-size="9" fill="#657187">'+y+'</text>';
      out += '</g>';
    }
    svg.innerHTML = out;
    if (state.viewMode === "ocr") return;
    $$(".diagram-element", svg).forEach((g) => {
      g.onpointerdown = elementPointerDown;
      g.onclick = (ev) => {
        if (hybrid && $(".inline-editor")) return;
        if (state.tool !== "link") selectElement(g.dataset.id);
      };
      g.onkeydown = (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          if (state.tool !== "link") selectElement(g.dataset.id);
        }
      };
    });
  }
  function renderList() {
    const root = $("#element-list"),
      allItems = state.doc && state.doc.elements || [],
      viewItems = state.viewMode === "ocr"
        ? allItems.filter((e) => e.type === "Text" && e.source === state.doc.ocr?.engine)
        : allItems,
      filter = $("#element-filter").value,
      items = filter === "all"
        ? viewItems
        : viewItems.filter((e) => e.type === filter);
    $("#element-heading").textContent = items.length +
      (items.length !== viewItems.length ? " / " + viewItems.length : "") + "개";
    if (!items.length) {
      root.className = "empty-list";
      root.textContent = state.doc
        ? "요소가 없습니다. 상단 도구로 직접 추가할 수 있습니다."
        : "분석된 요소가 여기에 표시됩니다.";
      return;
    }
    root.className = "";
    root.innerHTML = items.map((e) =>
      '<button class="element-row ' +
      (state.selectedIds.includes(e.id) || e.id === state.selected
        ? "selected"
        : "") +
      '" data-id="' + esc(e.id) +
      '"><span class="kind-icon ' + (e.type === "Arrow"
        ? "arrow"
        : e.shape === "decision"
        ? "decision"
        : "") +
      '">' + (e.type === "Text"
        ? "T"
        : e.type === "Arrow"
        ? "↗"
        : shapeIcon(e.shape || "process")) +
      '</span><span class="row-main"><span class="row-label">' +
      esc(e.text || kindName(e)) + '</span><span class="row-sub">' + esc(e.id) +
      " · " + esc(kindName(e)) + '</span></span><span class="row-confidence">' +
      (Number.isFinite(e.confidence)
        ? Math.round(e.confidence * 100) + "%"
        : "") +
      "</span></button>"
    ).join("");
    $$(".element-row", root).forEach((row) => {
      row.disabled = state.viewMode === "ocr";
      if (state.viewMode !== "ocr") row.onclick = () => selectElement(row.dataset.id);
    });
  }
  function renderSummary() {
    const allItems = state.doc && state.doc.elements || [],
      items = state.viewMode === "ocr"
        ? allItems.filter((e) => e.type === "Text" && e.source === state.doc.ocr?.engine)
        : allItems,
      rels = state.viewMode === "ocr" ? [] : state.doc && state.doc.relations || [],
      warnings = state.viewMode === "ocr" ? [] : state.doc && state.doc.warnings || [];
    $("#count-text").textContent =
      items.filter((e) => e.type === "Text").length;
    $("#count-shape").textContent =
      items.filter((e) => e.type === "Box").length;
    $("#count-arrow").textContent =
      items.filter((e) => e.type === "Arrow").length;
    $("#count-relation").textContent =
      rels.filter((r) => r.type === "flows_to").length;
    $("#warning-count").textContent = warnings.length +
      rels.filter((r) =>
        r.type === "flows_to" && r.status === "review" &&
        (!r.review_status || r.review_status === "pending")
      ).length;
    $("#analysis-state").textContent = !state.doc ? "대기 중"
      : state.viewMode === "ocr" ? "OCR 기준 보기" : "검토 중";
  }
  function renderInspector() {
    const root = $("#inspector-content"), doc = state.doc;
    $$(".tabs button").forEach((b) =>
      b.classList.toggle("active", b.dataset.tab === state.tab)
    );
    if (!doc) {
      root.innerHTML =
        '<div class="inspector-empty">이미지를 불러오면 텍스트, 유형, 위치와 관계를 검토할 수 있습니다.</div>';
      return;
    }
    if (state.viewMode === "ocr") {
      root.innerHTML =
        '<div class="inspector-empty">OCR 기준 보기에서는 선택한 텍스트 인식 엔진의 결과만 표시합니다. RADU가 검출한 도형·화살표와 관계는 RADU 복원 보기에서 확인할 수 있습니다.</div>';
      return;
    }
    if (state.tab === "warnings") {
      const warnings = doc.warnings || [];
      const visibleWarnings = state.warningFilter === "all" ? warnings : warnings.filter((w) =>
        (w.status || (w.code === "manual_arrow_unlinked" ? "pending" : "warning")) === state.warningFilter
      );
      root.innerHTML = '<div class="review-filter"><label>검수 상태<select class="control" id="warning-filter"><option value="all">전체</option><option value="pending">확인 필요</option><option value="confirmed">확인 완료</option><option value="deferred">보류</option></select></label></div>' + (visibleWarnings.length
        ? visibleWarnings.map((w) =>
          '<button class="warning-row" data-element="' +
          esc(w.element_id || "") +
          '"><span class="warning-badge">!</span><span class="warning-text">' +
          esc(w.message || w.code || "원본을 확인해 주세요.") +
          "</span></button>"
        ).join("")
        : '<div class="inspector-empty">' + (warnings.length ? "선택한 상태의 항목이 없습니다." : "현재 등록된 검토 항목이 없습니다.") + "</div>");
      $("#warning-filter", root).value = state.warningFilter;
      $("#warning-filter", root).onchange = (event) => { state.warningFilter = event.target.value; renderInspector(); };
      $$(".warning-row", root).forEach((row) =>
        row.onclick = () =>
          row.dataset.element && selectElement(row.dataset.element)
      );
      return;
    }
    if (state.tab === "relations") {
      const relations = doc.relations || [],
        rels = state.relationFilter === "all" ? relations : relations.filter((r) => (r.review_status || "pending") === state.relationFilter);
      root.innerHTML = '<div class="review-filter"><label>관계 검수 상태<select class="control" id="relation-filter"><option value="all">전체</option><option value="pending">확인 필요</option><option value="modified">수정됨</option><option value="confirmed">확인 완료</option><option value="deferred">보류</option></select></label></div>' + (rels.length
        ? rels.map((r) =>
          '<div class="relation-row"><div class="relation-top"><span>' +
          esc(
            r.type === "flows_to"
              ? "흐름 연결"
              : r.type === "contains"
              ? "도형 내부 텍스트"
              : "인접 텍스트",
          ) + " · " + esc(r.id) + '</span><span class="relation-status">' +
          (r.status === "review" ? "확인 필요" : "AI 추정") +
          '</span></div><label class="review-choice">검수 상태<select class="control review-state" data-id="' +
          esc(r.id) + '"><option value="pending" ' +
          (!r.review_status || r.review_status === "pending"
            ? "selected"
            : "") +
          '>미확인</option><option value="confirmed" ' +
          (r.review_status === "confirmed" ? "selected" : "") +
          '>확인 완료</option><option value="deferred" ' +
          (r.review_status === "deferred" ? "selected" : "") +
          '>보류</option><option value="modified" ' +
          (r.review_status === "modified" ? "selected" : "") +
          '>수정됨</option></select></label><div class="relation-flow"><select class="control relation-source" data-id="' +
          esc(r.id) + '"><option value="">시작 요소</option>' +
          doc.elements.filter((e) => e.type === "Box").map((e) =>
            '<option value="' + esc(e.id) + '" ' +
            (e.id === r.source ? "selected" : "") + ">" + esc(e.text || e.id) +
            "</option>"
          ).join("") +
          '</select><span>→</span><select class="control relation-target" data-id="' +
          esc(r.id) + '"><option value="">도착 요소</option>' +
          doc.elements.filter((e) =>
            r.type === "flows_to" ? e.type === "Box" : e.type === "Text"
          ).map((e) =>
            '<option value="' + esc(e.id) + '" ' +
            (e.id === r.target ? "selected" : "") + ">" + esc(e.text || e.id) +
            "</option>"
          ).join("") +
          '</select><button class="tool relation-delete" data-id="' +
          esc(r.id) + '" title="관계 삭제">×</button></div></div>'
        ).join("")
        : '<div class="inspector-empty">' + (relations.length ? "선택한 상태의 관계가 없습니다." : "화살표 도구로 연결을 그리거나 연결 도구로 두 도형을 선택하세요.") + "</div>");
      $("#relation-filter", root).value = state.relationFilter;
      $("#relation-filter", root).onchange = (event) => { state.relationFilter = event.target.value; renderInspector(); };
      $$(".relation-source,.relation-target,.review-state", root).forEach(
        (select) =>
          select.onchange = () => {
            const before = JSON.stringify(state.doc),
              r = doc.relations.find((item) => item.id === select.dataset.id);
            if (r) {
              if (select.classList.contains("review-state")) {
                r.review_status = select.value;
              } else {
                if (select.classList.contains("relation-source")) {
                  r.source = select.value || null;
                } else r.target = select.value || null;
                r.status = "review";
                r.review_status = "modified";
                syncArrowRelation(r);
              }
              if (JSON.stringify(state.doc) !== before) commit();
              renderDiagram();
              renderSummary();
              renderInspector();
            }
          },
      );
      $$(".relation-delete", root).forEach((b) =>
        b.onclick = () => {
          const relation = doc.relations.find((r) => r.id === b.dataset.id),
            arrowId = relation && relation.arrow_id;
          doc.relations = doc.relations.filter((r) => r.id !== b.dataset.id);
          if (
            arrowId && !doc.relations.some((r) => r.arrow_id === arrowId)
          ) doc.elements = doc.elements.filter((e) => e.id !== arrowId);
          commit();
          render();
        }
      );
      return;
    }
    const e = doc.elements.find((item) => item.id === state.selected);
    if (!e) {
      root.innerHTML =
        '<div class="inspector-empty">캔버스나 요소 목록에서 편집할 요소를 선택하세요.<br>도구로 새 요소를 추가할 수 있습니다.</div>';
      return;
    }
    const b = e.bbox, st = e.style || {};
    root.innerHTML = '<div class="inspect-head"><div class="inspect-type">' +
      esc(kindName(e)) + '</div><div class="inspect-id">' + esc(e.id) + " · " +
      esc(e.source || "수동 편집") + '</div></div><div class="inspect-body">' +
      (e.type === "Text"
        ? '<label class="field"><span>인식 텍스트</span><textarea class="control area" id="prop-text">' +
          esc(e.text || "") + "</textarea></label>" +
          (e.original_text !== undefined && e.original_text !== e.text
            ? '<div class="original-text">AI 원문: ' + esc(e.original_text) +
              "</div>"
            : "")
        : e.type === "Arrow" ? "" : '<label class="field"><span>요소 라벨</span><input class="control" id="prop-label" value="' +
          esc(e.text || "") +
          '" placeholder="예: 시작, 사용자 입력"></label>') +
      (e.type === "Box"
        ? '<label class="field"><span>도형 유형</span><select class="control" id="prop-shape"><option value="process" ' +
          (!e.shape || e.shape === "process" ? "selected" : "") +
          '>처리</option><option value="terminator" ' +
          (e.shape === "terminator" ? "selected" : "") +
          '>시작·종료</option><option value="decision" ' +
          (e.shape === "decision" ? "selected" : "") +
          '>판단</option><option value="input_output" ' +
          (e.shape === "input_output" ? "selected" : "") +
          '>입력·출력</option><option value="document" ' +
          (e.shape === "document" ? "selected" : "") +
          '>문서</option><option value="database" ' +
          (e.shape === "database" ? "selected" : "") +
          '>저장 데이터</option><option value="predefined_process" ' +
          (e.shape === "predefined_process" ? "selected" : "") +
          '>서브프로세스</option><option value="manual_input" ' +
          (e.shape === "manual_input" ? "selected" : "") +
          '>수동 입력</option><option value="preparation" ' +
          (e.shape === "preparation" ? "selected" : "") +
          '>준비</option><option value="connector" ' +
          (e.shape === "connector" ? "selected" : "") +
          '>연결자</option><option value="delay" ' +
          (e.shape === "delay" ? "selected" : "") +
          ">지연</option></select></label>"
        : "") +
      '<div class="style-panel"><span class="field-label">모양 및 글자</span><div class="two-col">' +
      (e.type === "Box" ? '<label class="coord-wrap">채우기<input class="control" type="color" id="prop-fill" value="' + safeColor(st.fill,"#ffffff") + '"></label>' : '') +
      '<label class="coord-wrap">선 색<input class="control" type="color" id="prop-stroke" value="' + safeColor(st.stroke,e.type==="Arrow"?"#199574":"#de8b48") + '"></label>' +
      '<label class="coord-wrap">선 굵기<input class="control" type="number" min="1" max="30" id="prop-stroke-width" value="' + (Number(st.stroke_width)||2) + '"></label>' +
      (e.type === "Arrow" ? '<label class="field">경로<select class="control" id="prop-route"><option value="straight" '+(!e.route||e.route==="straight"?"selected":"")+'>직선</option><option value="orthogonal" '+(e.route==="orthogonal"?"selected":"")+'>직각 꺾은선</option><option value="curved" '+(e.route==="curved"?"selected":"")+'>곡선</option></select></label><label class="field">선 종류<select class="control" id="prop-line-style"><option value="solid" '+(!st.line_style||st.line_style==="solid"?"selected":"")+'>실선</option><option value="dashed" '+(st.line_style==="dashed"?"selected":"")+'>파선</option><option value="dotted" '+(st.line_style==="dotted"?"selected":"")+'>점선</option></select></label><label class="field">시작 화살촉<select class="control" id="prop-marker-start"><option value="none" '+(!st.marker_start||st.marker_start==="none"?"selected":"")+'>없음</option><option value="arrow" '+(st.marker_start==="arrow"?"selected":"")+'>화살촉</option><option value="diamond" '+(st.marker_start==="diamond"?"selected":"")+'>마름모</option></select></label><label class="field">끝 화살촉<select class="control" id="prop-marker-end"><option value="none" '+(st.marker_end==="none"?"selected":"")+'>없음</option><option value="arrow" '+(!st.marker_end||st.marker_end==="arrow"?"selected":"")+'>화살촉</option><option value="diamond" '+(st.marker_end==="diamond"?"selected":"")+'>마름모</option></select></label><label class="field">연결 라벨<textarea class="control area" id="prop-arrow-label" placeholder="예: 예 / 아니오">'+esc(e.text||"")+'</textarea></label>' : '') +
      (e.type !== "Arrow" ? '<label class="coord-wrap">글자 크기<input class="control" type="number" min="8" max="96" id="prop-font-size" value="' + (Number(st.font_size)||16) + '"></label><label class="coord-wrap">글자 색<input class="control" type="color" id="prop-text-color" value="' + safeColor(st.text_color,"#334057") + '"></label><label class="coord-wrap">정렬<select class="control" id="prop-text-align"><option value="left" '+(st.text_align==="left"?"selected":"")+'>왼쪽</option><option value="center" '+(!st.text_align||st.text_align==="center"?"selected":"")+'>가운데</option><option value="right" '+(st.text_align==="right"?"selected":"")+'>오른쪽</option></select></label>' : '') +
      (e.type !== "Arrow" ? '<label class="coord-wrap">회전(°)<input class="control" type="number" min="-180" max="180" id="prop-rotation" value="' + (Number(e.rotation)||0) + '"></label>' : '') +
      '</div></div><label class="field"><span>위치 및 크기 (픽셀)</span><div class="two-col"><label class="coord-wrap">X<input class="control" type="number" id="prop-x" value="' +
      Math.round(b[0]) +
      '"></label><label class="coord-wrap">Y<input class="control" type="number" id="prop-y" value="' +
      Math.round(b[1]) +
      '"></label><label class="coord-wrap">W<input class="control" type="number" id="prop-w" min="1" value="' +
      Math.round(b[2] - b[0]) +
      '"></label><label class="coord-wrap">H<input class="control" type="number" id="prop-h" min="1" value="' +
      Math.round(b[3] - b[1]) +
      '"></label></div></label><label class="field"><span>인식 신뢰도 <b class="confidence-value">' +
      (Number.isFinite(e.confidence)
        ? Math.round(e.confidence * 100) + "%"
        : "수동") +
      '</b></span><input class="control" id="prop-confidence" type="range" min="0" max="1" step="0.01" value="' +
      (Number.isFinite(e.confidence) ? e.confidence : 1) + '" ' +
      (e.source === "manual" ? "" : "disabled") + "></label>" +
      (e.type === "Arrow"
        ? '<div class="quality-card">화살표의 시작점·끝점과 연결 대상을 원본에서 확인하세요.</div><button class="button full" id="reverse-arrow">화살표 방향 반전</button>'
        : "") +
      (Number.isFinite(e.confidence) && e.confidence < .7
        ? '<div class="quality-card">신뢰도가 낮습니다. 원본에서 내용과 위치를 확인해 주세요.</div>'
        : "") +
      "</div>";
    $$("[id^=prop-]", root).forEach((field) => field.onchange = applyChanges);
    $("#reverse-arrow", root)?.addEventListener("click", () => {
      const arrow = state.doc.elements.find((item) => item.id === state.selected),
        relation = state.doc.relations.find((item) => item.arrow_id === state.selected);
      if (!arrow) return;
      [arrow.start, arrow.end] = [arrow.end, arrow.start];
      if (relation) {
        [relation.source, relation.target] = [relation.target, relation.source];
        relation.direction = direction(arrow.start, arrow.end);
        relation.status = "review"; relation.review_status = "modified";
        if (relation.source && relation.target) syncArrowRelation(relation);
      }
      commit(); render(); status("화살표 방향과 연결 관계를 반전했습니다.");
    });
  }
  function markRelations(id) {
    (state.doc.relations || []).forEach((r) => {
      if (r.source === id || r.target === id) {
        r.status = "review";
        if (r.review_status !== "deferred") r.review_status = "modified";
      }
    });
  }
  function syncArrowRelation(relation) {
    if (relation.type !== "flows_to") return;
    const from = state.doc.elements.find((e) =>
        e.id === relation.source && e.type === "Box"
      ),
      to = state.doc.elements.find((e) =>
        e.id === relation.target && e.type === "Box"
      ),
      arrow = state.doc.elements.find((e) =>
        e.id === relation.arrow_id && e.type === "Arrow"
      );
    if (!arrow) return;
    if (from && to && relation.source === relation.target) {
      arrow.start = edgePoint(from, { x: center(from).x, y: from.bbox[1] - 100 });
      arrow.end = edgePoint(to, { x: to.bbox[2] + 100, y: center(to).y });
    } else {
      if (from) arrow.start = edgePoint(from, to ? center(to) : arrow.end);
      if (to) arrow.end = edgePoint(to, from ? center(from) : arrow.start);
    }
    const a = arrow.start, b = arrow.end;
    arrow.bbox = [
      Math.min(a.x, b.x),
      Math.min(a.y, b.y),
      Math.min(state.doc.image.width, Math.max(a.x, b.x) + 1),
      Math.min(state.doc.image.height, Math.max(a.y, b.y) + 1),
    ];
    relation.direction = direction(a, b);
    relation.endpoint_method = from && to ? "attached" : "manual";
  }
  function attachArrowByGeometry(arrow) {
    const relation = state.doc.relations.find((item) => item.arrow_id === arrow.id),
      boxes = state.doc.elements.filter((item) => item.type === "Box"),
      start = point(arrow.start, { x: arrow.bbox[0], y: arrow.bbox[1] }),
      end = point(arrow.end, { x: arrow.bbox[2], y: arrow.bbox[3] }),
      nearest = (position) => boxes.map((box) => {
        const b = box.bbox, dx = Math.max(b[0] - position.x, 0, position.x - b[2]),
          dy = Math.max(b[1] - position.y, 0, position.y - b[3]);
        return { box, distance: Math.hypot(dx, dy) };
      }).sort((a, b) => a.distance - b.distance)[0],
      tolerance = Math.max(24, Math.min(state.doc.image.width, state.doc.image.height) * .025),
      from = nearest(start), to = nearest(end);
    if (!relation) return;
    relation.source = from && from.distance <= tolerance ? from.box.id : null;
    relation.target = to && to.distance <= tolerance ? to.box.id : null;
    const source = boxes.find((item) => item.id === relation.source), target = boxes.find((item) => item.id === relation.target);
    if (source && target && source.id === target.id) {
      arrow.start = edgePoint(source, { x: center(source).x, y: source.bbox[1] - 100 });
      arrow.end = edgePoint(target, { x: target.bbox[2] + 100, y: center(target).y });
    } else {
      if (source) arrow.start = edgePoint(source, target ? center(target) : arrow.end);
      if (target) arrow.end = edgePoint(target, source ? center(source) : arrow.start);
    }
    if (source || target) {
      arrow.bbox = [Math.min(arrow.start.x, arrow.end.x), Math.min(arrow.start.y, arrow.end.y),
        Math.min(state.doc.image.width, Math.max(arrow.start.x, arrow.end.x) + 1),
        Math.min(state.doc.image.height, Math.max(arrow.start.y, arrow.end.y) + 1)];
    }
    relation.direction = direction(arrow.start, arrow.end);
    relation.endpoint_method = relation.source && relation.target ? "attached" : "manual";
    relation.status = "review";
    relation.review_status = "modified";
    if (relation.source && relation.target) {
      state.doc.warnings = (state.doc.warnings || []).filter((warning) =>
        !(warning.element_id === arrow.id && warning.code === "manual_arrow_unlinked")
      );
    }
  }
  function syncConnectedArrows(id, review = true) {
    (state.doc.relations || []).filter((r) =>
      r.source === id || r.target === id
    ).forEach(syncArrowRelation);
    if (review) markRelations(id);
  }
  function newBlank() {
    if (
      state.dirty &&
      !confirm("저장하지 않은 변경사항이 있습니다. 새 빈 캔버스를 만들까요?")
    ) return;
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    beginDocumentReplacement();
    state.previewUrl = null;
    state.file = null;
    state.doc = {
      version: "1.0",
      name: "새 순서도 프로젝트",
      image: { width: 1600, height: 1000 },
      elements: [],
      relations: [],
      warnings: [],
      created_at: new Date().toISOString(),
    };
    state.selected = null;
    state.selectedIds = [];
    state.viewMode = "restored";
    state.tool = "select";
    state.connectFrom = null;
    resetHistory();
    render();
    fitCanvas();
    status("빈 캔버스를 만들었습니다. 도구를 골라 순서도를 그리세요.");
  }
  function edgePoint(box, toward, port) { return RaduModel.boundary(box, toward, port); }
  function selectElement(id) {
    if (!state.doc || !state.doc.elements.some((e) => e.id === id)) return;
    const selectedElement = state.doc.elements.find(e=>e.id===id);
    if (!RaduModel.visible(state.doc, selectedElement)) {
      status("숨긴 레이어의 요소입니다. 레이어 표시를 켜고 선택하세요.", "error");
      return;
    }
    if (!state.selectedIds.includes(id)) state.selectedIds = RaduModel.expandGroup(state.doc,[id]);
    state.selected = id;
    if (state.tool === "link") {
      const target = state.doc.elements.find((e) => e.id === id);
      if (target.type !== "Box") {
        status("연결은 도형 요소끼리 만들 수 있습니다.", "error");
        return;
      }
      const layer = state.doc.layers?.find(item=>item.id===state.doc.active_layer);
      if (RaduModel.locked(state.doc, target) || layer?.locked || layer?.visible === false) {
        status("잠긴 도형이나 숨긴 작성 레이어에는 연결을 만들 수 없습니다.", "error");
        return;
      }
      if (!state.connectFrom) {
        state.connectFrom = id;
        status("시작 도형을 선택했습니다. 이제 도착 도형을 선택하세요.");
        render();
        return;
      }
      if (state.connectFrom === id) {
        state.connectFrom = null;
        status("연결을 취소했습니다.");
        render();
        return;
      }
      if (state.doc.relations.some((r) =>
        r.type === "flows_to" && r.source === state.connectFrom && r.target === id
      )) {
        state.connectFrom = null;
        status("같은 방향의 연결이 이미 있습니다.", "error");
        render();
        return;
      }
      const from = state.doc.elements.find((e) => e.id === state.connectFrom);
      if (!from || !RaduModel.visible(state.doc, from) || RaduModel.locked(state.doc, from)) {
        state.connectFrom = null;
        status("시작 도형이 숨겨졌거나 잠겼습니다. 다시 선택하세요.", "error");
        return;
      }
      const to = target,
        ca = center(from),
        cb = center(to),
        a = edgePoint(from, cb),
        b = edgePoint(to, ca),
        arrowId = "manual-arrow-" + crypto.randomUUID().slice(0, 8),
        relation = {
          id: "manual-edge-" + crypto.randomUUID().slice(0, 8),
          type: "flows_to",
          source: from.id,
          target: to.id,
          arrow_id: arrowId,
          direction: direction(ca, cb),
          endpoint_method: "attached",
          status: "review",
          review_status: "pending",
        };
      if (Math.hypot(a.x-b.x, a.y-b.y) < 8) {
        state.connectFrom = null;
        status("두 도형이 너무 가까워 연결선이 보이지 않습니다. 도형을 이동한 뒤 연결하세요.", "error");
        return;
      }
      state.doc.elements.push({
        id: arrowId,
        type: "Arrow",
        bbox: [
          Math.min(a.x, b.x),
          Math.min(a.y, b.y),
          Math.min(state.doc.image.width, Math.max(a.x, b.x) + 1),
          Math.min(state.doc.image.height, Math.max(a.y, b.y) + 1),
        ],
        start: a,
        end: b,
        confidence: 1,
        source: "manual",
        layer_id: state.doc.active_layer,
      });
      state.doc.relations.push(relation);
      state.selected = arrowId;
      state.connectFrom = null;
      commit();
      status("연결을 추가했습니다. 방향과 연결 대상을 검토하세요.");
      render();
      return;
    }
    if (state.tool === "style") {
      const item = state.doc.elements.find((element) => element.id === id);
      if (!state.styleClipboard) {
        state.styleClipboard = {
          style: clone(item.style || {}),
          rotation: item.rotation || 0,
          sourceId: item.id,
        };
        selectElement(id);
        status("서식을 복사했습니다. 적용할 요소를 차례로 클릭하세요.");
      } else if (state.styleClipboard.sourceId !== id) {
        item.style = clone(state.styleClipboard.style);
        item.rotation = state.styleClipboard.rotation;
        if (item.type === "Box") syncConnectedArrows(item.id);
        markRelations(item.id);
        state.selected = id;
        state.selectedIds = [id];
        commit();
        render();
        status("서식을 적용했습니다. 다른 요소에도 계속 적용할 수 있습니다.");
      }
      return;
    }
    render();
  }
  function direction(a, b) {
    return Math.abs(b.x - a.x) >= Math.abs(b.y - a.y)
      ? (b.x >= a.x ? "right" : "left")
      : (b.y >= a.y ? "down" : "up");
  }
  function closestBoxToPoint(boxes, p) {
    return boxes.map((box) => {
      const b = box.bbox,
        q = { x: Math.max(b[0], Math.min(b[2], p.x)), y: Math.max(b[1], Math.min(b[3], p.y)) };
      return { box, point: q, distance: Math.hypot(q.x - p.x, q.y - p.y) };
    }).sort((a, b) => a.distance - b.distance)[0] || null;
  }
  function elementPointerDown(event) {
    if (event.button !== 0) return;
    const g = event.currentTarget, id = g.dataset.id;
    if(state.busy) return;
    const selectedItem=state.doc.elements.find(e=>e.id===id);
    if(RaduModel.locked(state.doc,selectedItem)){event.stopPropagation();state.selected=id;state.selectedIds=[id];render();return;}
    if(state.tool === "style"){event.preventDefault();event.stopPropagation();selectElement(id);return;}
    state.pointerElementId = id;
    if (state.spacePan || state.tool === "pan") {
      event.preventDefault();
      event.stopPropagation();
      state.pan = { x: event.clientX, y: event.clientY };
      $("#diagram").setPointerCapture(event.pointerId);
      return;
    }
    if (state.tool === "link") {
      event.preventDefault();
      event.stopPropagation();
      selectElement(id);
      return;
    }
    // Drawing tools must receive pointer events even when the pointer starts
    // on an existing shape. Previously this handler swallowed those events.
    if (state.tool !== "select") return;
    event.preventDefault();
    event.stopPropagation();
    const additive = event.shiftKey || event.ctrlKey || event.metaKey;
    if (additive) {
      state.selectedIds = state.selectedIds.includes(id)
        ? state.selectedIds.filter((item) => item !== id)
        : [...state.selectedIds, id];
      state.selected = state.selectedIds.at(-1) || null;
      render();
      return;
    }
    const continuingMultiSelection = state.selectedIds.length > 1 &&
      state.selectedIds.includes(id);
    state.selected = id;
    if (!continuingMultiSelection) state.selectedIds = event.altKey ? [id] : RaduModel.expandGroup(state.doc,[id]);
    const handle = event.target.closest("[data-handle]");
    const item = state.doc.elements.find((e) => e.id === id);
    const lastTap = state.lastPointerTap;
    if (hybrid && !handle && lastTap?.id === id && performance.now() - lastTap.time < 650 && Math.hypot(event.clientX-lastTap.x, event.clientY-lastTap.y) < 6) {
      state.lastPointerTap = null; state.drag = null; editElementText(id); return;
    }
    state.drag = {
      id,
      mode: handle
        ? state.selectedIds.length > 1 ? "group-resize" : item.type === "Arrow" ? "arrow-endpoint" : "resize"
        : "move",
      handle: handle && handle.dataset.handle,
      start: pointAt(event),
      before: clone(state.doc),
      changed: false,
      ids: state.selectedIds.length > 1 && state.selectedIds.includes(id)
        ? state.selectedIds.slice()
        : [id],
      boxes: Object.fromEntries(
        (state.selectedIds.length > 1 && state.selectedIds.includes(id)
          ? state.selectedIds
          : [id]).map((
            selectedId,
          ) => [
            selectedId,
            state.doc.elements.find((e) => e.id === selectedId)?.bbox.slice(),
          ]),
      ),
      bbox: state.selectedIds.length>1 ? RaduModel.bounds(state.doc.elements.filter(e=>state.selectedIds.includes(e.id))) : item.bbox.slice(),
      arrowStart: item.start && clone(item.start),
      arrowEnd: item.end && clone(item.end),
      attachedArrows: state.doc.relations.filter((r) =>
        r.source === id || r.target === id
      ).map((r) => state.doc.elements.find((a) => a.id === r.arrow_id)).filter(
        Boolean,
      ).map((a) => ({
        id: a.id,
        bbox: a.bbox.slice(),
        start: a.start && clone(a.start),
        end: a.end && clone(a.end),
      })),
      group: g,
    };
    $("#diagram").setPointerCapture(event.pointerId);
    renderDiagram();
    renderList();
    renderInspector();
  }
  function createTable(start, end) {
    RaduModel.normalize(state.doc);
    const layer = state.doc.layers.find(l => l.id === state.doc.active_layer);
    if (layer?.locked || layer?.visible === false) { toast("작성 레이어를 확인하세요.", true); return; }
    const x = Math.min(start.x, end.x), y = Math.min(start.y, end.y),
      width = Math.abs(end.x - start.x), height = Math.abs(end.y - start.y);
    if (width < tableSize.cols * 12 || height < tableSize.rows * 12) { toast("표 영역을 더 크게 드래그하세요.", true); render(); return; }
    const tableId = RaduModel.uid("table"), cells = [];
    for (let row = 0; row < tableSize.rows; row++) for (let col = 0; col < tableSize.cols; col++) {
      cells.push({id: RaduModel.uid("cell"), type: "Box", shape: "process", semantic_type: "table-cell", table_id: tableId,
        bbox: [x+col*width/tableSize.cols, y+row*height/tableSize.rows, x+(col+1)*width/tableSize.cols, y+(row+1)*height/tableSize.rows],
        text: "", source: "manual", layer_id: state.doc.active_layer,
        style: {fill: "#ffffff", stroke: "#334057", stroke_width: 2, font_family: "NanumGothic", font_size: Math.min(96, Math.max(12, height/tableSize.rows*.35))}});
    }
    state.doc.elements.push(...cells); state.selected = cells[0].id; state.selectedIds = [cells[0].id];
    selectTool("select"); commit(); render(); status("표를 추가했습니다. 셀을 더블클릭하여 내용을 입력하세요.");
  }
  function tableDialog() {
    if (!state.doc) return;
    $("#modal-root").innerHTML = '<div class="modal-backdrop"><div class="modal" role="dialog" aria-label="표 추가"><div class="modal-head"><h2>표 추가</h2><button class="modal-close" aria-label="닫기">×</button></div><label class="field">행<input class="control" id="table-rows" type="number" min="1" max="20" value="3"></label><label class="field">열<input class="control" id="table-cols" type="number" min="1" max="20" value="3"></label><p class="hint">행과 열을 정한 뒤 캔버스에서 표 영역을 드래그하세요. 각 셀을 따로 선택·수정할 수 있습니다.</p><button class="button primary" id="draw-table">영역 그리기</button></div></div>';
    $(".modal-close").onclick = closeModal;
    $("#draw-table").onclick = () => {
      const rows = Number($("#table-rows").value), cols = Number($("#table-cols").value);
      if (![rows, cols].every(v => Number.isInteger(v) && v >= 1 && v <= 20) || rows*cols > 100) return toast("행·열은 1~20, 전체 셀은 100개 이하로 지정하세요.", true);
      tableSize = {rows, cols}; closeModal(); selectTool("table"); status("캔버스에서 표 영역을 드래그하세요.");
    };
  }
  function createElement(tool, start, end) {
    if (hybrid && tool === "table") return createTable(start, end);
    RaduModel.normalize(state.doc);
    const activeLayer = state.doc.layers.find(l=>l.id===state.doc.active_layer);
    if(activeLayer?.locked || activeLayer?.visible === false){toast("작성 레이어의 잠금과 표시 상태를 확인하세요.",true);render();return;}
    const size = dimensions(), dx = end.x - start.x, dy = end.y - start.y;
    if (tool === "arrow" && Math.hypot(dx, dy) < 8) {
      toast("화살표를 만들려면 시작점과 끝점을 드래그하세요.", true);
      return;
    }
    let x = Math.min(start.x, end.x),
      y = Math.min(start.y, end.y),
      w = Math.abs(dx),
      h = Math.abs(dy);
    if (w < 8) w = Math.min(size.width, tool === "text" ? 180 : 130);
    if (h < 8) h = Math.min(size.height, tool === "text" ? 42 : 58);
    x = Math.max(0, Math.min(size.width - w, x));
    y = Math.max(0, Math.min(size.height - h, y));
    if (state.gridSnap && tool !== "arrow") {
      const grid = state.doc.canvas?.grid_size || 10;
      w = Math.min(size.width, Math.max(grid, Math.round(w / grid) * grid));
      h = Math.min(size.height, Math.max(grid, Math.round(h / grid) * grid));
      x = Math.max(0, Math.min(size.width - w, Math.round(x / grid) * grid));
      y = Math.max(0, Math.min(size.height - h, Math.round(y / grid) * grid));
    }
    let e;
    if (tool === "arrow") {
      const arrowStart = {
          x: Math.max(0, Math.min(size.width - 1, start.x)),
          y: Math.max(0, Math.min(size.height - 1, start.y)),
        },
        arrowEnd = {
          x: Math.max(0, Math.min(size.width - 1, end.x)),
          y: Math.max(0, Math.min(size.height - 1, end.y)),
        };
      e = {
        id: "manual-arrow-" + crypto.randomUUID().slice(0, 8),
        type: "Arrow",
        bbox: [
          Math.min(arrowStart.x, arrowEnd.x),
          Math.min(arrowStart.y, arrowEnd.y),
          Math.min(size.width, Math.max(arrowStart.x, arrowEnd.x) + 1),
          Math.min(size.height, Math.max(arrowStart.y, arrowEnd.y) + 1),
        ],
        start: arrowStart,
        end: arrowEnd,
        confidence: 1,
        source: "manual",
      };
      const boxes = state.doc.elements.filter((item) => item.type === "Box"),
        fromCandidate = closestBoxToPoint(boxes, arrowStart),
        toCandidate = closestBoxToPoint(boxes, arrowEnd),
        tolerance = Math.max(24, Math.min(size.width, size.height) * .025),
        from = fromCandidate && fromCandidate.distance <= tolerance ? fromCandidate.box : null,
        to = toCandidate && toCandidate.distance <= tolerance ? toCandidate.box : null;
      if (from || to) {
        if (from) e.start = edgePoint(from, to ? center(to) : arrowEnd);
        if (to) e.end = edgePoint(to, from ? center(from) : arrowStart);
        e.bbox = [
          Math.min(e.start.x, e.end.x), Math.min(e.start.y, e.end.y),
          Math.min(size.width, Math.max(e.start.x, e.end.x) + 1),
          Math.min(size.height, Math.max(e.start.y, e.end.y) + 1),
        ];
      }
      state.doc.elements.push(e);
      state.doc.relations.push({
        id: "manual-edge-" + crypto.randomUUID().slice(0, 8),
        type: "flows_to",
        source: from ? from.id : null,
        target: to ? to.id : null,
        arrow_id: e.id,
        direction: direction(arrowStart, arrowEnd),
        endpoint_method: "manual",
        status: "review",
        review_status: "pending",
      });
      if (!from || !to) {
        state.doc.warnings.push({
          code: "manual_arrow_unlinked",
          element_id: e.id,
          message:
            "화살표 끝점 주변에서 연결 도형을 찾지 못했습니다. 관계 탭에서 대상을 지정하세요.",
        });
      }
    } else if (tool === "text") {
      e = {
        id: "manual-text-" + crypto.randomUUID().slice(0, 8),
        type: "Text",
        bbox: [x, y, x + w, y + h],
        text: "새 텍스트",
        confidence: 1,
        source: "manual",
      };
    } else {
      const shape = tool === "shape"
        ? state.shapeKind
        : tool === "decision" || tool === "terminator"
        ? tool
        : "process";
      e = {
        id: "manual-box-" + crypto.randomUUID().slice(0, 8),
        type: "Box",
        shape,
        bbox: [x, y, x + w, y + h],
        text: shape === "decision"
          ? "조건 입력"
          : shape === "terminator"
          ? "시작 / 종료"
          : shape === "input_output"
          ? "입력 / 출력"
          : shape === "document"
          ? "문서"
          : shape === "database"
          ? "데이터 저장"
          : kindName({ type: "Box", shape }),
        confidence: 1,
        source: "manual",
      };
    }
    if (tool !== "arrow") state.doc.elements.push(e);
    e.layer_id = state.doc.active_layer;
    if(tool === "arrow") syncArrowRelation(state.doc.relations.find(r=>r.arrow_id===e.id));
    state.selected = e.id;
    state.selectedIds = [e.id];
    commit();
    render();
    status("요소를 추가했습니다. 같은 도구로 계속 그리거나 V를 눌러 선택하세요.");
  }
  function resizeBox(bbox, handle, dx, dy, size, ratio) {
    return RaduModel.resize(bbox, handle, dx, dy, size, ratio);
  }
  function canvasPointerDown(event) {
    if (event.button !== 0 || !state.doc || state.busy || state.viewMode === "original" || state.tool === "style") return;
    state.pointerElementId = null;
    if (state.spacePan || state.tool === "pan") {
      event.preventDefault();
      state.pan = { x: event.clientX, y: event.clientY };
      $("#diagram").setPointerCapture(event.pointerId);
      return;
    }
    if (state.tool === "select" || state.tool === "link") {
      if (event.target === $("#diagram")) {
        if (state.tool === "select") {
          const start = pointAt(event);
          state.marquee = {
            start,
            end: start,
            additive: event.shiftKey || event.ctrlKey || event.metaKey,
            base: state.selectedIds.slice(),
          };
          $("#diagram").setPointerCapture(event.pointerId);
        } else {
          state.selected = null;
          state.selectedIds = [];
          render();
        }
      }
      return;
    }
    event.preventDefault();
    const start = pointAt(event);
    state.draw = { tool: state.tool, start: start, end: start, points: [start] };
    $("#diagram").setPointerCapture(event.pointerId);
    renderDiagram();
  }
  $("#diagram").addEventListener("pointermove", (event) => {
    if (state.pan) {
      $("#diagram").style.cursor = "grabbing";
      const area = $("#canvas-scroll"),
        dx = event.clientX - state.pan.x,
        dy = event.clientY - state.pan.y;
      area.scrollLeft -= dx;
      area.scrollTop -= dy;
      state.pan = { x: event.clientX, y: event.clientY };
      return;
    }
    if (state.draw) {
      state.draw.end = pointAt(event);
      if(state.draw.tool === "pen" && state.draw.points.length < 20000){const last=state.draw.points.at(-1);if(Math.hypot(last.x-state.draw.end.x,last.y-state.draw.end.y)>=2)state.draw.points.push(state.draw.end);}
      renderDiagram();
      return;
    }
    if (state.marquee) {
      state.marquee.end = pointAt(event);
      renderDiagram();
      return;
    }
    if (state.drag) {
      const d = state.drag,
        p = pointAt(event),
        dx = p.x - d.start.x,
        dy = p.y - d.start.y,
        e = state.doc.elements.find((item) => item.id === d.id);
      if (!e || (!dx && !dy)) return;
      d.changed = true;
      if(d.mode === "group-resize"){
        const next=resizeBox(d.bbox,d.handle,dx,dy,dimensions(),event.shiftKey),
          sx=(next[2]-next[0])/(d.bbox[2]-d.bbox[0]),sy=(next[3]-next[1])/(d.bbox[3]-d.bbox[1]),
          remap=p=>({x:next[0]+(p.x-d.bbox[0])*sx,y:next[1]+(p.y-d.bbox[1])*sy});
        d.ids.forEach(id=>{const old=d.before.elements.find(item=>item.id===id),current=state.doc.elements.find(item=>item.id===id);if(!old||!current)return;const tl=remap({x:old.bbox[0],y:old.bbox[1]}),br=remap({x:old.bbox[2],y:old.bbox[3]});current.bbox=[tl.x,tl.y,br.x,br.y];if(old.start)current.start=remap(old.start);if(old.end)current.end=remap(old.end);if(old.points)current.points=old.points.map(remap);});
      } else if (d.mode === "arrow-endpoint") {

        const p = pointAt(event),
          endpoint = d.handle === "start" ? "start" : "end",
          size = dimensions();
        e[endpoint] = {
          x: Math.max(0, Math.min(size.width - 1, p.x)),
          y: Math.max(0, Math.min(size.height - 1, p.y)),
        };
        e.bbox = [
          Math.min(e.start.x, e.end.x), Math.min(e.start.y, e.end.y),
          Math.min(size.width, Math.max(e.start.x, e.end.x) + 1),
          Math.min(size.height, Math.max(e.start.y, e.end.y) + 1),
        ];
      } else if (d.mode === "resize") {
        const old = e.bbox.slice(),
          next = resizeBox(d.bbox, d.handle, dx, dy, dimensions(), event.shiftKey || e.aspect_locked);
        if (e.type === "Arrow" && e.start && e.end) {
          const remap = (raw) => {
            const q = point(raw, { x: old[0], y: old[1] }),
              rx = (q.x - old[0]) / Math.max(1, old[2] - old[0]),
              ry = (q.y - old[1]) / Math.max(1, old[3] - old[1]);
            return {
              x: next[0] + rx * (next[2] - next[0]),
              y: next[1] + ry * (next[3] - next[1]),
            };
          };
          e.start = remap(e.start);
          e.end = remap(e.end);
        }
        if (state.gridSnap) {
          const grid = state.doc.canvas?.grid_size || 10;
          next[0] = Math.max(0, Math.round(next[0] / grid) * grid);
          next[1] = Math.max(0, Math.round(next[1] / grid) * grid);
          next[2] = Math.min(
            dimensions().width,
            Math.max(next[0] + grid, Math.round(next[2] / grid) * grid),
          );
          next[3] = Math.min(
            dimensions().height,
            Math.max(next[1] + grid, Math.round(next[3] / grid) * grid),
          );
        }
        if(e.points){const old=d.bbox,original=d.before.elements.find(item=>item.id===e.id);e.points=original.points.map(p=>({x:next[0]+(p.x-old[0])*(next[2]-next[0])/(old[2]-old[0]),y:next[1]+(p.y-old[1])*(next[3]-next[1])/(old[3]-old[1])}));}
        e.bbox = next;
      } else {
        const size = dimensions(),
          boxes = d.ids.map((id) => d.boxes[id]).filter(Boolean),
          minX = Math.min(...boxes.map((b) => b[0])),
          minY = Math.min(...boxes.map((b) => b[1])),
          maxX = Math.max(...boxes.map((b) => b[2])),
          maxY = Math.max(...boxes.map((b) => b[3]));
        let moveX = Math.max(-minX, Math.min(size.width - maxX, dx)),
          moveY = Math.max(-minY, Math.min(size.height - maxY, dy));
        state.guides = [];
        if (state.gridSnap) {
          moveX = Math.max(
            -minX,
            Math.min(
              size.width - maxX,
              Math.round((minX + moveX) / (state.doc.canvas?.grid_size||10)) * (state.doc.canvas?.grid_size||10) - minX,
            ),
          );
          moveY = Math.max(
            -minY,
            Math.min(
              size.height - maxY,
              Math.round((minY + moveY) / (state.doc.canvas?.grid_size||10)) * (state.doc.canvas?.grid_size||10) - minY,
            ),
          );
        } else {
          const movingIds = new Set(d.ids), others = state.doc.elements
            .filter((item) => !movingIds.has(item.id) && item.type !== "Arrow")
            .map((item) => item.bbox), threshold = 10;
          const snapAxis = (axis) => {
            const first = axis === "x" ? minX : minY,
              last = axis === "x" ? maxX : maxY,
              delta = axis === "x" ? moveX : moveY,
              candidates = [first, (first + last) / 2, last];
            let best = null;
            for (const box of others) {
              const targets = axis === "x" ? [box[0], (box[0] + box[2]) / 2, box[2]] : [box[1], (box[1] + box[3]) / 2, box[3]];
              for (const source of candidates) for (const target of targets) {
                const adjust = target - (source + delta);
                if (Math.abs(adjust) <= threshold && (!best || Math.abs(adjust) < Math.abs(best.adjust))) best = { adjust, target };
              }
            }
            if (!best) return delta;
            state.guides.push({ axis, value: best.target });
            return Math.max(axis === "x" ? -minX : -minY,
              Math.min(axis === "x" ? size.width - maxX : size.height - maxY, delta + best.adjust));
          };
          moveX = snapAxis("x");
          moveY = snapAxis("y");
        }
        d.ids.forEach((id) => {
          const item = state.doc.elements.find((candidate) =>
              candidate.id === id
            ),
            box = d.boxes[id];
          if (!item || !box) return;
          item.bbox = box.map((v, i) => v + (i % 2 === 0 ? moveX : moveY));
          const originalItem=d.before.elements.find(e=>e.id===id);
          if(originalItem.points)item.points=originalItem.points.map(p=>({x:p.x+moveX,y:p.y+moveY}));
          if (item.type === "Arrow" && item.start && item.end) {
            const original = d.before.elements.find(e => e.id === id),
              start = point(original.start, { x: 0, y: 0 }),
              end = point(original.end, { x: 0, y: 0 });
            item.start = { x: start.x + moveX, y: start.y + moveY };
            item.end = { x: end.x + moveX, y: end.y + moveY };
          }
        });
      }
      d.ids.forEach((id) => {
        const item = state.doc.elements.find((candidate) =>
          candidate.id === id
        );
        if (item?.type === "Box") syncConnectedArrows(id, false);
      });
      renderDiagram();
    }
  });
  $("#diagram").addEventListener("pointerup", (event) => {
    if (state.pan) {
      state.pan = null;
      $("#diagram").style.cursor = state.tool === "pan" ? "grab" : "default";
      return;
    }
    if (state.draw) {
      const d = state.draw;
      state.draw = null;
      if(d.tool === "pen") {
        const activeLayer = state.doc.layers?.find(layer=>layer.id===state.doc.active_layer);
        if (activeLayer?.locked || activeLayer?.visible === false) {
          toast("작성 레이어의 잠금과 표시 상태를 확인하세요.", true);
          render();
          return;
        }
        if(d.points.length<2){render();return;}
        const xs=d.points.map(p=>p.x),ys=d.points.map(p=>p.y),size=dimensions();
        const element={id:RaduModel.uid("stroke"),type:"Stroke",points:d.points,bbox:[Math.min(...xs),Math.min(...ys),Math.min(size.width,Math.max(...xs)+1),Math.min(size.height,Math.max(...ys)+1)],style:{stroke:"#334057",stroke_width:2},layer_id:state.doc.active_layer,source:"manual"};
        state.doc.elements.push(element);state.selected=element.id;state.selectedIds=[element.id];commit();render();
      } else createElement(d.tool, d.start, pointAt(event));
      return;
    }
    if (state.marquee) {
      const m = state.marquee,
        end = pointAt(event),
        x1 = Math.min(m.start.x, end.x),
        y1 = Math.min(m.start.y, end.y),
        x2 = Math.max(m.start.x, end.x),
        y2 = Math.max(m.start.y, end.y),
        hit = state.doc.elements.filter((item) =>
          RaduModel.visible(state.doc,item) && !RaduModel.locked(state.doc,item) && item.bbox[0] <= x2 && item.bbox[2] >= x1 &&
          item.bbox[1] <= y2 && item.bbox[3] >= y1
        ).map((item) => item.id);
      state.selectedIds = m.additive ? [...new Set([...m.base, ...hit])] : hit;
      state.selected = state.selectedIds.at(-1) || null;
      state.marquee = null;
      render();
      return;
    }
    if (state.drag) {
      const d = state.drag;
      if (hybrid) state.lastPointerTap = d.changed ? null : {id:d.id, time:performance.now(), x:event.clientX, y:event.clientY};
      state.drag = null;
      state.guides = [];
      const e = state.doc.elements.find((item) => item.id === d.id);
      if (!e) {
        render();
        return;
      }
      if (
        d.changed &&
        JSON.stringify(state.doc.elements) !== JSON.stringify(d.before.elements)
      ) {
        if (d.mode === "arrow-endpoint" && e.type === "Arrow") attachArrowByGeometry(e);
        d.ids.forEach((id) => {
          const item = state.doc.elements.find((candidate) =>
            candidate.id === id
          );
          if (item?.type === "Box") syncConnectedArrows(id);
          markRelations(id);
        });
        if (e.type === "Arrow") {
          if (d.mode !== "arrow-endpoint") attachArrowByGeometry(e);
        }
        commit();
      } else if (d.changed) state.doc = d.before;
      render();
    }
  });
  $("#diagram").addEventListener("pointercancel", () => {
    if (state.drag) {
      state.doc = state.drag.before;
      state.selected = state.doc.elements.some((e) => e.id === state.drag.id)
        ? state.drag.id
        : null;
    }
    state.drag = null;
    state.draw = null;
    state.marquee = null;
    state.pan = null;
    state.guides = [];
    render();
  });
  $("#diagram").addEventListener("pointerdown", canvasPointerDown);
  $("#source-image").onerror = () => {
    if (state.doc) {
      $("#source-image").hidden = true;
      status(
        "저장된 문서의 원본 이미지를 찾을 수 없습니다. 요소 편집은 계속할 수 있습니다.",
        "error",
      );
    }
  };
  function editElementText(id) {
    if (!state.doc) return;
    const item = state.doc.elements.find((e) => e.id === id);
    if (!item || !["Box", "Text", "Arrow"].includes(item.type)) return;
    if (RaduModel.locked(state.doc, item)) {
      toast("잠긴 요소 또는 레이어를 먼저 잠금 해제하세요.", true);
      return;
    }
    $(".inline-editor")?.remove();
    const editor = document.createElement("textarea"),
      [boxX, boxY, right, bottom] = item.bbox,
      x = item.type === "Arrow" ? (point(item.start, {x:boxX,y:boxY}).x + point(item.end, {x:right,y:bottom}).x) / 2 - 90 : boxX,
      y = item.type === "Arrow" ? (point(item.start, {x:boxX,y:boxY}).y + point(item.end, {x:right,y:bottom}).y) / 2 - 24 : boxY;
    editor.className = "inline-editor";
    editor.setAttribute("aria-label", "요소 텍스트 편집");
    editor.value = item.text || "";
    const originalText = editor.value;
    Object.assign(editor.style, {
      left: x + "px",
      top: y + "px",
      width: Math.max(120, item.type === "Arrow" ? 180 : right - boxX) + "px",
      height: Math.max(28, item.type === "Arrow" ? 48 : bottom - boxY) + "px",
    });
    $("#canvas-card").append(editor);
    let done = false;
    const finish = (save) => {
      if (done) return;
      done = true;
      editor.remove();
      if (!save || editor.value === originalText ||
          !state.doc.elements.some((e) => e.id === id)) return;
      item.text = editor.value;
      if (item.source !== "manual") item.source = "manual-edit";
      const relation = state.doc.relations.find((entry) => entry.arrow_id === item.id);
      if (relation) { relation.label = editor.value; relation.status = "review"; relation.review_status = "modified"; }
      else markRelations(item.id);
      commit();
      render();
      status("텍스트를 수정했습니다.");
    };
    editor.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        finish(false);
      } else if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        finish(true);
      }
    });
    editor.addEventListener("blur", () => finish(true));
    editor.focus();
    editor.select();
  }
  // Pointer capture and selection redraws can change the native dblclick target
  // from the SVG group to the SVG root. Track the element from pointerdown so
  // double-click editing works reliably even when the browser retargets click.
  $("#diagram").addEventListener("click", (event) => {
    if (hybrid || state.tool !== "select") return;
    const group = event.target.closest &&
      event.target.closest(".diagram-element");
    const id = group && group.dataset.id || state.pointerElementId;
    if (!id || !state.doc) return;
    const now = performance.now(), previous = state.lastElementClick;
    if (previous && previous.id === id && now - previous.time < 650) {
      state.lastElementClick = null;
      editElementText(id);
      return;
    }
    state.lastElementClick = { id, time: now };
  });
  function deleteSelected() {
    if (!state.doc || !(state.selectedIds.length || state.selected)) return;
    const removed = new Set(
      state.selectedIds.length ? state.selectedIds : [state.selected],
    );
    const attached = state.doc.relations.filter(r=>removed.has(r.source)||removed.has(r.target)).map(r=>state.doc.elements.find(e=>e.id===r.arrow_id)).filter(Boolean);
    if(state.doc.elements.some(e=>removed.has(e.id)&&RaduModel.locked(state.doc,e))||attached.some(e=>RaduModel.locked(state.doc,e))){toast('잠긴 요소 또는 연결 화살표를 먼저 잠금 해제하세요.',true);return;}
    state.doc.relations.filter((r) =>
      removed.has(r.source) || removed.has(r.target) || removed.has(r.arrow_id)
    )
      .forEach((r) => {
        if (r.arrow_id) removed.add(r.arrow_id);
      });
    state.doc.elements = state.doc.elements.filter((e) => !removed.has(e.id));
    state.doc.relations = state.doc.relations.filter((r) =>
      !removed.has(r.source) && !removed.has(r.target) &&
      !removed.has(r.arrow_id)
    );
    state.doc.warnings = state.doc.warnings.filter((w) =>
      !removed.has(w.element_id)
    );
    state.selected = null;
    state.selectedIds = [];
    commit();
    render();
    status("요소와 연결된 관계 및 화살표를 삭제했습니다.");
  }
  function addProcess() {
    if (!state.doc) {
      toast("먼저 빈 캔버스를 만들거나 이미지를 분석해 주세요.", true);
      return;
    }
    const w = Math.min(150, state.doc.image.width),
      h = Math.min(64, state.doc.image.height),
      size = dimensions(),
      id = "manual-box-" + crypto.randomUUID().slice(0, 8);
    let x = Math.max(0, (size.width - w) / 2),
      y = Math.max(0, (size.height - h) / 2);
    const occupied = state.doc.elements.map((item) => item.bbox);
    const overlaps = (left, top) => occupied.some((b) =>
      left < b[2] + 12 && left + w + 12 > b[0] &&
      top < b[3] + 12 && top + h + 12 > b[1]
    );
    if (overlaps(x, y)) {
      const step = Math.max(w, h) + 18, cols = Math.max(1, Math.floor(size.width / step));
      let placed = false;
      for (let i = 0; i < Math.ceil(size.width / step) * Math.ceil(size.height / step); i++) {
        const col = i % cols, row = Math.floor(i / cols);
        const candidateX = Math.min(size.width - w, col * step + 16);
        const candidateY = Math.min(size.height - h, row * step + 16);
        if (!overlaps(candidateX, candidateY)) { x = candidateX; y = candidateY; placed = true; break; }
      }
      if (!placed) {
        x = Math.max(0, Math.min(size.width - w, x + (state.doc.elements.length % 5) * 12));
        y = Math.max(0, Math.min(size.height - h, y + (state.doc.elements.length % 5) * 12));
      }
    }
    const shape = state.shapeKind || "process";
    const activeLayer = state.doc.layers?.find(l=>l.id===state.doc.active_layer);
    if(activeLayer?.locked || activeLayer?.visible === false){toast("작성 레이어의 잠금과 표시 상태를 확인하세요.",true);return;}
    state.doc.elements.push({
      id: id,
      type: "Box",
      shape,
      text: shape === "terminator" ? "시작 / 종료"
        : shape === "decision" ? "조건 입력"
        : shape === "input_output" ? "입력 / 출력"
        : kindName({ type: "Box", shape }),
      bbox: [x, y, x + w, y + h],
      confidence: 1,
      source: "manual",
      layer_id: state.doc.active_layer,
    });
    state.selected = id;
    state.selectedIds = [id];
    commit();
    render();
  }
  function openTemplateMenu() {
    const root = $("#modal-root");
    root.innerHTML = '<div class="modal-backdrop"><div class="modal" role="dialog" aria-modal="true"><div class="modal-head"><h2>순서도 템플릿</h2><button class="modal-close" aria-label="닫기">×</button></div><p class="hint">템플릿은 현재 문서를 새 순서도로 바꿉니다. 생성 후 도형과 연결을 자유롭게 편집할 수 있습니다.</p><div class="template-grid"><button class="button" data-template="linear">시작 → 처리 → 종료</button><button class="button" data-template="decision">조건 분기 (예 / 아니오)</button><button class="button" data-template="input">입력 → 처리 → 출력</button></div></div></div>';
    $(".modal-close", root).onclick = closeModal;
    $(".modal-backdrop", root).onclick = (event) => { if (event.target === event.currentTarget) closeModal(); };
    $$('[data-template]', root).forEach((button) => button.onclick = () => {
      if (state.doc && state.doc.elements.length && !confirm("현재 순서도를 템플릿으로 바꿀까요? 저장하지 않은 변경사항은 사라집니다.")) return;
      const kind = button.dataset.template;
      const size = dimensions();
      beginDocumentReplacement();
      if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = null;
      state.doc = {version:"1.0",name:"새 순서도 프로젝트",image:{width:size.width,height:size.height},elements:[],relations:[],warnings:[]};
      RaduModel.normalize(state.doc);
      state.doc.name = kind === "decision" ? "조건 분기 순서도" : kind === "input" ? "입출력 순서도" : "기본 순서도";
      const cx = size.width / 2, top = Math.max(24, size.height * .12), bw = Math.min(210, size.width * .24), bh = Math.min(84, size.height * .11);
      const specs = kind === "decision"
        ? [["terminator", "시작", cx, top], ["process", "요청 처리", cx, top + bh + 60], ["decision", "조건 확인", cx, top + 2 * (bh + 60)], ["process", "예 처리", cx - bw * 1.2, top + 3 * (bh + 48)], ["process", "아니오 처리", cx + bw * 1.2, top + 3 * (bh + 48)], ["terminator", "종료", cx, top + 4 * (bh + 48)]]
        : kind === "input"
        ? [["terminator", "시작", cx, top], ["input_output", "데이터 입력", cx, top + bh + 65], ["process", "데이터 처리", cx, top + 2 * (bh + 65)], ["input_output", "결과 출력", cx, top + 3 * (bh + 65)], ["terminator", "종료", cx, top + 4 * (bh + 55)]]
        : [["terminator", "시작", cx, top], ["process", "작업 단계", cx, top + bh + 80], ["terminator", "종료", cx, top + 2 * (bh + 80)]];
      const neededHeight = Math.ceil(Math.max(...specs.map(([, , , y]) => y + bh * 1.15)) + 50);
      state.doc.image.height = Math.max(size.height, neededHeight);
      const nodes = specs.map(([shape, text, centerX, y]) => {
        const width = shape === "decision" ? bw * .9 : bw, height = shape === "decision" ? bh * 1.15 : bh,
          x = Math.max(0, Math.min(size.width - width, centerX - width / 2)), yy = y;
        const node = { id: "template-node-" + crypto.randomUUID().slice(0, 8), type: "Box", shape, text, bbox: [x, yy, x + width, yy + height], confidence: 1, source: "manual" };
        state.doc.elements.push(node); return node;
      });
      const pairs = kind === "decision" ? [[0,1,""],[1,2,""],[2,3,"예"],[2,4,"아니오"],[3,5,""],[4,5,""]] : nodes.slice(1).map((_, i) => [i, i + 1, ""]);
      pairs.forEach(([fi, ti, label]) => {
        const from = nodes[fi], to = nodes[ti], a = edgePoint(from, center(to)), b = edgePoint(to, center(from)), id = "template-arrow-" + crypto.randomUUID().slice(0, 8);
        state.doc.elements.push({ id, type: "Arrow", bbox: [Math.min(a.x,b.x), Math.min(a.y,b.y), Math.min(size.width,Math.max(a.x,b.x)+1), Math.min(state.doc.image.height,Math.max(a.y,b.y)+1)], start: a, end: b, text: label, confidence: 1, source: "manual" });
        state.doc.relations.push({ id: "template-edge-" + crypto.randomUUID().slice(0,8), type: "flows_to", source: from.id, target: to.id, arrow_id: id, label, direction: direction(center(from),center(to)), endpoint_method: "attached", status: "review", review_status: "pending" });
      });
      state.doc.name = state.doc.name || "새 순서도 프로젝트"; state.selected = null; state.selectedIds = []; state.viewMode = "restored"; state.file = null;
      state.history = []; state.historyIndex = -1; state.savedDoc = null; closeModal(); render(); fitCanvas(); dirty(true); commit();
      status("템플릿을 만들었습니다. 도형과 연결 라벨을 편집할 수 있습니다.");
    });
  }
  function copySelection() {
    if (!state.doc) return;
    const ids = new Set(
      state.selectedIds.length
        ? state.selectedIds
        : state.selected
        ? [state.selected]
        : [],
    );
    if (!ids.size) return;
    const connected = state.doc.relations.filter(r=>r.type==="flows_to"&&ids.has(r.source)&&ids.has(r.target)&&r.arrow_id);
    connected.forEach(r=>ids.add(r.arrow_id));
    const elements = state.doc.elements.filter((e) => ids.has(e.id) && !RaduModel.locked(state.doc,e));
    const elementIds = new Set(elements.map((e) => e.id));
    const relations = state.doc.relations.filter((r) =>
      (r.source == null || elementIds.has(r.source)) &&
      (r.target == null || elementIds.has(r.target)) &&
      (r.arrow_id == null || elementIds.has(r.arrow_id))
    );
    state.clipboard = {
      elements: clone(elements),
      relations: clone(relations),
    };
    status(elements.length + "개 요소를 복사했습니다.");
  }
  function pasteSelection() {
    if (!state.doc || !state.clipboard?.elements?.length) return;
    RaduModel.normalize(state.doc);
    const activeLayer = state.doc.layers.find(layer => layer.id === state.doc.active_layer);
    if (activeLayer?.locked || activeLayer?.visible === false) {
      toast("작성 레이어의 잠금과 표시 상태를 확인하세요.", true);
      return;
    }
    const copied = clone(state.clipboard.elements), size = dimensions(), b = RaduModel.bounds(copied);
    if (b[2] - b[0] > size.width || b[3] - b[1] > size.height) {
      toast("복사한 요소가 현재 캔버스보다 큽니다.", true);
      return;
    }
    const offset = (low, high, limit) => limit - high >= 24 ? 24
      : low >= 24 ? -24 : limit - high > 0 ? limit - high : -low;
    const moveX = offset(b[0], b[2], size.width);
    const moveY = offset(b[1], b[3], size.height);
    const map = new Map(), used = new Set(state.doc.elements.map((e) => e.id));
    const freshId = (prefix) => {
      let id;
      do {
        id = prefix + "-" + crypto.randomUUID().slice(0, 8);
      } while (used.has(id));
      used.add(id);
      return id;
    };
    const elements = copied.map((e) => {
      const oldId = e.id;
      e.id = freshId("copy-" + e.type.toLowerCase());
      map.set(oldId, e.id);
      RaduModel.translate(e, moveX, moveY);
      e.source = "manual";
      delete e.group_id; e.locked=false; e.layer_id=state.doc.active_layer;
      return e;
    });
    const relations = clone(state.clipboard.relations).map((r) => ({
      ...r,
      id: freshId("copy-edge"),
      source: map.get(r.source) || null,
      target: map.get(r.target) || null,
      arrow_id: map.get(r.arrow_id) || r.arrow_id,
      review_status: "pending",
      status: "review",
    }));
    state.doc.elements.push(...elements);
    state.doc.relations.push(...relations);
    state.selectedIds = elements.map((e) => e.id);
    state.selected = state.selectedIds.at(-1);
    commit();
    render();
    status(elements.length + "개 요소를 붙여넣었습니다.");
  }
  function duplicateSelection() {
    copySelection();
    pasteSelection();
  }
  function arrangeSelection(action) {
    if (!state.doc || state.selectedIds.length < 2) {
      toast("정렬하려면 요소를 두 개 이상 선택하세요.", true);
      return;
    }
    const items = state.selectedIds.map((id) =>
      state.doc.elements.find((e) => e.id === id)
    ).filter(e=>e && e.type !== "Arrow" && !RaduModel.locked(state.doc,e));
    if(items.length<2){toast("잠금 해제된 도형이나 텍스트 두 개 이상을 선택하세요.",true);return;}
    const before = JSON.stringify(state.doc),
      boxes = items.map((e) => e.bbox.slice());
    const minX = Math.min(...boxes.map((b) => b[0])),
      minY = Math.min(...boxes.map((b) => b[1]));
    const maxX = Math.max(...boxes.map((b) => b[2])),
      maxY = Math.max(...boxes.map((b) => b[3]));
    if (action === "left" || action === "center-x" || action === "right") {
      const target = action === "left"
        ? minX
        : action === "right"
        ? maxX
        : (minX + maxX) / 2;
      items.forEach((e) => {
        const b = e.bbox,
          shift = action === "left"
            ? target - b[0]
            : action === "right"
            ? target - b[2]
            : target - (b[0] + b[2]) / 2;
        e.bbox = [b[0] + shift, b[1], b[2] + shift, b[3]];
      });
    } else if (
      action === "top" || action === "center-y" || action === "bottom"
    ) {
      const target = action === "top"
        ? minY
        : action === "bottom"
        ? maxY
        : (minY + maxY) / 2;
      items.forEach((e) => {
        const b = e.bbox,
          shift = action === "top"
            ? target - b[1]
            : action === "bottom"
            ? target - b[3]
            : target - (b[1] + b[3]) / 2;
        e.bbox = [b[0], b[1] + shift, b[2], b[3] + shift];
      });
    } else if (action === "same-width" || action === "same-height") {
      const axis = action === "same-width" ? 0 : 1,
        end = axis === 0 ? 2 : 3,
        target = boxes.reduce((sum, b) => sum + b[end] - b[axis], 0) / boxes.length;
      items.forEach((e) => {
        const b = e.bbox, size = dimensions(),
          value = Math.min(target, axis === 0 ? size.width : size.height);
        e.bbox = axis === 0
          ? [b[0], b[1], Math.min(size.width, b[0] + value), b[3]]
          : [b[0], b[1], b[2], Math.min(size.height, b[1] + value)];
      });
    } else {
      const horizontal = action === "distribute-x",
        axis = horizontal ? 0 : 1,
        end = horizontal ? 2 : 3;
      const ordered = items.slice().sort((a, b) => a.bbox[axis] - b.bbox[axis]);
      if (ordered.length > 2) {
        const start = ordered[0].bbox[axis], finish = ordered.at(-1).bbox[end];
        const occupied = ordered.reduce(
          (sum, e) => sum + e.bbox[end] - e.bbox[axis],
          0,
        );
        const gap = (finish - start - occupied) / (ordered.length - 1);
        let cursor = start;
        ordered.forEach((e) => {
          const b = e.bbox, size = b[end] - b[axis], shift = cursor - b[axis];
          e.bbox = horizontal
            ? [b[0] + shift, b[1], b[2] + shift, b[3]]
            : [b[0], b[1] + shift, b[2], b[3] + shift];
          cursor += size + gap;
        });
      } else {
        toast("균등 배치에는 세 개 이상의 요소가 필요합니다.", true);
        return;
      }
    }
    items.forEach((e) => {
      const old=JSON.parse(before).elements.find(item=>item.id===e.id);
      if(e.type==="Stroke"&&old?.points){const sx=(e.bbox[2]-e.bbox[0])/(old.bbox[2]-old.bbox[0]),sy=(e.bbox[3]-e.bbox[1])/(old.bbox[3]-old.bbox[1]);e.points=old.points.map(p=>({x:e.bbox[0]+(p.x-old.bbox[0])*sx,y:e.bbox[1]+(p.y-old.bbox[1])*sy}));}
      if (e.type === "Box") syncConnectedArrows(e.id);
      markRelations(e.id);
    });
    if (JSON.stringify(state.doc) !== before) commit();
    render();
    status("선택 요소 정렬을 적용했습니다.");
  }
  function openArrangeMenu() {
    const root = $("#modal-root");
    const choices = [
      ["left", "왼쪽 맞춤"],
      ["center-x", "가로 중앙 맞춤"],
      ["right", "오른쪽 맞춤"],
      ["top", "위쪽 맞춤"],
      ["center-y", "세로 중앙 맞춤"],
      ["bottom", "아래쪽 맞춤"],
      ["distribute-x", "가로 균등 배치"],
      ["distribute-y", "세로 균등 배치"],
      ["same-width", "너비 동일하게"],
      ["same-height", "높이 동일하게"],
    ];
    root.innerHTML =
      '<div class="modal-backdrop"><div class="modal" role="dialog" aria-modal="true"><div class="modal-head"><h2>정렬 및 간격</h2><button class="modal-close" aria-label="닫기">×</button></div><div class="arrange-grid">' +
      choices.map(([id, label]) =>
        '<button class="button" data-arrange="' + id + '">' + label +
        "</button>"
      ).join("") +
      '</div><p class="hint">여러 요소를 선택한 뒤 정렬하세요. 연결 화살표는 도형 위치에 맞춰 갱신됩니다.</p></div></div>';
    $(".modal-close", root).onclick = closeModal;
    $(".modal-backdrop", root).onclick = (e) => {
      if (e.target === e.currentTarget) closeModal();
    };
    $$("[data-arrange]", root).forEach((button) =>
      button.onclick = () => {
        const action = button.dataset.arrange;
        closeModal();
        arrangeSelection(action);
      }
    );
  }
  async function chooseFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/") &&
        !/\.(?:svg|heic|heif|avif|tif|tiff|bmp|dib|gif|jfif|jpe|jp2|j2k|tga|pcx|psd|ico|cur|ppm|pgm|pbm|pnm|qoi)$/i.test(file.name)) {
      toast("SVG 또는 지원되는 래스터 이미지 파일을 선택해 주세요. PDF는 아직 지원하지 않습니다.", true);
      return;
    }
    let url = URL.createObjectURL(file);
    let fileDimensions = null;
    try {
      try {
        const img = new Image();
        img.src = url;
        await img.decode();
        fileDimensions = { width: img.naturalWidth, height: img.naturalHeight };
      } catch (error) {
        URL.revokeObjectURL(url);
        url = null;
      }
      if (
        state.dirty && state.doc &&
        !confirm("저장하지 않은 변경사항이 있습니다. 새 이미지를 불러올까요?")
      ) {
        if (url) URL.revokeObjectURL(url);
        return;
      }
      if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = url;
      state.viewMode = "overlay";
      state.file = file;
      state.file.dimensions = fileDimensions;
      state.doc = null;
      beginDocumentReplacement();
      state.selected = null;
      state.selectedIds = [];
      state.history = [];
      state.historyIndex = -1;
      dirty(true);
      render();
      fitCanvas();
      status(fileDimensions
        ? "이미지가 준비됐습니다. 분석 엔진을 선택하세요."
        : "브라우저 미리보기는 지원하지 않지만, 서버 디코딩을 위해 업로드할 수 있습니다.");
    } catch (e) {
      if (url) URL.revokeObjectURL(url);
      toast(e.message || "이미지를 열 수 없습니다.", true);
    }
  }
  async function analyze() {
    if (state.busy) return;
    if (!state.file) {
      toast("먼저 순서도 이미지를 선택하세요.", true);
      return;
    }
    state.busy = true;
    $("#analyze").disabled = true;
    $("#retry-analysis").hidden = true;
    $("#analysis-state").textContent = "분석 중";
    const overlay = document.createElement("div");
    overlay.id = "busy-overlay";
    overlay.className = "busy-overlay";
    overlay.innerHTML =
      '<div class="busy-card"><i class="spinner"></i><span>이미지를 분석하고 있습니다…</span></div>';
    $("#canvas-scroll").append(overlay);
    status("이미지를 전처리하고 요소·관계를 분석하고 있습니다.", "busy");
    try {
      const query = new URLSearchParams({
        ocr_engine: $("#ocr-engine").value,
        device: $("#device").value,
      });
      const response = await request("/api/analyze?" + query.toString(), {
          method: "POST",
          headers: {
            "Content-Type": state.file.type || "application/octet-stream",
            "X-File-Name": encodeURIComponent(state.file.name),
          },
          body: state.file,
        }),
        doc = await response.json();
      validateImportedDoc(doc);
      beginDocumentReplacement();
      state.doc = doc;
      state.viewMode = doc.ocr ? "ocr" : doc.image.source_url ? "overlay" : "restored";
      state.selected = null;
      state.selectedIds = [];
      $("#retry-analysis").hidden = true;
      resetHistory();
      render();
      fitCanvas();
      const ocrWarning = doc.warnings?.find((warning) => warning.code === "ocr_unavailable"),
        ocrFailed = $("#ocr-engine").value !== "none" && !doc.ocr;
      if (ocrFailed) {
        const message = ocrWarning?.message ||
          "선택한 OCR 엔진이 결과를 반환하지 않았습니다.";
        const raduSkipped = doc.radu?.status === "not_run";
        status(
          (raduSkipped
            ? "PaddleOCR-VL 실패 · RADU/OpenCV 미실행. "
            : "OCR 실패 · OpenCV 후보만 표시 중입니다. ") + message,
          "error",
        );
        toast(raduSkipped
          ? "PaddleOCR-VL 추론에 실패했습니다. 결과가 표시되지 않습니다."
          : "OCR 추론에 실패했습니다. 현재 문서에는 OpenCV 후보만 있습니다.", true);
      } else if (doc.radu?.status === "not_run") {
        const metadata = doc.ocr.metadata || {},
          lineCount = doc.ocr.lines?.length || 0;
        status(
          "PaddleOCR-VL 완료 · " + (metadata.device || "장치 확인 불가") +
            " · 텍스트 " + lineCount + "줄 · RADU 미실행",
        );
        toast("PaddleOCR-VL 텍스트만 오버레이했습니다. RADU 검출·관계 복원은 실행하지 않았습니다.");
      } else {
        status(
          "분석 완료 · 요소 " + doc.elements.length + "개 · 관계 " +
            doc.relations.length + "개",
        );
        toast("분석이 끝났습니다. 검토 항목을 확인해 주세요.");
      }
    } catch (e) {
      $("#analysis-state").textContent = "실패";
      $("#retry-analysis").hidden = false;
      status("분석 실패: " + e.message, "error");
      toast(e.message, true);
    } finally {
      state.busy = false;
      $("#analyze").disabled = false;
      $("#busy-overlay")?.remove();
    }
  }
  async function analyzeRelations() {
    if (state.busy) return;
    if (!state.doc || !state.doc.elements.length) {
      toast("먼저 요소가 있는 순서도를 만들거나 불러오세요.", true);
      return;
    }
    state.busy = true;
    $("#analyze-relations").disabled = true;
    $("#analysis-state").textContent = "관계 분석 중";
    const overlay = document.createElement("div");
    overlay.id = "busy-overlay";
    overlay.className = "busy-overlay";
    overlay.innerHTML =
      '<div class="busy-card"><i class="spinner"></i><span>현재 순서도의 도형·텍스트·화살표 관계를 분석하고 있습니다…</span></div>';
    $("#canvas-scroll").append(overlay);
    status("현재 순서도의 요소와 화살표를 분석하고 있습니다.", "busy");
    const generation = state.docGeneration;
    const revision = state.editRevision;
    try {
      const response = await request("/api/analyze-relations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(clone(state.doc)),
        }),
        result = await response.json();
      validateImportedDoc(result);
      if (generation !== state.docGeneration || revision !== state.editRevision) {
        throw new Error("분석 중 문서가 바뀌어 결과를 적용하지 않았습니다. 다시 분석해 주세요.");
      }
      state.doc = result;
      commit();
      render();
      status(
        "관계 분석 완료 · 요소 " + result.elements.length + "개 · 관계 " +
          result.relations.length + "개",
      );
      toast("관계 분석이 끝났습니다. 검토 항목을 확인해 주세요.");
    } catch (error) {
      $("#analysis-state").textContent = "실패";
      status("관계 분석 실패: " + error.message, "error");
      toast(error.message, true);
    } finally {
      state.busy = false;
      $("#analyze-relations").disabled =
        !state.doc || !state.doc.elements.length;
      $("#busy-overlay")?.remove();
    }
  }
  function validateImportedDoc(doc, nestedPage = false) {
    if (
      !doc || typeof doc !== "object" || !doc.image ||
      !Array.isArray(doc.elements) || !Array.isArray(doc.relations)
    ) throw new Error("JSON에 image, elements, relations 필드가 필요합니다.");
    if (
      !Number.isFinite(doc.image.width) || !Number.isFinite(doc.image.height) ||
      doc.image.width <= 0 || doc.image.height <= 0
    ) throw new Error("JSON의 캔버스 크기가 올바르지 않습니다.");
    const ids = new Set();
    for (const e of doc.elements) {
      if (
        !e || typeof e.id !== "string" || ids.has(e.id) ||
        !["Text", "Box", "Arrow", "Image", "Stroke"].includes(e.type) || !Array.isArray(e.bbox) ||
        e.bbox.length !== 4 || !e.bbox.every(Number.isFinite) ||
        e.bbox[0] < 0 || e.bbox[1] < 0 || e.bbox[2] <= e.bbox[0] ||
        e.bbox[3] <= e.bbox[1] || e.bbox[2] > doc.image.width ||
        e.bbox[3] > doc.image.height ||
        (e.type === "Box" &&
          !["process", "decision", "terminator", "input_output", "document",
            "database", "predefined_process", "manual_input", "preparation",
            "connector", "offpage_connector", "delay", undefined].includes(e.shape))
      ) throw new Error("JSON에 잘못된 요소, 유형 또는 중복 ID가 있습니다.");
      if (e.style !== undefined) {
        if (!e.style || typeof e.style !== "object" || Array.isArray(e.style)) {
          throw new Error("요소 스타일 데이터 형식이 올바르지 않습니다.");
        }
        for (const key of ["fill", "stroke", "text_color"]) {
          if (e.style[key] !== undefined && !/^#(?:[\da-f]{3}|[\da-f]{4}|[\da-f]{6}|[\da-f]{8})$/i.test(e.style[key])) {
            throw new Error("JSON 색상 값이 올바르지 않습니다.");
          }
        }
        if (e.style.text_align !== undefined && !["left", "center", "right"].includes(e.style.text_align)) {
          throw new Error("JSON 텍스트 정렬 값이 올바르지 않습니다.");
        }
        if (e.style.line_style !== undefined && !["solid", "dashed", "dotted"].includes(e.style.line_style)) {
          throw new Error("JSON 선 종류 값이 올바르지 않습니다.");
        }
        if (e.style.marker_start !== undefined && !["none", "arrow", "diamond"].includes(e.style.marker_start) ||
            e.style.marker_end !== undefined && !["none", "arrow", "diamond"].includes(e.style.marker_end)) {
          throw new Error("JSON 화살촉 종류가 올바르지 않습니다.");
        }
        if (e.style.opacity !== undefined && (!Number.isFinite(e.style.opacity) || e.style.opacity < .1 || e.style.opacity > 1)) {
          throw new Error("JSON 불투명도가 올바르지 않습니다.");
        }
        for (const [key, min, max] of [["stroke_width", 1, 30], ["font_size", 8, 96]]) {
          if (e.style[key] !== undefined && (!Number.isFinite(e.style[key]) || e.style[key] < min || e.style[key] > max)) {
            throw new Error("JSON 스타일 숫자 값이 허용 범위를 벗어났습니다.");
          }
        }
      }
      if (e.type === "Arrow" && e.route !== undefined && !["straight", "orthogonal", "curved"].includes(e.route)) {
        throw new Error("JSON 화살표 경로가 올바르지 않습니다.");
      }
      if(e.type === "Image" && (typeof e.image_data !== "string" || !/^data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(e.image_data) || e.image_data.length > 8000000)) throw new Error("삽입된 이미지가 올바르지 않습니다.");
      if(e.type === "Stroke" && (!Array.isArray(e.points) || e.points.length<2 || e.points.length>20000 || e.points.some(p=>!Number.isFinite(p?.x)||!Number.isFinite(p?.y)||p.x<0||p.y<0||p.x>doc.image.width||p.y>doc.image.height)))throw new Error("펜 선 좌표가 올바르지 않습니다.");
      ids.add(e.id);
      if (
        e.type === "Arrow" && (e.start !== undefined || e.end !== undefined)
      ) {
        for (const endpoint of [e.start, e.end]) {
          const values = Array.isArray(endpoint)
            ? endpoint
            : [endpoint && endpoint.x, endpoint && endpoint.y];
          if (
            values.length < 2 || !Number.isFinite(values[0]) ||
            !Number.isFinite(values[1]) || values[0] < 0 || values[1] < 0 ||
            values[0] > doc.image.width || values[1] > doc.image.height
          ) throw new Error("화살표 좌표가 캔버스 범위를 벗어났습니다.");
        }
      }
    }
    const relationIds = new Set();
    const attachedArrowIds = new Set();
    const elementById = new Map(doc.elements.map(e=>[e.id,e]));
    for (const r of doc.relations) {
      if (
        !r || typeof r.id !== "string" || !r.id || relationIds.has(r.id) ||
        !["flows_to", "contains", "near_text"].includes(r.type) ||
        (r.source != null && !ids.has(r.source)) ||
        (r.target != null && !ids.has(r.target)) ||
        (r.arrow_id && elementById.get(r.arrow_id)?.type !== "Arrow") ||
        (r.type === "flows_to" && (r.source != null && elementById.get(r.source)?.type !== "Box" ||
          r.target != null && elementById.get(r.target)?.type !== "Box"))
      ) {
        throw new Error(
          "JSON 관계가 잘못되었거나 존재하지 않는 요소를 참조합니다.",
        );
      }
      relationIds.add(r.id);
      if (r.type === "flows_to" && r.arrow_id) {
        if (attachedArrowIds.has(r.arrow_id)) throw new Error("하나의 화살표가 여러 흐름 관계에 연결되어 있습니다.");
        attachedArrowIds.add(r.arrow_id);
      }
    }
    if(doc.layers!==undefined){
      if(!Array.isArray(doc.layers)||!doc.layers.length||doc.layers.length>100||doc.layers.some(l=>!l||typeof l.id!=="string"||!l.id))throw new Error("레이어 목록이 올바르지 않습니다.");
      const layerIds=new Set(doc.layers.map(l=>l.id));
      if(layerIds.size!==doc.layers.length||!layerIds.has(doc.active_layer)||doc.elements.some(e=>!layerIds.has(e.layer_id)))throw new Error("레이어 ID가 올바르지 않습니다.");
    }
    if (doc.canvas !== undefined) {
      const canvas = doc.canvas;
      if (!canvas || typeof canvas !== "object" || Array.isArray(canvas) ||
          canvas.grid_size !== undefined && (!Number.isFinite(canvas.grid_size) || canvas.grid_size < 2 || canvas.grid_size > 100) ||
          canvas.line_jump !== undefined && !["none", "arc", "gap", "sharp"].includes(canvas.line_jump) ||
          ["background", "grid_color"].some(key=>canvas[key]!==undefined && !/^#[0-9a-f]{6}$/i.test(canvas[key])) ||
          ["grid", "ruler"].some(key=>canvas[key]!==undefined && typeof canvas[key]!=="boolean")) {
        throw new Error("캔버스 설정이 올바르지 않습니다.");
      }
    }
    if(!nestedPage){if(doc.pages!==undefined&&!Array.isArray(doc.pages))throw new Error("페이지 목록이 올바르지 않습니다.");const pages=doc.pages||[];if(pages.length>50)throw new Error("페이지는 최대 50개입니다.");for(const page of pages)validateImportedDoc(page,true);const pageIds=[doc.active_page||"page-1",...pages.map(p=>p.id)];if(new Set(pageIds).size!==pageIds.length)throw new Error("페이지 ID가 중복되었습니다.");}
    doc.warnings = Array.isArray(doc.warnings) ? doc.warnings : [];
    return doc;
  }
  async function importJson(file) {
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) {
      toast("JSON 파일은 10 MB 이하여야 합니다.", true);
      return;
    }
    if (
      state.dirty &&
      !confirm("저장하지 않은 변경사항을 버리고 JSON을 여시겠습니까?")
    ) return;
    try {
      const doc = validateImportedDoc(JSON.parse(await file.text()));
      beginDocumentReplacement();
      state.doc = doc;
      state.file = null;
      if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = null;
      state.selected = null;
      state.selectedIds = [];
      state.viewMode = "restored";
      resetHistory();
      render();
      fitCanvas();
      status("JSON 문서를 불러왔습니다.");
      toast("JSON 문서를 열었습니다.");
    } catch (error) {
      toast("JSON을 불러오지 못했습니다: " + error.message, true);
    } finally {
      $("#json-file").value = "";
    }
  }
  async function saveDocument(options = {}) {
    const active = document.activeElement;
    if (
      state.tab === "properties" && active &&
      active.closest("#inspector-content") && active.id.startsWith("prop-")
    ) applyChanges();
    if (!state.doc) {
      if (!options.silent) toast("저장할 문서가 없습니다.", true);
      return;
    }
    if (state.autoSaving) {
      if (options.automatic) {
        clearTimeout(scheduleLocalDraft.serverTimer);
        scheduleLocalDraft.serverTimer = setTimeout(() => saveDocument(options), 1200);
      } else state.manualSaveQueued = true;
      return;
    }
    state.doc.name = $("#project-name").value.trim() || "이름 없는 순서도";
    if (
      state.dirty && state.historyIndex >= 0 &&
      JSON.stringify(state.doc) !==
        JSON.stringify(state.history[state.historyIndex])
    ) commit();
    const snapshot = clone(state.doc), revision = state.editRevision,
      generation = state.docGeneration;
    state.autoSaving = true;
    try {
      const response = await request(hybrid ? "/api/jobs/" + hybrid.jobId + "/vector?reviewed=" + !options.automatic : "/api/save", {
          method: hybrid ? "PUT" : "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(snapshot),
        }),
        result = await response.json();
      if (state.doc && state.docGeneration === generation) {
        state.doc.session_id = result.session_id;
        state.doc.updated_at = result.updated_at;
        state.doc.revision = result.revision;
      }
      snapshot.session_id = result.session_id;
      snapshot.updated_at = result.updated_at;
      snapshot.revision = result.revision;
      if (state.docGeneration === generation && state.editRevision === revision) {
        state.savedDoc = clone(snapshot);
        dirty(false);
        try { localStorage.removeItem(draftKey); } catch (storageError) {
          console.warn("Saved document but could not clear browser recovery draft", storageError);
        }
      } else if (state.docGeneration === generation) {
        scheduleLocalDraft();
      }
      if (!options.silent) {
        render();
        status("편집본을 서버에 저장했습니다.");
        toast("문서를 저장했습니다.");
      } else if (state.docGeneration === generation && state.editRevision === revision) {
        $("#save-state").textContent = "자동 저장됨";
      }
    } catch (e) {
      if (state.docGeneration !== generation) {
        console.warn("A save for a replaced RADU document failed", e);
      } else if (!options.silent) {
        status("저장 실패: " + e.message, "error");
        toast(e.message, true);
      } else if (e.status === 409) {
        $("#save-state").textContent = "저장 충돌 · JSON 내보내기 후 다시 열기";
        status(e.message, "error");
        toast(e.message, true);
      } else {
        $("#save-state").textContent = "자동 저장 실패 · 재시도 중";
        console.warn("RADU automatic server save failed", e);
        clearTimeout(scheduleLocalDraft.serverTimer);
        scheduleLocalDraft.serverTimer = setTimeout(() => saveDocument(options), 5000);
      }
    } finally {
      state.autoSaving = false;
      if (state.manualSaveQueued) {
        state.manualSaveQueued = false;
        setTimeout(() => saveDocument(), 0);
      }
    }
  }
  async function exportDocument(format, options = {}) {
    if (!state.doc) {
      toast("먼저 이미지를 분석해 주세요.", true);
      return;
    }
    try {
      const query = new URLSearchParams({scale:String(options.scale||1),transparent:String(!!options.transparent),all_pages:String(!!options.all_pages)});
      const response = await request("/api/export/" + format + "?" + query, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(state.doc),
        }),
        blob = await response.blob(),
        url = URL.createObjectURL(blob),
        a = document.createElement("a");
      a.href = url;
      a.download = (hybrid ? "HybridOCR." : "radu-flowchart.") + format;
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      status(format.toUpperCase() + " 내보내기가 완료됐습니다.");
      toast(format.toUpperCase() + " 파일을 내보냈습니다.");
    } catch (e) {
      status("내보내기 실패: " + e.message, "error");
      toast(e.message, true);
    }
  }
  function closeModal() {
    $("#modal-root").replaceChildren();
  }
  async function loadSession(id) {
    if (state.doc && state.doc.session_id === id) {
      closeModal();
      return;
    }
    if (
      state.dirty &&
      !confirm("저장하지 않은 변경사항을 버리고 문서를 여시겠습니까?")
    ) return;
    try {
      const response = await request("/api/sessions/" + encodeURIComponent(id));
      const doc = await response.json();
      beginDocumentReplacement();
      state.doc = doc;
      state.viewMode = doc.image.source_url ? "overlay" : "restored";
      state.file = null;
      if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = null;
      state.selected = null;
      state.selectedIds = [];
      resetHistory();
      render();
      fitCanvas();
      closeModal();
      status("저장된 문서를 열었습니다.");
    } catch (e) {
      toast(e.message, true);
    }
  }
  async function openSessions() {
    try {
      const response = await request("/api/sessions"),
        data = await response.json(),
        root = $("#modal-root");
      root.innerHTML =
        '<div class="modal-backdrop"><div class="modal" role="dialog" aria-modal="true"><div class="modal-head"><h2>저장된 문서</h2><button class="modal-close" aria-label="닫기">×</button></div><label class="field"><span>문서 검색</span><input id="session-search" class="control" type="search" placeholder="이름으로 찾기"></label><div id="session-list"></div></div></div>';
      $(".modal-close", root).onclick = closeModal;
      $(".modal-backdrop", root).onclick = (e) => {
        if (e.target === e.currentTarget) closeModal();
      };
      const list = $("#session-list", root);
      if (!data.sessions || !data.sessions.length) {
        list.innerHTML =
          '<div class="empty-list">저장된 문서가 없습니다.</div>';
        return;
      }
      const renderSessions = (query = "") => {
        const matches = data.sessions.filter(item => String(item.name || "").toLocaleLowerCase().includes(query.toLocaleLowerCase()));
        if (!matches.length) { list.innerHTML = '<div class="empty-list">검색 결과가 없습니다.</div>'; return; }
        list.innerHTML = matches.map((item) =>
        '<div class="session-row"><button class="row-main" data-open="' +
        esc(item.session_id) +
        '" style="border:0;background:none;text-align:left"><span class="row-label">' +
        esc(item.name) + '</span><span class="row-sub">요소 ' +
        item.element_count + "개 · " +
        new Date(item.updated_at).toLocaleString("ko-KR") +
        '</span></button><button class="button" data-duplicate="' + esc(item.session_id) + '">복제</button><button class="button" data-share="' + esc(item.session_id) + '">링크</button><button class="button danger" data-delete="' +
        esc(item.session_id) + '">삭제</button></div>'
        ).join("");
      $$("[data-open]", list).forEach((b) =>
        b.onclick = () => loadSession(b.dataset.open)
      );
      $$('[data-share]',list).forEach(b=>b.onclick=async()=>{
        const url=new URL(location.href);url.searchParams.set('session',b.dataset.share);
        try{await navigator.clipboard.writeText(url.href);toast('문서 링크를 복사했습니다. 같은 RADU 서버에 접근할 수 있는 사람만 열 수 있습니다.');}
        catch(error){toast('링크 복사 권한이 없습니다: '+url.href,true);}
      });
      $$('[data-duplicate]',list).forEach(b=>b.onclick=async()=>{
        try{const result=await(await request('/api/sessions/'+b.dataset.duplicate+'/duplicate',{method:'POST'})).json();await loadSession(result.session_id);}
        catch(error){toast(error.message,true);}
      });
      $$("[data-delete]", list).forEach((b) =>
        b.onclick = async () => {
          if (!confirm("이 문서를 삭제할까요?")) return;
          try {
            await request("/api/sessions/" + b.dataset.delete, {
              method: "DELETE",
            });
            toast("문서를 삭제했습니다.");
            openSessions();
          } catch (e) {
            toast(e.message, true);
          }
        }
      );
      };
      renderSessions();
      $("#session-search", root).oninput = event => renderSessions(event.target.value.trim());
    } catch (e) {
      toast(e.message, true);
    }
  }
  function exportMenu() {
    if (!state.doc) { toast("먼저 순서도를 만들거나 이미지를 분석해 주세요.", true); return; }
    const root = $("#modal-root");
    root.innerHTML = '<div class="modal-backdrop"><div class="modal" role="dialog" aria-modal="true"><div class="modal-head"><h2>내보내기</h2><button class="modal-close" aria-label="닫기">×</button></div><p class="hint">JSON은 모든 페이지와 편집 데이터를 보존합니다. SVG·PNG는 현재 페이지를 출력합니다. PDF는 모든 페이지를 선택할 수 있습니다.</p><label class="field"><span>PNG 해상도</span><select class="control" id="export-scale"><option value="1">원본 크기</option><option value="2">2배</option><option value="4">4배</option></select></label><label class="field"><span><input type="checkbox" id="export-transparent"> SVG·PNG 투명 배경</span></label><label class="field"><span><input type="checkbox" id="export-all-pages" checked> PDF에 모든 페이지 포함</span></label><div class="button-row" style="flex-wrap:wrap"><button class="button" id="export-preview">미리보기</button>' + ["json", "svg", "png", "pdf"].map(f=>'<button class="button" data-format="'+f+'">'+f.toUpperCase()+'</button>').join('') + '</div><div id="export-preview-result" class="export-preview"></div></div></div>';
    const close=()=>{const img=$("#export-preview-result img",root);if(img?.src?.startsWith("blob:"))URL.revokeObjectURL(img.src);root.replaceChildren();};
    $(".modal-close",root).onclick=close;
    $(".modal-backdrop",root).onclick=e=>{if(e.target===e.currentTarget)close();};
    const options=()=>({scale:Number($("#export-scale",root).value),transparent:$("#export-transparent",root).checked,all_pages:$("#export-all-pages",root).checked});
    $("#export-preview",root).onclick=async()=>{
      const target=$("#export-preview-result",root);target.textContent="출력 결과를 생성하는 중…";
      try{const response=await request("/api/export/svg?transparent="+options().transparent,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(state.doc)});const url=URL.createObjectURL(await response.blob());target.innerHTML='<img alt="현재 페이지 출력 미리보기"><p class="hint">선택 테두리·조절 핸들 없이 출력되는 현재 페이지입니다.</p>';target.querySelector('img').src=url;}catch(error){target.textContent='미리보기를 만들 수 없습니다: '+error.message;}
    };
    $$('[data-format]',root).forEach(b=>b.onclick=()=>{const format=b.dataset.format,settings=options();close();exportDocument(format,settings);});
  }
  function applyChanges() {
    const before = JSON.stringify(state.doc),
      root = $("#inspector-content"),
      e = state.doc &&
        state.doc.elements.find((item) => item.id === state.selected);
    if (!e) return;
    if(RaduModel.locked(state.doc,e)){toast("잠금을 해제한 후 편집하세요.",true);render();return;}
    const x = Number($("#prop-x", root)?.value),
      y = Number($("#prop-y", root)?.value),
      w = Number($("#prop-w", root)?.value),
      h = Number($("#prop-h", root)?.value);
    const size = dimensions();
    if (
      ![x, y, w, h].every(Number.isFinite) || x < 0 || y < 0 || w <= 0 ||
      h <= 0 || x + w > size.width || y + h > size.height
    ) {
      toast("위치와 크기는 캔버스 안의 유효한 값이어야 합니다.", true);
      return;
    }
    if (e.type === "Text") e.text = $("#prop-text", root)?.value ?? e.text;
    else e.text = $("#prop-label", root)?.value ?? e.text;
    if (e.type === "Box") e.shape = $("#prop-shape", root)?.value || e.shape;
    if (e.type === "Arrow") {
      e.route = $("#prop-route", root)?.value || "straight";
      e.style = e.style || {};
      e.style.line_style = $("#prop-line-style", root)?.value || "solid";
      e.style.marker_start = $("#prop-marker-start", root)?.value || "none";
      e.style.marker_end = $("#prop-marker-end", root)?.value || "arrow";
      e.text = $("#prop-arrow-label", root)?.value ?? "";
      const relation = state.doc.relations.find((item) => item.arrow_id === e.id);
      if (relation) relation.label = e.text;
    }
    e.style = e.style || {};
    const fill = $("#prop-fill", root), stroke = $("#prop-stroke", root),
      strokeWidth = Number($("#prop-stroke-width", root)?.value),
      fontSize = Number($("#prop-font-size", root)?.value),
      textColor = $("#prop-text-color", root),
      textAlign = $("#prop-text-align", root),
      rotation = $("#prop-rotation", root);
    if (fill) e.style.fill = safeColor(fill.value, "#ffffff");
    if (stroke) e.style.stroke = safeColor(stroke.value, "#334057");
    if (Number.isFinite(strokeWidth)) e.style.stroke_width = Math.max(1,Math.min(30,strokeWidth));
    if (Number.isFinite(fontSize)) e.style.font_size = Math.max(8,Math.min(96,fontSize));
    if (textColor) e.style.text_color = safeColor(textColor.value, "#334057");
    if (textAlign) e.style.text_align = textAlign.value;
    if (rotation) e.rotation = Math.max(-180,Math.min(180,Number(rotation.value)||0));
    if (e.source && e.source !== "manual") e.source = "manual-edit";
    const old = e.bbox.slice(), next = [x, y, x + w, y + h];
    if (e.type === "Arrow" && e.start && e.end) {
      const move = (raw) => {
        const p = point(raw, { x: old[0], y: old[1] }),
          rx = (p.x - old[0]) / Math.max(1, old[2] - old[0]),
          ry = (p.y - old[1]) / Math.max(1, old[3] - old[1]);
        return { x: x + rx * w, y: y + ry * h };
      };
      e.start = move(e.start);
      e.end = move(e.end);
    }
    e.bbox = next;
    const c = $("#prop-confidence", root);
    if (c && !c.disabled) e.confidence = Number(c.value);
    if (e.type === "Box") syncConnectedArrows(e.id);
    else if (e.type === "Arrow") attachArrowByGeometry(e);
    else markRelations(e.id);
    if (JSON.stringify(state.doc) !== before) commit();
    render();
    toast("요소 변경사항을 적용했습니다.");
  }
  $$(".tool[data-tool]").forEach((b) =>
    b.onclick = () => selectTool(b.dataset.tool)
  );
  $("#shape-kind").onchange = () => {
    state.shapeKind = $("#shape-kind").value;
    const icons = {
      process: "□", terminator: "⬭", decision: "◇", input_output: "▱",
      document: "▤", database: "▤", predefined_process: "▣",
      manual_input: "▱", preparation: "⬡", connector: "○", delay: "D",
    };
    $("[data-shape-tool]").innerHTML = shapeIcon(state.shapeKind);
    $("[data-shape-tool]").setAttribute("aria-label", kindName({ type: "Box", shape: state.shapeKind }) + " 그리기");
    if (state.tool === "shape") {
      status(kindName({ type: "Box", shape: state.shapeKind }) +
        "을 그립니다. 캔버스에서 드래그하세요.");
    }
  };
  $("#element-filter").onchange = renderList;
  $("#replace-image").onclick = () => $("#image-file").click();
  $("#image-file").onchange = (e) => {
    const file = e.target.files && e.target.files[0];
    e.target.value = "";
    chooseFile(file);
  };
  $("#upload-box").ondragover = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add("dragover");
  };
  $("#upload-box").ondragleave = (e) =>
    e.currentTarget.classList.remove("dragover");
  $("#upload-box").ondrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove("dragover");
    chooseFile(e.dataTransfer.files && e.dataTransfer.files[0]);
  };
  $("#analyze").onclick = analyze;
  $("#analyze-relations").onclick = analyzeRelations;
  $("#retry-analysis").onclick = analyze;
  $("#save").onclick = saveDocument;
  $("#export-menu").onclick = exportMenu;
  $("#open-sessions").onclick = openSessions;
  $("#delete-selected").onclick = deleteSelected;
  $("#apply").onclick = applyChanges;
  $("#add-process").onclick = addProcess;
  $("#undo").onclick = undo;
  $("#redo").onclick = redo;
  $("#arrange").onclick = openArrangeMenu;
  $("#templates").onclick = openTemplateMenu;
  $("#grid-snap").onclick = () => {
    state.gridSnap = !state.gridSnap;
    $("#grid-snap").classList.toggle("active", state.gridSnap);
    status(
      state.gridSnap
        ? "요소를 10픽셀 격자에 맞춥니다."
        : "격자 맞춤을 해제했습니다.",
    );
  };
  $$(".tabs button").forEach((b) =>
    b.onclick = () => {
      state.tab = b.dataset.tab;
      renderInspector();
      editorTools?.onRender();
      editorTools?.onRender();
    }
  );
  $("#zoom-in").onclick = () => {
    state.fitMode = false;
    state.zoom = Math.min(3, state.zoom * 1.15);
    setCanvasSize();
  };
  $("#zoom-out").onclick = () => {
    state.fitMode = false;
    state.zoom = Math.max(.12, state.zoom / 1.15);
    setCanvasSize();
  };
  $("#fit").onclick = fitCanvas;
  $("#project-name").oninput = () => {
    if (state.doc) {
      state.doc.name = $("#project-name").value;
      dirty(true);
      scheduleLocalDraft();
      state.editRevision += 1;
    }
  };
  $("#new-blank").onclick = newBlank;
  $("#new-from-image").onclick = () => $("#image-file").click();
  $("#open-json").onclick = () => $("#json-file").click();
  $("#json-file").onchange = (e) => {
    const file = e.target.files && e.target.files[0];
    e.target.value = "";
    importJson(file);
  };
  $("#view-mode").onclick = () => {
    state.viewMode = ({
      overlay: "original",
      original: "restored",
      restored: "compare",
      compare: "overlay",
      ocr: "overlay",
      radu: "overlay",
    })[state.viewMode] || "overlay";
    render();
    if (state.fitMode) fitCanvas();
    status(
      state.viewMode === "compare"
        ? "원본과 복원 결과를 같은 배율로 나란히 표시합니다."
        : state.viewMode === "overlay"
        ? "분석 오버레이를 표시합니다."
        : state.viewMode === "original"
        ? "전처리 전 원본을 표시합니다."
        : "복원된 다이어그램만 표시합니다.",
    );
  };
  $("#ocr-view").onclick = () => {
    if (!state.doc?.ocr) return;
    state.viewMode = "ocr";
    state.selected = null;
    state.selectedIds = [];
    render();
    if (state.fitMode) fitCanvas();
    status("선택한 OCR 엔진이 인식한 텍스트와 위치만 표시합니다.");
  };
  $("#radu-view").onclick = () => {
    if (!state.doc?.ocr) return;
    state.viewMode = "radu";
    state.selected = null;
    state.selectedIds = [];
    render();
    if (state.fitMode) fitCanvas();
    status(state.doc.radu?.status === "not_run"
      ? "RADU 미실행 상태입니다. 현재는 OCR 텍스트 오버레이만 표시합니다."
      : "같은 OCR 텍스트에 RADU 도형·화살표 검출과 관계 복원을 함께 표시합니다.");
  };
  window.addEventListener("resize", () => {
    if (state.fitMode) fitCanvas();
  });
  window.addEventListener("keydown", (e) => {
    const tag = document.activeElement &&
        document.activeElement.tagName.toLowerCase(),
      editing = ["input", "textarea", "select"].includes(tag);
    if (e.code === "Space" && !editing) {
      state.spacePan = true;
      $("#diagram").style.cursor = "grab";
      e.preventDefault();
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      saveDocument();
      return;
    }
    if (e.key === "Escape" && $(".modal-backdrop")) {
      e.preventDefault();
      closeModal();
      return;
    }
    if (editing) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "a") {
      if (state.doc) {
        e.preventDefault();
        state.selectedIds = state.doc.elements.filter(item=>RaduModel.visible(state.doc,item)&&!RaduModel.locked(state.doc,item)).map((item) => item.id);
        state.selected = state.selectedIds.at(-1) || null;
        render();
      }
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "c") {
      e.preventDefault();
      copySelection();
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "v") {
      e.preventDefault();
      pasteSelection();
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "d") {
      e.preventDefault();
      duplicateSelection();
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
      e.preventDefault();
      e.shiftKey ? redo() : undo();
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
      e.preventDefault();
      redo();
    } else if (e.key === "Delete" || e.key === "Backspace") deleteSelected();
    else if (
      ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key) &&
      state.doc && state.selectedIds.length
    ) {
      e.preventDefault();
      const delta = e.shiftKey ? 10 : 1, size = dimensions();
      state.selectedIds.forEach((id) => {
        const item = state.doc.elements.find((candidate) =>
          candidate.id === id
        );
        if (!item || item.type === "Arrow" || RaduModel.locked(state.doc,item)) return;
        const dx = e.key === "ArrowLeft"
          ? -delta
          : e.key === "ArrowRight"
          ? delta
          : 0;
        const dy = e.key === "ArrowUp"
          ? -delta
          : e.key === "ArrowDown"
          ? delta
          : 0;
        const mx = Math.max(
          -item.bbox[0],
          Math.min(size.width - item.bbox[2], dx),
        );
        const my = Math.max(
          -item.bbox[1],
          Math.min(size.height - item.bbox[3], dy),
        );
        item.bbox = item.bbox.map((v, i) => v + (i % 2 === 0 ? mx : my));
        if (item.type === "Box") syncConnectedArrows(id);
        markRelations(id);
      });
      commit();
      render();
    } else if (e.key === "Escape") {
      if (state.drag) {
        state.doc = state.drag.before;
        state.drag = null;
        state.guides = [];
      }
      if (state.draw || state.marquee) {
        state.draw = null;
        state.marquee = null;
      }
      state.pan = null;
      state.connectFrom = null;
      state.styleClipboard = null;
      selectTool("select");
      closeModal();
      render();
    } else if (e.key.toLowerCase() === "v") selectTool("select");
  });
  $("#canvas-scroll").addEventListener("wheel", (e) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    const previous = state.zoom,
      next = Math.max(
        .12,
        Math.min(3, previous * (e.deltaY < 0 ? 1.08 : 1 / 1.08)),
      );
    const bounds = $("#canvas-card").getBoundingClientRect();
    const rx = (e.clientX - bounds.left) / Math.max(1, bounds.width),
      ry = (e.clientY - bounds.top) / Math.max(1, bounds.height);
    state.zoom = next;
    state.fitMode = false;
    setCanvasSize();
    const area = $("#canvas-scroll"),
      after = $("#canvas-card").getBoundingClientRect();
    area.scrollLeft += (after.left - bounds.left) +
      (next - previous) * dimensions().width * rx;
    area.scrollTop += (after.top - bounds.top) +
      (next - previous) * dimensions().height * ry;
  }, { passive: false });
  window.addEventListener("keyup", (e) => {
    if (e.code === "Space") {
      state.spacePan = false;
      $("#diagram").style.cursor = state.tool === "select"
        ? "default"
        : state.tool === "pan"
        ? "grab"
        : "crosshair";
    }
  });
  window.addEventListener("beforeunload", (e) => {
    if (state.dirty) {
      e.preventDefault();
      e.returnValue = "";
    }
  });
  try {
    const draft = JSON.parse(localStorage.getItem(draftKey) || "null");
    if (!hybrid && draft?.document && confirm("이 브라우저에 저장된 자동 복구 문서가 있습니다. 복구할까요?")) {
      beginDocumentReplacement();
      state.doc = validateImportedDoc(draft.document);
      state.file = null;
      state.selected = null;
      state.selectedIds = [];
      state.viewMode = "restored";
      state.savedDoc = null;
      state.history = [clone(state.doc)];
      state.historyIndex = 0;
      dirty(true);
      toast("자동 복구 문서를 열었습니다.");
    }
  } catch (error) {
    try { localStorage.removeItem(draftKey); } catch (_) {}
    console.warn("RADU local draft was invalid and has been removed", error);
  }
  editorTools=RaduTools.install({state,commit,render,toast,status,selectTool,selectElement,createElement,copySelection,pasteSelection,duplicateSelection,deleteSelected,undo,redo,arrangeSelection,syncArrowRelation,fitCanvas,request,saveDocument,validateImportedDoc,beginDocumentReplacement,resetHistory,shapeIcon,pointAt});
  if (hybrid) {
    $("#hybrid-table").onclick = tableDialog;
    $("#hybrid-pdf").onclick = () => exportDocument("pdf");
  }
  render();
  historyButtons();
  const sharedId=hybrid?.jobId || new URLSearchParams(location.search).get('session');
  if(sharedId && /^[a-f0-9]{32}$/.test(sharedId)) loadSession(sharedId);
  setInterval(async()=>{
    if(!state.doc?.session_id || state.dirty || state.busy || document.querySelector('.inline-editor'))return;
    try{
      const fresh=await(await request('/api/sessions/'+state.doc.session_id)).json();
      if((fresh.revision||0)>(state.doc.revision||0)){
        beginDocumentReplacement();state.doc=validateImportedDoc(fresh);state.selected=null;state.selectedIds=[];
        resetHistory();render();status('다른 창에서 저장된 최신 내용을 불러왔습니다.');
      }
    }catch(error){console.warn('Shared document refresh failed',error);}
  },5000);
})();
