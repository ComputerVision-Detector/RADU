// Load an image from the prepared datasets in the project folder as the source image.
(() => {
  "use strict";
  const $ = (selector) => document.querySelector(selector);
  const source = $("#local-source"), split = $("#local-split"), sample = $("#local-sample"),
    loadButton = $("#local-load"), summary = $("#local-summary");
  if (!source) return;
  const kinds = { diagram: "도식", formula: "수식", table: "표" };
  const say = (message, error = false) => {
    summary.textContent = message;
    summary.classList.toggle("labeled-error", error);
  };
  async function api(url) {
    const response = await fetch(url);
    if (!response.ok) {
      let detail = response.statusText;
      try { detail = (await response.json()).detail || detail; } catch (e) { /* not JSON */ }
      throw new Error(detail);
    }
    return response;
  }
  const labelText = (labels) => Object.entries(labels).map(([key, count]) => key + " " + count).join(", ");

  async function listSources() {
    try {
      const data = await (await api("/api/local-datasets")).json();
      for (const kind of Object.keys(kinds)) {
        const group = document.createElement("optgroup");
        group.label = kinds[kind];
        for (const item of data.sources.filter((entry) => entry.kind === kind)) {
          const option = document.createElement("option");
          option.value = item.id;
          option.textContent = item.id + " · validation " + item.splits.validation + " / test " + item.splits.test;
          group.append(option);
        }
        source.append(group);
      }
      say(data.sources.length + "개 데이터셋 · " + data.root);
    } catch (error) {
      say("프로젝트 데이터셋을 불러오지 못했습니다: " + error.message, true);
    }
  }

  async function listSamples() {
    sample.innerHTML = '<option value="">이미지 선택…</option>';
    loadButton.disabled = true;
    if (!source.value) return;
    try {
      const items = await (await api("/api/local-datasets/" + encodeURIComponent(source.value) + "/" + split.value)).json();
      for (const item of items.samples) {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = item.name + " · 정답 " + labelText(item.labels);
        sample.append(option);
      }
      say(source.value + " " + split.value + " · " + items.samples.length + "장");
    } catch (error) {
      say("목록을 불러오지 못했습니다: " + error.message, true);
    }
  }

  async function loadSample() {
    if (!sample.value) return;
    loadButton.disabled = true;
    const name = sample.selectedOptions[0].textContent.split(" · ")[0];
    say(name + " 이미지를 불러오는 중…");
    try {
      const blob = await (await api("/api/local-datasets/image?id=" + encodeURIComponent(sample.value))).blob();
      const transfer = new DataTransfer();
      transfer.items.add(new File([blob], name, { type: blob.type || "application/octet-stream" }));
      const input = $("#image-file");
      input.files = transfer.files;
      input.dispatchEvent(new Event("change", { bubbles: true }));
      const kind = source.selectedOptions[0].parentElement.label;
      say(name + " 준비됨 (" + kind + " 데이터). '이미지 분석'을 누르세요.");
    } catch (error) {
      say("이미지를 불러오지 못했습니다: " + error.message, true);
    } finally {
      loadButton.disabled = !sample.value;
    }
  }

  source.onchange = listSamples;
  split.onchange = listSamples;
  sample.onchange = () => { loadButton.disabled = !sample.value; };
  loadButton.onclick = loadSample;
  listSources();
})();
