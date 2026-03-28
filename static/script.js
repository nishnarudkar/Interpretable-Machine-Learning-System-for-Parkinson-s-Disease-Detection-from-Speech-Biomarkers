// ── Tab switching ──
function openTab(tab, btn) {
  document.querySelectorAll(".tabcontent").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
  document.getElementById(tab).classList.add("active");
  if (btn) btn.classList.add("active");
  if (tab === "importance") loadTopFeatures();
  if (tab === "comparison") loadModelComparison();
  if (tab === "prediction") loadPredictionFields();
}

/** Top-5 feature names and defaults from /feature-config (filled after first fetch). */
let predictionFeatureNames = [];
let predictionDefaults = {};

// ── Top features list ──
async function loadTopFeatures() {
  const list = document.getElementById("top-features-list");
  if (!list || list.dataset.loaded) return;
  try {
    const res  = await fetch("/top-features");
    const data = await res.json();
    list.innerHTML = data.top_features.map(f =>
      `<li>
        <span class="feat-rank">#${f.rank}</span>
        <span class="feat-name">${f.name}</span>
        <span class="feat-score">${f.importance.toFixed(4)}</span>
      </li>`
    ).join("");
    list.dataset.loaded = "1";
  } catch {
    list.innerHTML = "<li>Could not load features.</li>";
  }
}

// ── Prediction inputs (top 5 features) ──
async function loadPredictionFields() {
  const container = document.getElementById("prediction-fields");
  if (!container || container.dataset.loaded === "1") return;

  try {
    const res = await fetch("/feature-config");
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json();
    const names = data.top_features || [];
    const defaults = data.defaults || {};
    predictionFeatureNames = names;
    predictionDefaults = defaults;

    if (names.length === 0) {
      container.innerHTML =
        "<p class=\"prediction-fields-error\">No top features configured. Run training.</p>";
      return;
    }

    container.innerHTML = names
      .map((name, i) => {
        const def = defaults[name];
        const defStr =
          def !== undefined && def !== null && Number.isFinite(Number(def))
            ? Number(def)
            : "";
        const id = `pred-field-${i}`;
        const safeLabel = escapeHtml(name);
        const valAttr =
          defStr === "" ? "" : ` value="${String(defStr)}" placeholder="${String(defStr)}"`;
        return `<div class="prediction-field-row">
  <label class="prediction-field-label" for="${id}">${safeLabel}</label>
  <input type="number" step="any" class="prediction-field-input" id="${id}" data-feature="${i}"${valAttr} />
</div>`;
      })
      .join("");

    container.dataset.loaded = "1";
  } catch (e) {
    console.error(e);
    container.innerHTML =
      "<p class=\"prediction-fields-error\">Could not load feature inputs. Is the API running?</p>";
  }
}

// ── Prediction ──
/**
 * POST /predict sends a JSON object (not an array), e.g.
 * { "feature1": 0.5, "feature2": 1.2, ... }
 * Keys are real column names from the server; values are numbers.
 */
async function predict() {
  const container = document.getElementById("prediction-fields");
  if (!container || container.dataset.loaded !== "1") {
    showError("Feature inputs are not ready yet. Open the Prediction tab and wait for fields to load.");
    return;
  }

  const inputs = container.querySelectorAll(".prediction-field-input");
  /** @type {Record<string, number>} */
  const body = {};

  for (let i = 0; i < predictionFeatureNames.length; i++) {
    const name = predictionFeatureNames[i];
    const el = inputs[i];
    if (!el) continue;
    const raw = el.value.trim();
    if (raw === "") {
      body[name] = Number(predictionDefaults[name]);
      continue;
    }
    const num = Number(raw);
    if (Number.isNaN(num)) {
      showError(`Invalid number for ${name}.`);
      return;
    }
    body[name] = num;
  }

  if (Object.keys(body).length === 0) {
    showError("No features to send.");
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    if (!response.ok) throw new Error("Server error: " + response.status);

    const data = await response.json();
    renderResult(data);
    renderShapChart(data.top_contributions);
    if (data.shap_bar_url) renderShapImage(data.shap_bar_url);
    if (data.top_contributions) renderTopInfluencing(data.top_contributions);
  } catch (err) {
    showError("Prediction failed. Check the server and try again.");
    console.error(err);
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  const btn = document.querySelector(".predict-btn");
  const text = document.getElementById("btn-text");
  const spinner = document.getElementById("btn-spinner");
  btn.disabled = on;
  text.textContent = on ? "Running..." : "Run Prediction";
  spinner.classList.toggle("hidden", !on);
}

function renderResult(data) {
  const isParkinson = data.prediction === 1;
  const prob = data.probability;

  const card = document.getElementById("result-card");
  card.className = "result-card " + (isParkinson ? "parkinson" : "healthy");
  card.classList.remove("hidden");

  document.getElementById("result-icon").textContent = isParkinson ? "⚠️" : "✅";
  document.getElementById("result-label").textContent =
    isParkinson ? "Parkinson's Detected" : "Healthy";

  document.getElementById("prob-bar").style.width = (prob * 100).toFixed(1) + "%";
  document.getElementById("prob-text").textContent =
    "Confidence: " + (prob * 100).toFixed(1) + "%";
}

function showError(msg) {
  const card = document.getElementById("result-card");
  card.className = "result-card parkinson";
  card.classList.remove("hidden");
  document.getElementById("result-icon").textContent = "❌";
  document.getElementById("result-label").textContent = msg;
  document.getElementById("prob-bar").style.width = "0%";
  document.getElementById("prob-text").textContent = "";
}

// ── SHAP server-generated PNG ──
function renderShapImage(url) {
  const wrapper = document.getElementById("shap-img-wrapper");
  const img     = document.getElementById("shap-bar-img");
  img.src = url + "?t=" + Date.now();
  wrapper.classList.remove("hidden");
}

// ── Model comparison (from GET /model-comparison) ──
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function apiErrorDetail(data) {
  if (!data || typeof data !== "object") return null;
  const d = data.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d))
    return d.map(e => (e && e.msg ? e.msg : String(e))).join(" ");
  return null;
}

function num(v, decimals) {
  const n = typeof v === "number" ? v : parseFloat(v);
  return Number.isFinite(n) ? n.toFixed(decimals) : "—";
}

async function loadModelComparison() {
  const tbody = document.getElementById("comparison-tbody");
  const noteEl = document.getElementById("comparison-note");
  if (!tbody) return;

  tbody.innerHTML =
    "<tr><td colspan=\"3\" class=\"table-loading\">Loading model comparison…</td></tr>";
  if (noteEl) {
    noteEl.hidden = true;
    noteEl.textContent = "";
  }

  try {
    const res = await fetch("/model-comparison", { cache: "no-store" });
    let data = {};
    try {
      data = await res.json();
    } catch {
      tbody.innerHTML =
        "<tr><td colspan=\"3\" class=\"table-error\">Invalid response from server.</td></tr>";
      return;
    }

    if (!res.ok) {
      const msg =
        apiErrorDetail(data) ||
        (typeof data === "string" ? data : null) ||
        `Could not load model comparison (HTTP ${res.status}).`;
      tbody.innerHTML =
        `<tr><td colspan="3" class="table-error">${escapeHtml(msg)}</td></tr>`;
      return;
    }

    const models = data.models;
    if (!Array.isArray(models) || models.length === 0) {
      tbody.innerHTML =
        "<tr><td colspan='3' class='table-error'>No models in response.</td></tr>";
      return;
    }

    const rocVals = models.map(m => parseFloat(m.roc_auc));
    const f1Vals = models.map(m => parseFloat(m.macro_f1));
    const maxRoc = Math.max(...rocVals.filter(Number.isFinite));
    const maxF1 = Math.max(...f1Vals.filter(Number.isFinite));

    tbody.innerHTML = models
      .map(m => {
        const roc = parseFloat(m.roc_auc);
        const f1 = parseFloat(m.macro_f1);
        const selected = m.selected === true;
        const bestRoc = Number.isFinite(roc) && roc === maxRoc;
        const bestF1 = Number.isFinite(f1) && f1 === maxF1;
        const rowClass = selected ? "row-selected" : "";
        const star = selected
          ? '<span class="model-star" title="Selected model" aria-hidden="true">★</span>'
          : "";
        const name = escapeHtml(m.model);
        return `
      <tr class="${rowClass}"${selected ? ' data-selected="true"' : ""}>
        <td class="col-model">${star}<span class="model-name">${name}</span></td>
        <td class="col-metric ${bestRoc ? "cell-best-roc" : ""}">${num(m.roc_auc, 3)}</td>
        <td class="col-metric ${bestF1 ? "cell-best-f1" : ""}">${num(m.macro_f1, 3)}</td>
      </tr>`;
      })
      .join("");

    if (noteEl) {
      const sel = models.find(x => x.selected === true);
      noteEl.textContent = sel
        ? `${sel.model} selected for interpretability and stable performance`
        : "Selection follows the best test-set metrics from training.";
      noteEl.hidden = false;
    }
  } catch (err) {
    console.error(err);
    tbody.innerHTML =
      "<tr><td colspan=\"3\" class=\"table-error\">Could not reach the API. Check that the server is running.</td></tr>";
  }
}
function renderTopInfluencing(contributions) {
  const section = document.getElementById("top-influencing");
  const list    = document.getElementById("top-influencing-list");
  const top5    = contributions.slice(0, 5);
  list.innerHTML = top5.map((f, i) =>
    `<li>
      <span class="feat-rank">#${i + 1}</span>
      <span class="feat-name">${f.feature_name || "Feature " + f.feature_index}</span>
      <span class="feat-score" style="color:${f.impact >= 0 ? "#f87171" : "#34d399"}">
        ${f.impact >= 0 ? "▲" : "▼"} ${Math.abs(f.impact).toFixed(4)}
      </span>
    </li>`
  ).join("");
  section.classList.remove("hidden");
}

// ── SHAP Chart ──
function renderShapChart(contributions) {
  const section = document.getElementById("shap-section");
  section.classList.remove("hidden");

  const labels = contributions.map(c => c.feature_name || `Feature ${c.feature_index}`);
  const values = contributions.map(c => c.impact);
  const colors = values.map(v => v >= 0 ? "rgba(248,113,113,0.8)" : "rgba(52,211,153,0.8)");
  const borders = values.map(v => v >= 0 ? "#f87171" : "#34d399");

  const ctx = document.getElementById("shapChart").getContext("2d");

  if (window._shapChart) window._shapChart.destroy();

  window._shapChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "SHAP Impact",
        data: values,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 1.5,
        borderRadius: 5,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` Impact: ${ctx.parsed.x.toFixed(4)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#8892b0" },
          title: { display: true, text: "SHAP Value", color: "#8892b0" }
        },
        y: {
          grid: { display: false },
          ticks: { color: "#8892b0" }
        }
      }
    }
  });
}
