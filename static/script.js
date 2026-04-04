// ── Tab switching ──
function openTab(tab, btn) {
  document.querySelectorAll(".tabcontent").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
  document.getElementById(tab).classList.add("active");
  if (btn) btn.classList.add("active");
  if (tab === "importance") loadTopFeatures();
  if (tab === "comparison") loadModelComparison();
  if (tab === "prediction") loadPredictionFields();
  if (tab === "insights")   loadInsights();
  if (tab === "drift")      loadDriftStatus();
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

// ── Prediction inputs (top 5 features as grid cards) ──
let _allColumns  = [];
let _allMedians  = {};

async function loadPredictionFields() {
  const container = document.getElementById("prediction-fields");
  if (!container || container.dataset.loaded === "1") return;

  try {
    const res = await fetch("/feature-defaults");
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json();

    _allColumns = data.columns || [];
    _allMedians = data.medians || {};
    const top5  = data.top5   || [];

    if (top5.length === 0) {
      container.innerHTML = "<p class='prediction-fields-loading'>No features found. Run training first.</p>";
      return;
    }

    container.innerHTML = top5.map((f, i) => `
      <div class="feat-card">
        <div class="feat-card-label">
          <span>${escapeHtml(f.label)}</span>
          ${f.tooltip ? `<span class="feat-tooltip-icon" title="${escapeHtml(f.tooltip)}">ℹ</span>` : ""}
        </div>
        <input
          type="number" step="any"
          class="feat-card-input"
          id="feat-${i}"
          data-name="${escapeHtml(f.name)}"
          value="${f.median}"
          placeholder="${f.median}"
        />
        ${(f.min !== undefined && f.max !== undefined)
          ? `<span class="feat-range">Range: ${f.min} – ${f.max}</span>`
          : ""}
      </div>`
    ).join("");

    container.dataset.loaded = "1";
  } catch (e) {
    console.error(e);
    container.innerHTML = "<p class='prediction-fields-loading'>Could not load feature inputs.</p>";
  }
}

// ── Prediction ──
async function predict() {
  const container = document.getElementById("prediction-fields");
  if (!container || container.dataset.loaded !== "1") {
    showError("Feature inputs are not ready yet.");
    return;
  }

  // Start with all 753 median defaults
  const featureMap = { ..._allMedians };

  // Override with user-edited values
  const inputs = container.querySelectorAll(".feat-card-input");
  for (const input of inputs) {
    const name = input.dataset.name;
    const raw  = input.value.trim();
    if (raw === "") continue;
    const num = Number(raw);
    if (isNaN(num)) {
      showError(`Invalid number for "${name}". Please enter a numeric value.`);
      return;
    }
    featureMap[name] = num;
  }

  // Build ordered array matching column_order
  if (_allColumns.length === 0) {
    showError("Column order not loaded. Please refresh the page.");
    return;
  }
  const features = _allColumns.map(col => featureMap[col] ?? 0);

  setLoading(true);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
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

// ── Feature Insights ──
async function loadInsights() {
  const grid = document.getElementById("insights-grid");
  if (!grid || grid.dataset.loaded) return;
  try {
    const res  = await fetch("/static/feature_insights.json");
    const data = await res.json();

    grid.innerHTML = data.map(f => {
      const higher = f.trend === "Higher in Parkinson";
      const trendClass = higher ? "trend-high" : "trend-low";
      const trendIcon  = higher ? "▲" : "▼";
      const diff = Math.abs(f.parkinson - f.healthy).toFixed(4);

      return `
        <div class="insight-card">
          <div class="insight-header">
            <span class="insight-name">${escapeHtml(f.label || f.feature)}</span>
            <span class="insight-trend ${trendClass}">${trendIcon} ${escapeHtml(f.trend)}</span>
          </div>
          <div class="insight-feature-id">${escapeHtml(f.feature)}</div>
          <div class="insight-means">
            <div class="insight-mean parkinson-mean">
              <span class="mean-label">Parkinson's</span>
              <span class="mean-value">${f.parkinson.toFixed(4)}</span>
            </div>
            <div class="insight-diff">Δ ${diff}</div>
            <div class="insight-mean healthy-mean">
              <span class="mean-label">Healthy</span>
              <span class="mean-value">${f.healthy.toFixed(4)}</span>
            </div>
          </div>
          <p class="insight-reason">${escapeHtml(f.reason)}</p>
        </div>`;
    }).join("");

    grid.dataset.loaded = "1";
  } catch (e) {
    grid.innerHTML = "<p class='insights-loading'>Could not load feature insights.</p>";
    console.error(e);
  }
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


// ── Drift Monitor ──────────────────────────────────────────────────────────

let _driftFeatures = [];

async function loadDriftStatus() {
  const banner   = document.getElementById("drift-summary-banner");
  if (banner.dataset.loaded === "1") return;

  // Reset error state on each attempt
  const errorEl = document.getElementById("drift-error");
  errorEl.classList.add("hidden");
  banner.classList.remove("hidden");

  try {
    const res  = await fetch("/drift-status", { cache: "no-store" });
    const data = await res.json();

    if (!res.ok) {
      showDriftError(data.detail || "Could not load drift data.");
      return;  // do NOT set loaded=1 so user can retry
    }

    const s = data.summary;
    _driftFeatures = data.features || [];

    // Banner state
    const drifted = s.drifted_count ?? 0;
    const total   = s.total_features ?? 0;
    const pct     = s.drift_pct ?? 0;
    const isDrift = pct > 50;

    banner.className = "drift-banner";
    banner.style.borderColor  = isDrift ? "rgba(248,113,113,0.45)" : "rgba(52,211,153,0.45)";
    banner.style.background   = isDrift ? "rgba(248,113,113,0.06)" : "rgba(52,211,153,0.06)";
    document.getElementById("drift-status-icon").textContent  = isDrift ? "⚠️" : "✅";
    document.getElementById("drift-status-label").textContent = s.status || (isDrift ? "Drift Detected" : "No Significant Drift");
    document.getElementById("drift-generated-at").textContent = s.generated_at ? "Last checked: " + s.generated_at : "";
    document.getElementById("drift-pct-value").textContent    = pct.toFixed(1) + "%";
    document.getElementById("drift-gauge-text").textContent   = drifted + " / " + total;

    const bar = document.getElementById("drift-gauge-bar");
    bar.style.width      = Math.min(pct, 100) + "%";
    bar.style.background = isDrift ? "var(--danger-color)" : "var(--success-color)";

    if (s.note) {
      const noteEl = document.getElementById("drift-note");
      noteEl.textContent = "ℹ " + s.note;
      noteEl.classList.remove("hidden");
    }

    // Chart — top 15 drifted features by lowest p-value
    const driftedFeats = _driftFeatures.filter(f => f.drifted).slice(0, 15);
    if (driftedFeats.length > 0) {
      document.getElementById("drift-chart-section").classList.remove("hidden");
      renderDriftChart(driftedFeats);
    }

    // Table
    document.getElementById("drift-table-section").classList.remove("hidden");
    renderDriftTable(_driftFeatures, "all");

    banner.dataset.loaded = "1";
  } catch (err) {
    showDriftError("Could not reach the API. Check that the server is running.");
    console.error(err);
  }
}

function renderDriftChart(features) {
  const labels = features.map(f => f.feature);
  const values = features.map(f => f.p_value);
  const colors = features.map(f => f.p_value < 0.01 ? "rgba(248,113,113,0.8)" : "rgba(251,191,36,0.8)");

  const ctx = document.getElementById("driftChart").getContext("2d");
  if (window._driftChart) window._driftChart.destroy();

  window._driftChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "p-value",
        data: values,
        backgroundColor: colors,
        borderColor: colors.map(c => c.replace("0.8", "1")),
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => " p = " + ctx.parsed.x.toExponential(2)
          }
        }
      },
      scales: {
        x: {
          min: 0,
          max: 0.05,
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#8892b0", callback: v => v.toExponential(1) },
          title: { display: true, text: "p-value (KS test) — lower = more drifted", color: "#8892b0" }
        },
        y: {
          grid: { display: false },
          ticks: { color: "#8892b0", font: { family: "'Courier New', monospace", size: 11 } }
        }
      }
    }
  });
}

function renderDriftTable(features, filter) {
  const tbody = document.getElementById("drift-table-body");
  const rows  = filter === "drifted" ? features.filter(f => f.drifted)
              : filter === "stable"  ? features.filter(f => !f.drifted)
              : features;

  tbody.innerHTML = rows.map(f => {
    const rowClass  = f.drifted ? "dt-row-drifted" : "";
    const badge     = f.drifted
      ? '<span class="drift-badge drift-badge--drifted">Drifted</span>'
      : '<span class="drift-badge drift-badge--stable">Stable</span>';
    const pDisplay  = f.p_value === 0 ? "< 1e-6" : f.p_value.toExponential(2);
    return `<tr class="${rowClass}">
      <td class="dt-col-feature"><span class="dt-feat-name">${escapeHtml(f.feature)}</span></td>
      <td class="dt-col-pval">${pDisplay}</td>
      <td class="dt-col-status">${badge}</td>
    </tr>`;
  }).join("");
}

function filterDriftTable(filter, btn) {
  document.querySelectorAll(".drift-filter-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  renderDriftTable(_driftFeatures, filter);
}

function showDriftError(msg) {
  const el = document.getElementById("drift-error");
  el.textContent = msg;
  el.classList.remove("hidden");
  document.getElementById("drift-summary-banner").classList.add("hidden");
}


