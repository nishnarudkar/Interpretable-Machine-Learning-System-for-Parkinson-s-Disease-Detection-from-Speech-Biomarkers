// ── Tab switching ──
function openTab(tab, btn) {
  document.querySelectorAll(".tabcontent").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
  document.getElementById(tab).classList.add("active");
  if (btn) btn.classList.add("active");
  if (tab === "importance") loadTopFeatures();
}

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

// ── Prediction ──
async function predict() {
  const input = document.getElementById("inputFeatures").value.trim();
  if (!input) return;

  const features = input.split(",").map(Number);
  if (features.some(isNaN)) {
    showError("Please enter valid comma-separated numbers.");
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features })
    });

    if (!response.ok) throw new Error("Server error: " + response.status);

    const data = await response.json();
    renderResult(data);
    renderShapChart(data.top_contributions);
    if (data.shap_bar_url) renderShapImage(data.shap_bar_url);
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
  // Append cache-buster so the browser reloads the image on each prediction
  img.src = url + "?t=" + Date.now();
  wrapper.classList.remove("hidden");
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
