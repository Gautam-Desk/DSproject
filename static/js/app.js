/**
 * VeritasAI - Frontend Client Logic
 * Handles interactive tabs, live TensorFlow predictions, token saliency rendering,
 * benchmark chart drawing, dataset inspection, and share modal with QR code.
 */

document.addEventListener('DOMContentLoaded', () => {
  // State
  let benchmarkData = null;
  let samplesList = [];
  let shareInfo = null;

  // DOM Elements
  const navTabs = document.querySelectorAll('.nav-tab');
  const tabContents = document.querySelectorAll('.tab-content');
  
  // Detector Elements
  const newsForm = document.getElementById('news-form');
  const newsTitleInput = document.getElementById('news-title');
  const newsTextInput = document.getElementById('news-text');
  const bodyCharCount = document.getElementById('body-char-count');
  const sampleSelect = document.getElementById('sample-select');
  const btnClear = document.getElementById('btn-clear');
  const latencyVal = document.getElementById('latency-val');
  const activeModelName = document.getElementById('active-model-name');

  // Result States
  const resultsEmptyState = document.getElementById('results-empty-state');
  const resultsLoadingState = document.getElementById('results-loading-state');
  const resultsActiveState = document.getElementById('results-active-state');

  // Verdict Hero Elements
  const verdictHeroCard = document.getElementById('verdict-hero-card');
  const verdictBadge = document.getElementById('verdict-badge');
  const verdictHeadline = document.getElementById('verdict-headline');
  const verdictDesc = document.getElementById('verdict-desc');
  const gaugeCircle = document.getElementById('gauge-circle');
  const gaugeScore = document.getElementById('gauge-score');
  const gaugeLabel = document.getElementById('gauge-label');

  // Metrics Grid
  const barReal = document.getElementById('bar-real');
  const barFake = document.getElementById('bar-fake');
  const valReal = document.getElementById('val-real');
  const valFake = document.getElementById('val-fake');
  const valRisk = document.getElementById('val-risk');
  const valRiskDesc = document.getElementById('val-risk-desc');
  const valReadability = document.getElementById('val-readability');

  // Saliency & Diagnostics
  const saliencyContainer = document.getElementById('saliency-container');
  const lingSensationalVal = document.getElementById('ling-sensational-val');
  const lingSensationalBar = document.getElementById('ling-sensational-bar');
  const lingCredibleVal = document.getElementById('ling-credible-val');
  const lingCredibleBar = document.getElementById('ling-credible-bar');
  const lingCapVal = document.getElementById('ling-cap-val');
  const lingCapBar = document.getElementById('ling-cap-bar');
  const lingPunctVal = document.getElementById('ling-punct-val');
  const lingPunctBar = document.getElementById('ling-punct-bar');
  const detectedTagsContainer = document.getElementById('detected-tags-container');

  // Lab Elements
  const benchmarkCardsGrid = document.getElementById('benchmark-cards-grid');
  const cmModelSelect = document.getElementById('cm-model-select');
  const cmDisplayContainer = document.getElementById('cm-display-container');
  const rocCanvas = document.getElementById('roc-canvas');

  // Batch Elements
  const batchTextarea = document.getElementById('batch-textarea');
  const btnRunBatch = document.getElementById('btn-run-batch');
  const btnLoadBatchDemo = document.getElementById('btn-load-batch-demo');
  const batchSummaryCard = document.getElementById('batch-summary-card');
  const batchTableContainer = document.getElementById('batch-table-container');
  const batchTableBody = document.getElementById('batch-table-body');
  const batchSumTotal = document.getElementById('batch-sum-total');
  const batchSumReal = document.getElementById('batch-sum-real');
  const batchSumFake = document.getElementById('batch-sum-fake');
  const batchSumLatency = document.getElementById('batch-sum-latency');

  // Python Code Elements
  const btnCopyCode = document.getElementById('btn-copy-code');
  const pythonCodeSnippet = document.getElementById('python-code-snippet');

  // Share Modal Elements
  const btnOpenShare = document.getElementById('btn-open-share');
  const shareModal = document.getElementById('share-modal');
  const btnCloseShare = document.getElementById('btn-close-share');
  const modalQrImg = document.getElementById('modal-qr-img');
  const shareNetworkUrl = document.getElementById('share-network-url');
  const btnCopyUrl = document.getElementById('btn-copy-url');

  // ==========================================
  // 1. Tab Navigation
  // ==========================================
  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-target');
      
      navTabs.forEach(t => t.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      tab.classList.add('active');
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add('active');
      }

      if (targetId === 'tab-lab' && benchmarkData) {
        renderBenchmarkCards();
        renderConfusionMatrix(cmModelSelect.value);
        drawRocCurves();
      }
    });
  });

  // ==========================================
  // 2. Word Count & Form Clear
  // ==========================================
  function updateWordCount() {
    const text = newsTextInput.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    bodyCharCount.textContent = `${words} words`;
  }

  newsTextInput.addEventListener('input', updateWordCount);

  btnClear.addEventListener('click', () => {
    newsTitleInput.value = '';
    newsTextInput.value = '';
    sampleSelect.value = '';
    updateWordCount();
    resultsActiveState.classList.add('hidden');
    resultsLoadingState.classList.add('hidden');
    resultsEmptyState.classList.remove('hidden');
  });

  // ==========================================
  // 3. Samples Loader
  // ==========================================
  async function loadSamples() {
    try {
      const res = await fetch('/api/samples');
      if (res.ok) {
        samplesList = await res.json();
        samplesList.forEach((sample, idx) => {
          const opt = document.createElement('option');
          opt.value = idx;
          opt.textContent = `[${sample.badge}] ${sample.title.slice(0, 50)}...`;
          sampleSelect.appendChild(opt);
        });
      }
    } catch (err) {
      console.error('Failed to load samples:', err);
    }
  }

  sampleSelect.addEventListener('change', (e) => {
    const idx = e.target.value;
    if (idx !== '') {
      const s = samplesList[idx];
      if (s) {
        newsTitleInput.value = s.title;
        newsTextInput.value = s.text;
        updateWordCount();
        runAnalysis();
      }
    }
  });

  // ==========================================
  // 4. Live Prediction & Explainability
  // ==========================================
  async function runAnalysis() {
    const title = newsTitleInput.value.trim();
    const text = newsTextInput.value.trim();

    if (!title || !text) {
      alert('Please provide both headline and body text.');
      return;
    }

    resultsEmptyState.classList.add('hidden');
    resultsActiveState.classList.add('hidden');
    resultsLoadingState.classList.remove('hidden');

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, text })
      });

      if (!res.ok) {
        throw new Error('Inference request failed');
      }

      const data = await res.json();
      renderPredictionResults(data);
    } catch (err) {
      alert('Error analyzing article: ' + err.message);
      resultsLoadingState.classList.add('hidden');
      resultsEmptyState.classList.remove('hidden');
    }
  }

  newsForm.addEventListener('submit', (e) => {
    e.preventDefault();
    runAnalysis();
  });

  function renderPredictionResults(data) {
    resultsLoadingState.classList.add('hidden');
    resultsActiveState.classList.remove('hidden');

    // Update Status Bar
    latencyVal.textContent = `${data.inference_latency_ms} ms`;
    activeModelName.textContent = data.model_architecture || 'BiLSTM with Attention';

    const isFake = data.is_fake;
    const confidence = data.confidence;
    const realProb = data.real_probability;
    const fakeProb = data.fake_probability;

    // Verdict Hero Card Styling
    verdictHeroCard.className = `verdict-hero card ${isFake ? 'state-fake' : 'state-real'}`;
    verdictBadge.className = `verdict-badge ${isFake ? 'fake' : 'real'}`;
    verdictBadge.textContent = isFake ? 'FLAGGED MISINFORMATION' : 'VERIFIED AUTHENTIC';

    if (isFake) {
      verdictHeadline.textContent = 'High Misinformation & Deceptive Risk';
      verdictDesc.textContent = data.risk_description || 'Neural patterns detected sensational clickbait, exaggerated claims, or conspiracy patterns.';
    } else {
      verdictHeadline.textContent = 'Empirical & Fact-Aligned Content';
      verdictDesc.textContent = data.risk_description || 'High consistency with empirical reporting, institutional attribution, and verified facts.';
    }

    // Circular Progress Gauge
    const scoreVal = isFake ? fakeProb : realProb;
    const gaugeColor = isFake ? 'var(--fake-color)' : 'var(--real-color)';
    const angle = (scoreVal / 100) * 360;
    
    gaugeCircle.style.background = `conic-gradient(${gaugeColor} 0deg ${angle}deg, rgba(255,255,255,0.08) ${angle}deg 360deg)`;
    gaugeScore.textContent = `${scoreVal.toFixed(1)}%`;
    gaugeLabel.textContent = isFake ? 'Fake Risk' : 'Authentic';

    // Metrics Bars
    barReal.style.width = `${realProb}%`;
    valReal.textContent = `${realProb.toFixed(1)}%`;
    barFake.style.width = `${fakeProb}%`;
    valFake.textContent = `${fakeProb.toFixed(1)}%`;

    // Risk Assessment Tag
    valRisk.className = `risk-tag ${data.risk_level.toLowerCase()}`;
    valRisk.textContent = data.risk_level;
    valRiskDesc.textContent = isFake ? 'Critical anomaly score' : 'Normal journalistic profile';

    // Readability
    valReadability.textContent = data.linguistics ? data.linguistics.reading_ease : '68.0';

    // Word-Level Saliency Highlighter
    renderSaliencyHighlights(data.saliency_tokens);

    // Linguistic Diagnostics
    if (data.linguistics) {
      const ling = data.linguistics;
      lingSensationalVal.textContent = `${ling.sensationalism_score}%`;
      lingSensationalBar.style.width = `${ling.sensationalism_score}%`;

      lingCredibleVal.textContent = `${ling.credibility_marker_score}%`;
      lingCredibleBar.style.width = `${ling.credibility_marker_score}%`;

      lingCapVal.textContent = `${ling.capitalization_anomaly_score}%`;
      lingCapBar.style.width = `${ling.capitalization_anomaly_score}%`;

      lingPunctVal.textContent = `${ling.punctuation_anomaly_score}%`;
      lingPunctBar.style.width = `${ling.punctuation_anomaly_score}%`;

      // Detected Keywords Tags
      detectedTagsContainer.innerHTML = '';
      if (ling.detected_sensational_terms && ling.detected_sensational_terms.length > 0) {
        ling.detected_sensational_terms.forEach(term => {
          const tag = document.createElement('span');
          tag.className = 'term-tag sensational';
          tag.textContent = `⚠ ${term}`;
          detectedTagsContainer.appendChild(tag);
        });
      }
      if (ling.detected_credible_terms && ling.detected_credible_terms.length > 0) {
        ling.detected_credible_terms.forEach(term => {
          const tag = document.createElement('span');
          tag.className = 'term-tag credible';
          tag.textContent = `✓ ${term}`;
          detectedTagsContainer.appendChild(tag);
        });
      }
    }
  }

  function renderSaliencyHighlights(tokens) {
    saliencyContainer.innerHTML = '';
    if (!tokens || tokens.length === 0) {
      saliencyContainer.textContent = 'No tokens available.';
      return;
    }

    tokens.forEach(item => {
      const span = document.createElement('span');
      span.className = `saliency-token ${item.class}`;
      span.textContent = item.token;

      if (item.class !== 'neutral') {
        const sign = item.weight > 0 ? '+' : '';
        span.title = `Contribution Weight: ${sign}${item.weight}%`;
      }
      saliencyContainer.appendChild(span);
    });
  }

  // ==========================================
  // 5. Benchmark Lab & Charts
  // ==========================================
  async function loadBenchmarks() {
    try {
      const res = await fetch('/api/benchmark');
      if (res.ok) {
        benchmarkData = await res.json();
        renderBenchmarkCards();
        renderConfusionMatrix(cmModelSelect.value);
        drawRocCurves();
      }
    } catch (err) {
      console.error('Failed to load benchmarks:', err);
    }
  }

  function renderBenchmarkCards() {
    if (!benchmarkData || !benchmarkData.models) return;
    benchmarkCardsGrid.innerHTML = '';

    const bestName = benchmarkData.best_model_name;

    Object.entries(benchmarkData.models).forEach(([name, m]) => {
      const isBest = name === bestName;
      const card = document.createElement('div');
      card.className = `card model-card ${isBest ? 'best' : ''}`;
      
      card.innerHTML = `
        ${isBest ? '<span class="best-badge">Best Performer</span>' : ''}
        <div class="model-header">
          <h4 class="model-name">${name}</h4>
          <span class="model-sub">TensorFlow Deep Architecture</span>
        </div>
        <div class="model-stats-row">
          <div class="m-stat">
            <span class="m-label">Accuracy</span>
            <span class="m-val">${(m.accuracy * 100).toFixed(1)}%</span>
          </div>
          <div class="m-stat">
            <span class="m-label">F1-Score</span>
            <span class="m-val">${(m.f1_score * 100).toFixed(1)}%</span>
          </div>
        </div>
        <div class="model-stats-row">
          <div class="m-stat">
            <span class="m-label">Precision</span>
            <span class="m-val">${(m.precision * 100).toFixed(1)}%</span>
          </div>
          <div class="m-stat">
            <span class="m-label">Recall</span>
            <span class="m-val">${(m.recall * 100).toFixed(1)}%</span>
          </div>
        </div>
        <div class="model-footer-info">
          <span>Latency: <strong>${m.latency_ms} ms</strong></span>
          <span>Params: <strong>${(m.parameter_count || 0).toLocaleString()}</strong></span>
        </div>
      `;
      benchmarkCardsGrid.appendChild(card);
    });
  }

  function renderConfusionMatrix(modelName) {
    if (!benchmarkData || !benchmarkData.models) return;
    const model = benchmarkData.models[modelName];
    if (!model || !model.confusion_matrix) return;

    const cm = model.confusion_matrix;
    cmDisplayContainer.innerHTML = `
      <div class="cm-grid-2x2">
        <div class="cm-cell tn">
          <span class="cm-count">${cm.true_negative}</span>
          <span class="cm-label">True Negatives (Real News Correct)</span>
        </div>
        <div class="cm-cell fp">
          <span class="cm-count">${cm.false_positive}</span>
          <span class="cm-label">False Positives (Type I Error)</span>
        </div>
        <div class="cm-cell fn">
          <span class="cm-count">${cm.false_negative}</span>
          <span class="cm-label">False Negatives (Type II Error)</span>
        </div>
        <div class="cm-cell tp">
          <span class="cm-count">${cm.true_positive}</span>
          <span class="cm-label">True Positives (Fake News Correct)</span>
        </div>
      </div>
    `;
  }

  cmModelSelect.addEventListener('change', (e) => {
    renderConfusionMatrix(e.target.value);
  });

  function drawRocCurves() {
    if (!rocCanvas || !benchmarkData || !benchmarkData.models) return;
    const ctx = rocCanvas.getContext('2d');
    const width = rocCanvas.width;
    const height = rocCanvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Padding & dimensions
    const pad = 35;
    const chartW = width - pad * 2;
    const chartH = height - pad * 2;

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;

    for (let i = 0; i <= 4; i++) {
      const y = pad + (chartH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(pad + chartW, y);
      ctx.stroke();

      const x = pad + (chartW / 4) * i;
      ctx.beginPath();
      ctx.moveTo(x, pad);
      ctx.lineTo(x, pad + chartH);
      ctx.stroke();
    }

    // Diagonal Random Baseline
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad, pad + chartH);
    ctx.lineTo(pad + chartW, pad);
    ctx.stroke();
    ctx.setLineDash([]);

    // Colors for models
    const colors = {
      'BiLSTM with Attention': '#38bdf8',
      'CNN-BiLSTM Hybrid': '#10b981',
      'Self-Attention Transformer': '#f59e0b',
      'TF-IDF Baseline': '#a855f7'
    };

    // Plot each model curve
    Object.entries(benchmarkData.models).forEach(([name, m]) => {
      if (!m.roc_curve || m.roc_curve.length === 0) return;
      ctx.strokeStyle = colors[name] || '#38bdf8';
      ctx.lineWidth = 2.5;
      ctx.beginPath();

      m.roc_curve.forEach((pt, idx) => {
        const px = pad + pt.fpr * chartW;
        const py = pad + chartH - pt.tpr * chartH;
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    });

    // Draw Labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText('0.0 (FPR)', pad, height - 12);
    ctx.fillText('1.0', pad + chartW - 15, height - 12);
    ctx.fillText('1.0 (TPR)', 6, pad + 10);
  }

  // ==========================================
  // 6. Dataset Explorer Counters
  // ==========================================
  async function loadDatasetStats() {
    try {
      const res = await fetch('/api/dataset-stats');
      if (res.ok) {
        const stats = await res.json();
        if (document.getElementById('stat-total')) {
          document.getElementById('stat-total').textContent = stats.total_samples || '840';
          document.getElementById('stat-train').textContent = stats.train_samples || '588';
          document.getElementById('stat-val').textContent = stats.val_samples || '126';
          document.getElementById('stat-test').textContent = stats.test_samples || '126';
          document.getElementById('stat-avg-len').textContent = stats.avg_words_per_article ? stats.avg_words_per_article.toFixed(1) : '62.8';
        }
      }
    } catch (err) {
      console.error('Failed to load dataset stats:', err);
    }
  }

  // ==========================================
  // 7. Batch Scanner
  // ==========================================
  const DEMO_BATCH_TEXT = `NASA James Webb Telescope | Discovers ancient galaxy formed shortly after Big Bang with peer-reviewed data.
SHOCKING BOMBSHELL 5G | Secret mind control frequencies are being beamed into residential areas by shadow elites!
Federal Reserve Interest Rates | Policy makers held interest rates steady citing cooling inflation metrics and job growth.
MIRACLE CURE DOCTORS BANNED | Himalayan kitchen spice cures 100 percent of cancers in 48 hours order now!
European Union Energy Report | Combined wind and solar installations generated more electricity than coal power grids.`;

  btnLoadBatchDemo.addEventListener('click', () => {
    batchTextarea.value = DEMO_BATCH_TEXT;
  });

  btnRunBatch.addEventListener('click', async () => {
    const raw = batchTextarea.value.trim();
    if (!raw) {
      alert('Please enter at least one article or click "Load 5 Demo Articles".');
      return;
    }

    const lines = raw.split('\n').filter(l => l.trim().length > 0);
    const articles = lines.map(line => {
      const parts = line.split('|');
      return {
        title: parts[0].trim(),
        text: parts[1] ? parts[1].trim() : parts[0].trim()
      };
    });

    btnRunBatch.textContent = 'Scanning Batch...';
    btnRunBatch.disabled = true;

    try {
      const res = await fetch('/api/batch-predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ articles })
      });

      if (!res.ok) throw new Error('Batch request failed');
      const data = await res.json();

      // Show summary banner
      batchSummaryCard.classList.remove('hidden');
      batchTableContainer.classList.remove('hidden');

      batchSumTotal.textContent = data.summary.total;
      batchSumReal.textContent = data.summary.real_count;
      batchSumFake.textContent = data.summary.fake_count;
      batchSumLatency.textContent = `${data.summary.batch_latency_ms} ms (${data.summary.avg_ms_per_article} ms/item)`;

      // Populate Table
      batchTableBody.innerHTML = '';
      data.results.forEach((item, idx) => {
        const tr = document.createElement('tr');
        const isFake = item.verdict === 'FAKE';
        tr.innerHTML = `
          <td>${idx + 1}</td>
          <td><strong>${item.title.slice(0, 70)}...</strong></td>
          <td><span class="verdict-badge ${isFake ? 'fake' : 'real'}">${item.verdict}</span></td>
          <td><strong style="color: var(--real-color); font-family: var(--font-mono);">${item.real_probability}%</strong></td>
          <td><strong style="color: var(--fake-color); font-family: var(--font-mono);">${item.fake_probability}%</strong></td>
          <td><strong style="font-family: var(--font-mono);">${item.confidence}%</strong></td>
        `;
        batchTableBody.appendChild(tr);
      });

    } catch (err) {
      alert('Batch scan error: ' + err.message);
    } finally {
      btnRunBatch.textContent = 'Scan All Articles';
      btnRunBatch.disabled = false;
    }
  });

  // ==========================================
  // 8. Python Code Copy
  // ==========================================
  btnCopyCode.addEventListener('click', () => {
    const code = pythonCodeSnippet.textContent;
    navigator.clipboard.writeText(code).then(() => {
      btnCopyCode.textContent = 'Copied!';
      setTimeout(() => { btnCopyCode.textContent = 'Copy Python Code'; }, 2000);
    });
  });

  // ==========================================
  // 9. Share & QR Modal
  // ==========================================
  async function loadShareInfo() {
    try {
      const res = await fetch('/api/share-info');
      if (res.ok) {
        shareInfo = await res.json();
        modalQrImg.src = shareInfo.qr_code_base64;
        shareNetworkUrl.value = shareInfo.network_url;
      }
    } catch (err) {
      console.error('Failed to load share info:', err);
    }
  }

  btnOpenShare.addEventListener('click', () => {
    loadShareInfo();
    shareModal.showModal();
  });

  btnCloseShare.addEventListener('click', () => {
    shareModal.close();
  });

  btnCopyUrl.addEventListener('click', () => {
    navigator.clipboard.writeText(shareNetworkUrl.value).then(() => {
      btnCopyUrl.textContent = 'Copied!';
      setTimeout(() => { btnCopyUrl.textContent = 'Copy Link'; }, 2000);
    });
  });

  // Initialize Data
  loadSamples();
  loadBenchmarks();
  loadDatasetStats();
  loadShareInfo();
});
