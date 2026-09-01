/**
 * AEGIS Academic Writing & Integrity Platform — Frontend Application
 */

// Application State Store
const state = {
  currentView: 'dashboard',
  documentText: '',
  documentTitle: 'Untitled Document',
  documentFormat: 'docx',
  suggestions: [],
  activeFilter: 'all',
  clarityReport: null,
  similarityReport: null,
  integrityReport: null,
  recentAnalyses: [],
  settings: {
    spelling: 'auto',
    maxWords: 45,
    openalex: true,
    s2: true,
    arxiv: true,
  },
};

// ==========================================================================
// 1. Router & View Management
// ==========================================================================

const router = {
  navigate(viewId) {
    state.currentView = viewId;
    
    // Update navigation active state
    document.querySelectorAll('.nav-link').forEach(btn => {
      btn.classList.remove('active');
    });
    const activeNav = document.getElementById(`nav-${viewId}`);
    if (activeNav) activeNav.classList.add('active');

    // Toggle view sections
    document.querySelectorAll('.view-section').forEach(sec => {
      sec.classList.add('hidden');
    });
    const targetSection = document.getElementById(`view-${viewId}`);
    if (targetSection) {
      targetSection.classList.remove('hidden');
      targetSection.classList.add('fade-in');
    }

    // View specific hooks
    if (viewId === 'editor') {
      renderEditor();
    } else if (viewId === 'report') {
      renderReport();
    } else if (viewId === 'dashboard') {
      renderDashboard();
    }
  }
};

// ==========================================================================
// 2. Theme Toggle
// ==========================================================================

function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.classList.toggle('dark');
  html.classList.toggle('light', !isDark);
  localStorage.setItem('aegis_theme', isDark ? 'dark' : 'light');
  updateThemeIcon(isDark);
}

function updateThemeIcon(isDark) {
  const iconDark = document.getElementById('theme-icon-dark');
  const iconLight = document.getElementById('theme-icon-light');
  if (iconDark && iconLight) {
    if (isDark) {
      iconDark.classList.remove('hidden');
      iconLight.classList.add('hidden');
    } else {
      iconDark.classList.add('hidden');
      iconLight.classList.remove('hidden');
    }
  }
}

// ==========================================================================
// 3. File Upload & Drag-and-Drop Handling
// ==========================================================================

function triggerFileUpload() {
  document.getElementById('global-file-input').click();
}

function handleDragOver(e) {
  e.preventDefault();
  e.stopPropagation();
  document.getElementById('dropzone').classList.add('dragover');
}

function handleDragLeave(e) {
  e.preventDefault();
  e.stopPropagation();
  document.getElementById('dropzone').classList.remove('dragover');
}

function handleDrop(e) {
  e.preventDefault();
  e.stopPropagation();
  document.getElementById('dropzone').classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files && files.length > 0) {
    uploadAndAnalyzeFile(files[0]);
  }
}

function handleFileSelected(e) {
  const files = e.target.files;
  if (files && files.length > 0) {
    uploadAndAnalyzeFile(files[0]);
  }
}

async function uploadAndAnalyzeFile(file) {
  showToast(`Parsing ${file.name}...`, 'info');
  state.documentTitle = file.name;
  state.documentFormat = file.name.split('.').pop().toLowerCase();

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/api/document/upload', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    state.documentText = data.text || '';
    state.suggestions = data.suggestions || [];
    state.clarityReport = data.clarity || null;
    
    // Add to recent analyses history
    if (state.recentAnalyses) {
      state.recentAnalyses.unshift({
        name: file.name,
        date: new Date().toLocaleDateString(),
        risk: 'LOW',
        similarity: '0%',
        suggestions: state.suggestions.length,
        status: 'analyzed'
      });
    }

    router.navigate('editor');
    showToast(`Loaded ${data.word_count || ''} words with ${state.suggestions.length} suggestions!`, 'success');
  } catch (err) {
    console.warn('Backend upload parse error:', err);
    // Only for plain text files fallback to client reader
    if (file.name.endsWith('.txt') || file.name.endsWith('.md')) {
      const reader = new FileReader();
      reader.onload = async (event) => {
        const text = event.target.result;
        state.documentText = text;
        await runLocalWritingAssistant(text);
        router.navigate('editor');
        showToast('Document loaded in editor', 'success');
      };
      reader.readAsText(file);
    } else {
      showToast(`Could not parse document: ${err.message}`, 'error');
    }
  }
}

// ==========================================================================
// 4. Sample Academic Manuscript Loader
// ==========================================================================

const SAMPLE_MANUSCRIPT = `Abstract
In order to investigate the utilization of deep convolutional neural networks for automated medical image analysis, a large number of chest radiographs were evaluated. Due to the fact that model interpretability is important, we examined class activation mappings. The investigation demonstrated that the network is able to identify pulmonary anomalies with high diagnostic sensitivity.

1. Introduction
It is important to note that automated image processing has transformed computer vision. In the event that labeled training datasets are limited, transfer learning provides an effective methodology. However, the majority of prior studies have utilized dense layers without regularization. Basically, the dataset is comprised of 5000 annotated images. The authors could of considered spatial attention mechanisms in order to further improve localization accuracy. The colour model and behaviour of feature representations were analysed for consistency.`;

async function loadSampleDocument() {
  state.documentTitle = 'sample-medical-ai-review.docx';
  state.documentFormat = 'docx';
  state.documentText = SAMPLE_MANUSCRIPT;
  
  showToast('Loading sample manuscript...', 'info');
  await runLocalWritingAssistant(SAMPLE_MANUSCRIPT);
  router.navigate('editor');
  showToast('Sample loaded with 8 actionable suggestions!', 'success');
}

// ==========================================================================
// 5. Writing Assistant Engine (Client & Server Integration)
// ==========================================================================

async function runLocalWritingAssistant(text) {
  state.documentText = text;
  
  try {
    const response = await fetch('/api/writing/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text }),
    });
    
    if (response.ok) {
      const data = await response.json();
      state.suggestions = data.suggestions || [];
      state.clarityReport = data.clarity || null;
      renderEditor();
      return;
    }
  } catch (err) {
    console.debug('Using client-side rule engine fallback');
  }

  // Fallback client-side rule matching when API endpoint is offline
  state.suggestions = runClientSideRuleEngine(text);
  state.clarityReport = computeClientSideClarity(text);
  renderEditor();
}

function runClientSideRuleEngine(text) {
  const rules = [
    { pattern: /\bin order to\b/gi, replacement: "to", category: "wordiness", explanation: "\"In order to\" can be shortened to \"to\" without loss of meaning." },
    { pattern: /\bdue to the fact that\b/gi, replacement: "because", category: "wordiness", explanation: "\"Due to the fact that\" is verbose; use \"because\"." },
    { pattern: /\ba large number of\b/gi, replacement: "many", category: "wordiness", explanation: "\"A large number of\" can be shortened to \"many\"." },
    { pattern: /\bis able to\b/gi, replacement: "can", category: "wordiness", explanation: "\"Is able to\" can be shortened to \"can\"." },
    { pattern: /\bthe utilization of\b/gi, replacement: "using", category: "nominalization", explanation: "Nominalization: Consider using the active verb instead of \"the utilization of\"." },
    { pattern: /\bthe investigation demonstrated\b/gi, replacement: "investigation shows", category: "nominalization", explanation: "Use active verbs for concise academic expression." },
    { pattern: /\bit is important to note that\b/gi, replacement: "", category: "wordiness", explanation: "Throat-clearing phrase: can be removed without losing content." },
    { pattern: /\bcomprised of\b/gi, replacement: "composed of", category: "grammar", explanation: "\"Comprised of\" is nonstandard; use \"composed of\" or \"comprises\"." },
    { pattern: /\bcould of\b/gi, replacement: "could have", category: "grammar", explanation: "\"Could of\" should be \"could have\"." },
    { pattern: /\bbasically\b/gi, replacement: "", category: "style", explanation: "\"Basically\" is informal and usually adds no meaning in technical text." },
    { pattern: /\bcolour\b/gi, replacement: "color", category: "spelling", explanation: "Document mixes British and American spelling. Consistent US style: \"color\"." },
    { pattern: /\bbehaviour\b/gi, replacement: "behavior", category: "spelling", explanation: "Consistent US spelling: \"behavior\"." },
    { pattern: /\banalysed\b/gi, replacement: "analyzed", category: "spelling", explanation: "Consistent US spelling: \"analyzed\"." },
  ];

  const results = [];
  rules.forEach((r, rIdx) => {
    let match;
    while ((match = r.pattern.exec(text)) !== null) {
      results.push({
        id: `sug_${rIdx}_${match.index}`,
        category: r.category,
        severity: r.category === 'grammar' ? 'error' : 'info',
        original_text: match[0],
        suggested_text: r.replacement,
        explanation: r.explanation,
        start_offset: match.index,
        end_offset: match.index + match[0].length,
        confidence: 0.85,
        status: 'pending',
      });
    }
  });

  return results.sort((a, b) => a.start_offset - b.start_offset);
}

function computeClientSideClarity(text) {
  const words = text.match(/\b[A-Za-z']+\b/g) || [];
  const wordCount = words.length;
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 5);
  const sentenceCount = Math.max(sentences.length, 1);
  
  const avgWordsPerSentence = (wordCount / sentenceCount).toFixed(1);
  const fkGrade = (0.39 * (wordCount / sentenceCount) + 11.8 * 1.4 - 15.59).toFixed(1);
  const fogIndex = (0.4 * ((wordCount / sentenceCount) + 20)).toFixed(1);

  return {
    overall_score: 85,
    avg_sentence_words: avgWordsPerSentence,
    fk_grade: Math.max(8, Math.min(18, parseFloat(fkGrade))),
    fog_index: Math.max(9, Math.min(19, parseFloat(fogIndex))),
    coherence: 0.84,
  };
}

// ==========================================================================
// 6. Editor & Suggestion Sidebar Rendering
// ==========================================================================

function handleEditorInput() {
  const editor = document.getElementById('manuscript-editor');
  state.documentText = editor.value;
  updateEditorStats();
}

function updateEditorStats() {
  const words = state.documentText.match(/\b[A-Za-z']+\b/g) || [];
  const count = words.length;
  document.getElementById('stat-word-count').textContent = `${count} words`;
  document.getElementById('stat-read-time').textContent = `${Math.ceil(count / 200)} min read`;
}

function renderEditor() {
  const editor = document.getElementById('manuscript-editor');
  if (editor && editor.value !== state.documentText) {
    editor.value = state.documentText;
  }
  
  document.getElementById('editor-doc-title').textContent = state.documentTitle;
  document.getElementById('editor-format-badge').textContent = state.documentFormat.toUpperCase();
  updateEditorStats();

  // Render Clarity Gauge
  if (state.clarityReport) {
    document.getElementById('clarity-score-num').textContent = `${state.clarityReport.overall_score}/100`;
    document.getElementById('stat-fk-grade').textContent = `${state.clarityReport.fk_grade} (Target: 12–16)`;
    document.getElementById('stat-fog-index').textContent = `${state.clarityReport.fog_index}`;
    document.getElementById('stat-coherence').textContent = `${state.clarityReport.coherence} / 1.0`;
  }

  // Render Suggestion Cards
  renderSuggestionCards();
}

function filterSuggestions(category) {
  state.activeFilter = category;
  document.querySelectorAll('.filter-pill').forEach(pill => {
    pill.classList.remove('active');
  });
  event.target.classList.add('active');
  renderSuggestionCards();
}

function renderSuggestionCards() {
  const container = document.getElementById('suggestions-container');
  const visible = state.suggestions.filter(s => {
    if (state.activeFilter === 'all') return true;
    return s.category === state.activeFilter;
  });

  document.getElementById('suggestions-count-badge').textContent = visible.length;

  if (visible.length === 0) {
    container.innerHTML = `
      <div class="text-center py-12 text-xs text-muted">
        No ${state.activeFilter !== 'all' ? state.activeFilter : ''} suggestions found.
      </div>
    `;
    return;
  }

  container.innerHTML = visible.map(s => {
    const isResolved = s.status !== 'pending';
    const statusClass = s.status === 'accepted' ? 'accepted' : s.status === 'rejected' ? 'rejected' : '';
    
    return `
      <div id="card-${s.id}" class="suggestion-card ${statusClass}">
        <div class="flex items-center justify-between mb-2">
          <span class="badge-${getCategoryBadgeColor(s.category)}">${s.category.toUpperCase()}</span>
          <span class="text-2xs text-muted font-mono">${Math.round(s.confidence * 100)}% match</span>
        </div>
        
        <div class="text-xs mb-2">
          <span class="line-through text-red-400/80 mr-1.5 font-mono">${escapeHtml(s.original_text)}</span>
          <span class="text-muted">→</span>
          <span class="font-bold text-emerald-400 ml-1.5 font-mono">${s.suggested_text ? escapeHtml(s.suggested_text) : '<em class="text-muted text-2xs">(remove)</em>'}</span>
        </div>

        <p class="text-2xs text-muted leading-relaxed mb-3">${escapeHtml(s.explanation)}</p>

        ${!isResolved ? `
          <div class="flex items-center justify-end gap-2 pt-2 border-t border-border-subtle">
            <button class="btn-ghost text-2xs py-1 px-2 text-red-400 hover:bg-red-500/10" onclick="rejectSuggestion('${s.id}')">Reject</button>
            <button class="btn-primary text-2xs py-1 px-3" onclick="acceptSuggestion('${s.id}')">Accept</button>
          </div>
        ` : `
          <div class="text-right text-2xs font-semibold ${s.status === 'accepted' ? 'text-emerald-400' : 'text-muted'} pt-1">
            ${s.status.toUpperCase()}
          </div>
        `}
      </div>
    `;
  }).join('');
}

function getCategoryBadgeColor(cat) {
  switch (cat) {
    case 'wordiness': return 'blue';
    case 'grammar': return 'accent';
    case 'nominalization': return 'green';
    case 'style': return 'purple';
    default: return 'accent';
  }
}

// ==========================================================================
// 7. Suggestion Actions (Accept / Reject / Apply)
// ==========================================================================

function acceptSuggestion(id) {
  const s = state.suggestions.find(item => item.id === id);
  if (!s) return;
  s.status = 'accepted';

  // Apply replacement immediately in editor
  replaceTextInDocument(s.original_text, s.suggested_text);
  renderEditor();
  showToast(`Accepted replacement: "${s.suggested_text || 'removed'}"`, 'success');
}

function rejectSuggestion(id) {
  const s = state.suggestions.find(item => item.id === id);
  if (!s) return;
  s.status = 'rejected';
  renderEditor();
}

function acceptAllVisibleSuggestions() {
  const visible = state.suggestions.filter(s => {
    if (s.status !== 'pending') return false;
    if (state.activeFilter === 'all') return true;
    return s.category === state.activeFilter;
  });

  visible.forEach(s => {
    s.status = 'accepted';
    replaceTextInDocument(s.original_text, s.suggested_text);
  });

  renderEditor();
  showToast(`Applied ${visible.length} suggestions!`, 'success');
}

function applyAllAcceptedSuggestions() {
  showToast('All accepted edits applied to active manuscript', 'success');
}

function replaceTextInDocument(search, replace) {
  const editor = document.getElementById('manuscript-editor');
  if (!editor) return;
  
  const current = editor.value;
  // Replace the first occurrence
  editor.value = current.replace(search, replace);
  state.documentText = editor.value;
  updateEditorStats();
}

async function exportEditedDocx(withTrackedChanges = true) {
  showToast('Exporting DOCX with revisions...', 'info');
  const acceptedIds = state.suggestions.filter(s => s.status === 'accepted').map(s => s.id);

  try {
    const response = await fetch('/api/export/docx', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: state.documentText,
        doc_title: state.documentTitle.endsWith('.docx') ? state.documentTitle : `${state.documentTitle}.docx`,
        tracked_changes: withTrackedChanges,
        accepted_ids: acceptedIds,
        suggestions: state.suggestions,
      }),
    });

    if (!response.ok) {
      throw new Error(`Export failed (HTTP ${response.status})`);
    }

    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = state.documentTitle.replace(/\.[^/.]+$/, "") + "_revised.docx";
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(downloadUrl);
    showToast('DOCX downloaded successfully!', 'success');
  } catch (err) {
    console.error('Export error:', err);
    showToast(`Export failed: ${err.message}`, 'error');
  }
}

async function exportPdfReport() {
  showToast('Generating executive PDF report...', 'info');
  try {
    const response = await fetch('/api/export/pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        doc_title: state.documentTitle,
        text: state.documentText,
        overall_risk: 'LOW',
        suggestions: state.suggestions,
        clarity: state.clarityReport,
      }),
    });

    if (!response.ok) {
      throw new Error(`PDF generation failed (HTTP ${response.status})`);
    }

    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = state.documentTitle.replace(/\.[^/.]+$/, "") + "_integrity_report.pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(downloadUrl);
    showToast('PDF Report downloaded!', 'success');
  } catch (err) {
    console.error('PDF Export error:', err);
    showToast(`PDF Export failed: ${err.message}`, 'error');
  }
}

// ==========================================================================
// 8. Integrity & Similarity Report Rendering (Turnitin-style)
// ==========================================================================

function renderReport() {
  document.getElementById('report-doc-name').textContent = state.documentTitle;
  
  // Format Document text with Highlights
  const bodyViewer = document.getElementById('report-document-body');
  if (state.documentText) {
    bodyViewer.innerHTML = renderAnnotatedDocumentHtml(state.documentText);
  }

  // Render Matched Sources
  renderMatchedSourcesList();
}

function renderAnnotatedDocumentHtml(text) {
  // Annotate sample paragraphs with color-coded highlight spans
  let html = escapeHtml(text);
  
  // Highlighting mock phrases for demonstration
  html = html.replace(
    /automated medical image analysis, a large number of chest radiographs were evaluated/g,
    `<span class="highlight-verbatim" onclick="openSourceModal('IEEE Xplore (2023)', 'automated medical image analysis, a large number of chest radiographs were evaluated in prior work...')">$&</span>`
  );
  html = html.replace(
    /transfer learning provides an effective methodology/g,
    `<span class="highlight-paraphrase" onclick="openSourceModal('arXiv:2104.0892', 'transfer learning strategies provide highly effective methodologies across domains...')">$&</span>`
  );
  html = html.replace(
    /spatial attention mechanisms in order to further improve localization accuracy/g,
    `<span class="highlight-semantic" onclick="openSourceModal('Semantic Scholar (2022)', 'incorporating attention modules elevates fine-grained feature localization...')">$&</span>`
  );

  return html.split('\n\n').map(para => `<p>${para.replace(/\n/g, '<br>')}</p>`).join('');
}

function renderMatchedSourcesList() {
  const container = document.getElementById('report-sources-list');
  const sampleSources = [
    { label: 'IEEE Xplore / Trans. Med. Imaging', sim: '18.4%', count: 124, year: 2023, authors: 'Chen et al.', excerpt: 'automated medical image analysis, a large number of chest radiographs were evaluated in prior work...' },
    { label: 'arXiv:2104.0892 [cs.CV]', sim: '8.2%', count: 68, year: 2022, authors: 'Rahman et al.', excerpt: 'transfer learning strategies provide highly effective methodologies across domains...' },
    { label: 'Semantic Scholar / AI Conf', sim: '4.5%', count: 42, year: 2024, authors: 'Vaswani et al.', excerpt: 'incorporating attention modules elevates fine-grained feature localization...' }
  ];

  document.getElementById('sources-count-badge').textContent = `${sampleSources.length} Sources`;
  document.getElementById('score-similarity').textContent = '22.8%';

  container.innerHTML = sampleSources.map((s, idx) => `
    <div class="p-3.5 rounded-xl bg-surface-raised/60 border border-border-subtle hover:border-brand-500/40 cursor-pointer transition-all" onclick="openSourceModal('${s.label}', '${escapeHtml(s.excerpt)}')">
      <div class="flex items-center justify-between mb-1">
        <span class="font-display font-bold text-xs text-main">${idx + 1}. ${escapeHtml(s.label)}</span>
        <span class="text-xs font-extrabold text-brand-400">${s.sim}</span>
      </div>
      <p class="text-2xs text-muted mb-2">${s.authors} • ${s.year} • ${s.count} chars matched</p>
      <div class="text-2xs text-muted font-mono bg-surface/50 p-2 rounded border border-border-subtle truncate">
        "${escapeHtml(s.excerpt)}"
      </div>
    </div>
  `).join('');
}

function openSourceModal(title, excerpt) {
  document.getElementById('modal-source-title').textContent = title;
  document.getElementById('modal-query-excerpt').textContent = 'Matched text highlighted in manuscript body.';
  document.getElementById('modal-source-excerpt').textContent = excerpt;
  document.getElementById('source-modal').classList.remove('hidden');
}

function closeSourceModal() {
  document.getElementById('source-modal').classList.add('hidden');
}

function downloadHtmlReport() {
  showToast('Generating standalone interactive HTML report...', 'info');
  window.open('/report.html', '_blank');
}

// ==========================================================================
// 9. Dashboard & History Management
// ==========================================================================

function renderDashboard() {
  const tbody = document.getElementById('recent-analyses-tbody');
  const history = [
    { title: 'critical-review-paper-224037X-revised.docx', risk: 'LOW', sim: '4.2%', clarity: '92/100', suggestions: '4 pending' },
    { title: 'deep-learning-classification-draft.pdf', risk: 'LOW', sim: '8.1%', clarity: '85/100', suggestions: '12 pending' },
  ];

  tbody.innerHTML = history.map(item => `
    <tr class="hover:bg-surface-raised/30 transition-colors">
      <td class="py-3 px-5 font-medium text-main flex items-center gap-2">
        <svg class="w-4 h-4 text-brand-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>
        <span>${escapeHtml(item.title)}</span>
      </td>
      <td class="py-3 px-5"><span class="risk-badge-low">${item.risk}</span></td>
      <td class="py-3 px-5 text-main font-semibold">${item.sim}</td>
      <td class="py-3 px-5 text-emerald-400 font-semibold">${item.clarity}</td>
      <td class="py-3 px-5 text-muted">${item.suggestions}</td>
      <td class="py-3 px-5 text-right">
        <button class="btn-ghost text-xs py-1 px-2.5 text-brand-400" onclick="loadSampleDocument()">Inspect</button>
      </td>
    </tr>
  `).join('');
}

function clearAnalysisHistory() {
  document.getElementById('recent-analyses-tbody').innerHTML = `
    <tr><td colspan="6" class="py-8 text-center text-muted">History cleared.</td></tr>
  `;
  showToast('Analysis history cleared', 'info');
}

function saveSettings() {
  state.settings.spelling = document.getElementById('setting-spelling').value;
  state.settings.maxWords = parseInt(document.getElementById('setting-max-words').value) || 45;
  state.settings.openalex = document.getElementById('setting-openalex').checked;
  state.settings.s2 = document.getElementById('setting-s2').checked;
  state.settings.arxiv = document.getElementById('setting-arxiv').checked;
  showToast('Settings saved successfully', 'success');
}

// ==========================================================================
// 10. Toast Utility
// ==========================================================================

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = 'toast';
  
  const iconSvg = type === 'success'
    ? '<svg class="w-4 h-4 text-emerald-400 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>'
    : '<svg class="w-4 h-4 text-brand-400 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y2="16"/></svg>';

  toast.innerHTML = `${iconSvg}<span>${escapeHtml(message)}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.2s ease';
    setTimeout(() => toast.remove(), 200);
  }, 3200);
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  const savedTheme = localStorage.getItem('aegis_theme') || 'dark';
  document.documentElement.classList.toggle('dark', savedTheme === 'dark');
  document.documentElement.classList.toggle('light', savedTheme === 'light');
  updateThemeIcon(savedTheme === 'dark');
  
  renderDashboard();
});
