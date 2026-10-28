/* ================================================================
   Exam JS
   - Pre-exam setup + proctoring start
   - Question navigation
   - Countdown timer (auto-submit)
   - Webcam thumbnail
   - Tab-visibility / fullscreen violation logging
   - Submit flow
================================================================ */

'use strict';

// State
let currentQ     = 0;
const answers    = {};          // { questionId: optionIndex }
let timerSeconds = DURATION;
let timerInterval = null;
let examStarted  = false;
let camStream    = null;

// ── DOM refs (resolved after overlay is dismissed) ──────────────
const preOverlay   = () => document.getElementById('preExamOverlay');
const examShell    = () => document.getElementById('examShell');
const qArea        = () => document.getElementById('questionArea');
const progressBar  = () => document.getElementById('progressBar');
const timerEl      = () => document.getElementById('timerDisplay');
const timerWrap    = () => document.getElementById('timerWrap');
const qCurrent     = () => document.getElementById('qCurrent');
const qTotal       = () => document.getElementById('qTotal');
const prevBtn      = () => document.getElementById('prevBtn');
const nextBtn      = () => document.getElementById('nextBtn');

// ── Utility ─────────────────────────────────────────────────────

function pad(n) { return String(n).padStart(2, '0'); }

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${pad(m)}:${pad(sec)}`;
}

function logEvent(type, detail = '', severity = 'info') {
  fetch('/api/log-event', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type, detail, severity })
  }).catch(() => {});
}

// ── Webcam thumbnail ─────────────────────────────────────────────


async function startCamThumb() {
  try {
    /* 
      Windows Camera Collision Fix:
      If the browser accesses the webcam via WebRTC, the background 
      Python process (OpenCV) will be unable to simultaneously read from cv2.VideoCapture(0).
      Since the backend must perform the YOLO and MediaPipe detections, we disable the 
      frontend webcam thumbnail. The backend will pop up its own window.
    */
    console.log("Frontend webcam thumbnail disabled to allow backend OpenCV access.");
    // camStream = await navigator.mediaDevices.getUserMedia({
    //   video: { width: { ideal: 320 }, height: { ideal: 240 }, facingMode: 'user' }
    // });
    // const vid = document.getElementById('camThumb');
    // if (vid) vid.srcObject = camStream;
  } catch (e) {
    console.warn('Camera unavailable for thumbnail:', e);
    logEvent('camera_error', 'Webcam access denied', 'high');
  }
}

function stopCamThumb() {
  if (camStream) {
    camStream.getTracks().forEach(t => t.stop());
    camStream = null;
  }
}

// ── Timer ────────────────────────────────────────────────────────

function startTimer() {
  timerEl().textContent = fmtTime(timerSeconds);
  timerInterval = setInterval(() => {
    timerSeconds--;
    timerEl().textContent = fmtTime(timerSeconds);

    const wrap = timerWrap();
    if (timerSeconds <= 60) {
      wrap.className = 'timer-wrap timer-critical';
    } else if (timerSeconds <= 300) {
      wrap.className = 'timer-wrap timer-warn';
    }

    if (timerSeconds <= 0) {
      clearInterval(timerInterval);
      logEvent('timer_expired', 'Exam auto-submitted due to timeout', 'medium');
      doSubmit(true);
    }
  }, 1000);
}

// ── Question rendering ───────────────────────────────────────────

function renderQuestion(idx) {
  const q   = QUESTIONS[idx];
  const sel = answers[q.id];

  qArea().innerHTML = `
    <div class="question-meta">
      Question ${idx + 1} of ${QUESTIONS.length} &nbsp;·&nbsp; ${q.topic}
    </div>
    <div class="question-text">${q.text}</div>
    <div class="options-list">
      ${q.options.map((opt, i) => `
        <button class="option-btn ${sel === i ? 'selected' : ''}"
                onclick="selectOption(${q.id}, ${i}, this)">
          <span class="option-marker">${'ABCD'[i]}</span>
          ${opt}
        </button>
      `).join('')}
    </div>
  `;

  // Update nav
  prevBtn().disabled = (idx === 0);
  nextBtn().disabled = (idx === QUESTIONS.length - 1);
  qCurrent().textContent = idx + 1;

  // Progress bar
  const answered = Object.keys(answers).length;
  progressBar().style.width = `${(answered / QUESTIONS.length) * 100}%`;

  // Question map
  updateQmap(idx);
}

function selectOption(qId, optIdx, btn) {
  answers[qId] = optIdx;
  // Visual update: deselect siblings
  btn.closest('.options-list').querySelectorAll('.option-btn').forEach(b => {
    b.classList.remove('selected');
    b.querySelector('.option-marker').textContent = 'ABCD'[
      Array.from(b.closest('.options-list').children).indexOf(b)
    ];
  });
  btn.classList.add('selected');
  btn.querySelector('.option-marker').textContent = '✓';

  // Progress
  const answered = Object.keys(answers).length;
  progressBar().style.width = `${(answered / QUESTIONS.length) * 100}%`;
  updateQmap(currentQ);
}

function navigate(dir) {
  const next = currentQ + dir;
  if (next < 0 || next >= QUESTIONS.length) return;
  currentQ = next;
  renderQuestion(currentQ);
}

function goToQuestion(idx) {
  currentQ = idx;
  renderQuestion(currentQ);
}

function updateQmap(currentIdx) {
  QUESTIONS.forEach((q, i) => {
    const btn = document.getElementById(`qmap-${i + 1}`);
    if (!btn) return;
    btn.className = 'qmap-btn';
    if (i === currentIdx) btn.classList.add('qmap-current');
    else if (answers[q.id] !== undefined) btn.classList.add('qmap-answered');
  });
}

// ── Submit flow ──────────────────────────────────────────────────

function openSubmitModal() {
  const answered = Object.keys(answers).length;
  document.getElementById('answeredCount').textContent = answered;
  document.getElementById('submitModal').style.display = 'flex';
}

function closeModal(id) {
  document.getElementById(id).style.display = 'none';
}

document.addEventListener('DOMContentLoaded', () => {
  // Wire submit buttons
  ['submitTopBtn', 'submitSideBtn'].forEach(id => {
    document.getElementById(id)?.addEventListener('click', openSubmitModal);
  });

  document.getElementById('confirmSubmitBtn')?.addEventListener('click', () => {
    closeModal('submitModal');
    doSubmit(false);
  });

  // Prevent accidental navigation away
  window.addEventListener('beforeunload', (e) => {
    if (examStarted) {
      e.preventDefault();
      e.returnValue = 'Exam is in progress. Are you sure?';
    }
  });

  // Tab visibility detection
  document.addEventListener('visibilitychange', () => {
    if (examStarted && document.hidden) {
      logEvent('tab_switch', 'User switched tab or minimised window', 'high');
    }
  });

  // Fullscreen exit detection
  document.addEventListener('fullscreenchange', () => {
    if (examStarted && !document.fullscreenElement) {
      logEvent('fullscreen_exit', 'Exited fullscreen during exam', 'medium');
    }
  });

  // Disable right-click and copy during exam
  document.addEventListener('contextmenu', (e) => { if (examStarted) e.preventDefault(); });
  document.addEventListener('copy',        (e) => {
    if (examStarted) {
      e.preventDefault();
      logEvent('copy_attempt', 'User tried to copy text', 'medium');
    }
  });

  // Start Exam button
  document.getElementById('startExamBtn')?.addEventListener('click', startExam);
});

async function startExam() {
  const btn = document.getElementById('startExamBtn');
  btn.disabled = true;
  btn.querySelector('span').textContent = 'Initialising…';

  try {
    const res  = await fetch('/api/start-exam', { method: 'POST' });
    const data = await res.json();
    if (!data.ok) { alert('Failed to start exam session.'); btn.disabled = false; return; }
  } catch (e) {
    alert('Network error. Please refresh and try again.');
    btn.disabled = false;
    return;
  }

  // Request fullscreen
  try { await document.documentElement.requestFullscreen(); } catch (_) {}

  // Hide overlay, show exam
  preOverlay().style.display = 'none';
  examShell().style.display  = 'flex';
  examStarted = true;

  // Init UI
  qTotal().textContent = QUESTIONS.length;
  renderQuestion(0);
  startTimer();
  startCamThumb();

  logEvent('exam_started', 'Exam session started', 'info');
}

async function doSubmit(autoSubmit = false) {
  examStarted = false;
  clearInterval(timerInterval);
  stopCamThumb();

  // Exit fullscreen
  try { document.exitFullscreen(); } catch (_) {}

  // Disable all interactive elements
  document.querySelectorAll('button').forEach(b => b.disabled = true);

  if (autoSubmit) {
    logEvent('auto_submit', 'Timer expired — exam auto-submitted', 'info');
  } else {
    logEvent('manual_submit', 'User manually submitted exam', 'info');
  }

  // Build answer payload: { "questionId": selectedOptionIndex }
  const payload = {};
  QUESTIONS.forEach(q => {
    payload[q.id] = answers[q.id] ?? -1;
  });

  try {
    const res  = await fetch('/api/submit-exam', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answers: payload })
    });
    const data = await res.json();
    if (data.ok && data.redirect) {
      window.location.href = data.redirect;
    } else {
      alert('Submission error. Please contact your invigilator.');
    }
  } catch (e) {
    alert('Network error during submission. Please contact your invigilator.');
  }
}
