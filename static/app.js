/* 
    app.js — Agamotto Dashboard Logic
    ──────────────────────────────────
    Handles real-time score updates, violation feed, and HUD state.
*/

class ProctorDashboard {
  constructor() {
    this.pollInterval = 500; // ms
    this.lastViolationId = null;
    this.isAlerting = false;
    
    // UI Elements
    this.elements = {
      clock: document.getElementById('clock'),
      fps: document.getElementById('fps-display'),
      faceStatus: document.getElementById('face-status'),
      headDir: document.getElementById('head-dir'),
      phoneStatus: document.getElementById('phone-status'),
      scoreVal: document.getElementById('score-val'),
      scoreGauge: document.getElementById('score-gauge'),
      violationFeed: document.getElementById('violation-feed'),
      violationCount: document.getElementById('violation-count'),
      alertText: document.getElementById('alert-status-text')
    };

    this.init();
  }

  init() {
    this.startClock();
    this.startPolling();
    console.log("[Agamotto] Dashboard initialized.");
  }

  startClock() {
    const update = () => {
      const now = new Date();
      this.elements.clock.textContent = now.toLocaleTimeString();
    };
    setInterval(update, 1000);
    update();
  }

  async startPolling() {
    while (true) {
      try {
        const response = await fetch('/stats');
        const data = await response.json();
        this.updateUI(data);
      } catch (err) {
        console.warn("[Agamotto] State sync failed:", err);
      }
      await new Promise(r => setTimeout(r, this.pollInterval));
    }
  }

  updateUI(data) {
    const { live, violations } = data;

    // 1. Basic Stats
    this.elements.fps.textContent = `${live.fps.toFixed(1)} FPS`;
    this.elements.faceStatus.textContent = live.face_present ? "DETECTED" : "ABSENT";
    this.elements.faceStatus.parentElement.parentElement.classList.toggle('danger', !live.face_present);
    
    this.elements.headDir.textContent = live.head_direction.toUpperCase();
    this.elements.phoneStatus.textContent = live.phone_detected ? "DETECTED!" : "CLEAN";
    this.elements.phoneStatus.classList.toggle('text-danger', live.phone_detected);

    // 2. Suspicion Score & Gauge
    this.elements.scoreVal.textContent = live.suspicion_score.toFixed(2);
    const scorePct = Math.min((live.suspicion_score / 16.0) * 100, 100);
    this.elements.scoreGauge.style.width = `${scorePct}%`;

    // 3. Status Banner
    const statusSpan = this.elements.alertText.querySelector('span');
    statusSpan.textContent = live.alert_level.toUpperCase();
    
    if (live.alert_level === 'Alert') {
      statusSpan.style.color = 'var(--danger)';
      document.body.classList.add('critical-alert');
    } else if (live.alert_level === 'Warning') {
      statusSpan.style.color = 'var(--warning)';
      document.body.classList.remove('critical-alert');
    } else {
      statusSpan.style.color = 'var(--success)';
      document.body.classList.remove('critical-alert');
    }

    // 4. Violation Feed
    this.updateViolationFeed(violations);
  }

  updateViolationFeed(violations) {
    if (!violations || violations.length === 0) return;

    this.elements.violationCount.textContent = violations.length;
    
    // Check if we have new violations
    const topV = violations[0];
    if (topV.id !== this.lastViolationId) {
      this.lastViolationId = topV.id;
      this.renderViolations(violations);
    }
  }

  renderViolations(violations) {
    const container = this.elements.violationFeed;
    container.innerHTML = ''; // Clear feed

    violations.forEach(v => {
      const item = document.createElement('div');
      item.className = 'violation-item';
      
      const levelClass = v.level.toLowerCase();
      
      item.innerHTML = `
        <div class="v-head ${levelClass}">
          <span>${v.level.toUpperCase()}</span>
          <span>${v.time}</span>
        </div>
        <div class="v-body">${v.reason}</div>
        <div class="v-time">Score Influence: +${v.score}</div>
      `;
      container.appendChild(item);
    });
  }
}

// Initialize on load
window.addEventListener('DOMContentLoaded', () => {
  window.dashboard = new ProctorDashboard();
});
