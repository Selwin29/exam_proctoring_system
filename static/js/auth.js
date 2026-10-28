/* Auth page JS — login + register */

document.addEventListener('DOMContentLoaded', () => {

  // ── Login form ──────────────────────────────────────────
  const loginForm = document.getElementById('loginForm');
  if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = document.getElementById('loginBtn');
      const label = btn.querySelector('.btn-label');
      label.textContent = 'Signing in…';
      btn.disabled = true;

      const res = await fetch('/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: document.getElementById('username').value,
          password: document.getElementById('password').value,
        })
      });

      const data = await res.json();
      if (data.ok) {
        label.textContent = '✓ Redirecting…';
        window.location.href = '/dashboard';
      } else {
        label.textContent = 'Sign In';
        btn.disabled = false;
        showError(data.error || 'Login failed.');
      }
    });
  }

  // ── Register form ────────────────────────────────────────
  const regForm = document.getElementById('registerForm');
  if (regForm) {
    regForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = document.getElementById('registerBtn');
      const label = btn.querySelector('.btn-label');
      const pw  = document.getElementById('password').value;
      const pw2 = document.getElementById('confirm').value;

      if (pw !== pw2) { showError('Passwords do not match.'); return; }

      label.textContent = 'Creating account…';
      btn.disabled = true;

      const res = await fetch('/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: document.getElementById('username').value,
          email:    document.getElementById('email').value,
          password: pw,
        })
      });

      const data = await res.json();
      if (data.ok) {
        document.getElementById('registerForm').style.display = 'none';
        document.getElementById('successMsg').style.display = 'flex';
        setTimeout(() => { window.location.href = '/login'; }, 1800);
      } else {
        label.textContent = 'Create Account';
        btn.disabled = false;
        showError(data.error || 'Registration failed.');
      }
    });
  }

  function showError(msg) {
    let el = document.querySelector('.alert-error');
    if (!el) {
      el = document.createElement('div');
      el.className = 'alert alert-error';
      el.innerHTML = '<span class="alert-icon">⚠</span><span></span>';
      const form = document.querySelector('.auth-form-wrap');
      form.insertBefore(el, form.querySelector('form'));
    }
    el.querySelector('span:last-child').textContent = msg;
    el.style.display = 'flex';
    // Auto-hide after 4s
    clearTimeout(el._timer);
    el._timer = setTimeout(() => { el.style.display = 'none'; }, 4000);
  }
});
