/* Dashboard JS — face registration modal */

let webcamStream = null;

async function openFaceModal() {
  document.getElementById('faceModal').style.display = 'flex';
  document.getElementById('faceStep1').style.display = 'block';
  document.getElementById('faceStep2').style.display = 'none';
  await startWebcam();
}

function closeFaceModal() {
  stopWebcam();
  document.getElementById('faceModal').style.display = 'none';
}

async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    });
    document.getElementById('webcamFeed').srcObject = webcamStream;
  } catch (err) {
    alert('Camera access denied. Please allow camera access and try again.');
  }
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
}

async function captureAndRegister() {
  const video  = document.getElementById('webcamFeed');
  const canvas = document.getElementById('snapCanvas');
  const btn    = document.getElementById('captureBtn');

  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  const imageData = canvas.toDataURL('image/jpeg', 0.92);

  btn.disabled = true;
  btn.textContent = 'Processing…';

  try {
    const res = await fetch('/api/register-face', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData })
    });
    const data = await res.json();

    stopWebcam();
    document.getElementById('faceStep1').style.display = 'none';
    document.getElementById('faceStep2').style.display = 'block';

    const resultEl = document.getElementById('faceResult');
    if (data.ok) {
      resultEl.className = 'face-result success';
      resultEl.innerHTML = `
        <div style="font-size:40px;margin-bottom:12px">✓</div>
        <strong style="font-size:16px">${data.message}</strong>
        <p style="margin-top:10px;font-size:13px;opacity:0.8">
          You can now start your exam. Reloading dashboard…
        </p>
      `;
      setTimeout(() => window.location.reload(), 2000);
    } else {
      resultEl.className = 'face-result error';
      resultEl.innerHTML = `
        <div style="font-size:40px;margin-bottom:12px">⚠</div>
        <strong>${data.error || 'Registration failed'}</strong>
        <p style="margin-top:10px;font-size:13px;opacity:0.8">Please try again with better lighting.</p>
        <button onclick="openFaceModal()" style="margin-top:16px;padding:8px 20px;background:transparent;border:1px solid currentColor;border-radius:6px;color:inherit;cursor:pointer;font-family:inherit">
          Try Again
        </button>
      `;
    }
  } catch (err) {
    btn.disabled = false;
    btn.innerHTML = '◉ Capture &amp; Register';
    alert('Network error. Please try again.');
  }
}

// Close modal on overlay click
document.getElementById('faceModal')?.addEventListener('click', function(e) {
  if (e.target === this) closeFaceModal();
});
