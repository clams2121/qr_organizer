/* Continuous QR scanning in the browser.
 *
 * Prefers the native BarcodeDetector where it exists (Chrome/Android: hardware
 * accelerated, no main-thread cost) and falls back to the bundled jsQR
 * elsewhere. Camera access requires a secure context, so an http:// page says
 * so plainly rather than showing a dead viewfinder.
 */
(function () {
  const video = document.getElementById('video');
  const canvas = document.getElementById('frame');
  const hint = document.getElementById('hint');
  const insecure = document.getElementById('insecure');
  const viewfinder = document.getElementById('viewfinder');

  let scanning = true;
  let detector = null;

  window.go = function (raw) {
    if (!raw) return;
    fetch('/api/scan/resolve', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Requested-With': 'fetch'},
      body: JSON.stringify({payload: raw}),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.ok) {
          window.location = data.url;
        } else {
          hint.textContent = data.error;
          scanning = true;   // let the user try another label
        }
      })
      .catch((err) => { hint.textContent = 'Could not reach the server: ' + err; });
  };

  if (!window.isSecureContext || !navigator.mediaDevices) {
    insecure.hidden = false;
    viewfinder.hidden = true;
    return;
  }

  async function start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {facingMode: 'environment'},
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
    } catch (err) {
      hint.textContent = 'Camera unavailable: ' + err.message +
        ' — you can type the code below instead.';
      return;
    }

    if ('BarcodeDetector' in window) {
      try {
        const formats = await window.BarcodeDetector.getSupportedFormats();
        if (formats.includes('qr_code')) {
          detector = new window.BarcodeDetector({formats: ['qr_code']});
        }
      } catch (err) { detector = null; }
    }
    requestAnimationFrame(tick);
  }

  async function tick() {
    if (scanning && video.readyState === video.HAVE_ENOUGH_DATA) {
      const found = detector ? await viaDetector() : viaJsQR();
      if (found) {
        scanning = false;
        hint.textContent = 'Found ' + found;
        if (navigator.vibrate) navigator.vibrate(60);
        window.go(found);
      }
    }
    requestAnimationFrame(tick);
  }

  async function viaDetector() {
    try {
      const codes = await detector.detect(video);
      return codes.length ? codes[0].rawValue : null;
    } catch (err) {
      detector = null;   // fall back permanently rather than erroring every frame
      return null;
    }
  }

  function viaJsQR() {
    if (typeof jsQR !== 'function') return null;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return null;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', {willReadFrequently: true});
    context.drawImage(video, 0, 0, width, height);
    const image = context.getImageData(0, 0, width, height);
    const result = jsQR(image.data, width, height, {inversionAttempts: 'dontInvert'});
    return result ? result.data : null;
  }

  start();
})();
