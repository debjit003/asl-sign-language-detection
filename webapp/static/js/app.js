/**
 * ASL Sign Language Detection — Frontend Logic
 * Handles WebSocket, webcam, video upload, landmark drawing, and UI updates.
 */

// ── State ────────────────────────────────────────────────────────────
let socket = null;
let stream = null;
let sendInterval = null;
let cameraActive = false;
let frameCount = 0;
let fpsTimestamp = Date.now();
let lastLetter = '';
let waitingForResponse = false;  // Client-side frame pacing

// ── DOM Elements ─────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const video = $('#webcamVideo');
const canvas = $('#webcamCanvas');
const ctx = canvas.getContext('2d');

// Landmark overlay canvas
let overlayCanvas = null;
let overlayCtx = null;

function ensureOverlayCanvas() {
    if (overlayCanvas) return;
    overlayCanvas = document.createElement('canvas');
    overlayCanvas.id = 'landmarkOverlay';
    overlayCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;transform:scaleX(-1);';
    const wrapper = $('#webcamWrapper');
    wrapper.style.position = 'relative';
    wrapper.appendChild(overlayCanvas);
    overlayCtx = overlayCanvas.getContext('2d');
}

// ── Socket.IO Connection ─────────────────────────────────────────────
function initSocket() {
    socket = io({ transports: ['websocket', 'polling'] });

    socket.on('connect', () => {
        updateConnectionStatus('connected', 'Connected');
    });

    socket.on('disconnect', () => {
        updateConnectionStatus('disconnected', 'Disconnected');
    });

    socket.on('connect_error', () => {
        updateConnectionStatus('disconnected', 'Connection Error');
    });

    socket.on('prediction', (data) => {
        waitingForResponse = false;  // Allow next frame to be sent
        updatePrediction(data);
        drawLandmarks(data);
    });
}

function updateConnectionStatus(state, text) {
    const el = $('#connectionStatus');
    el.className = 'connection-status ' + state;
    el.querySelector('.status-text').textContent = text;
}

// ── Tab Switching ────────────────────────────────────────────────────
$$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        $$('.tab-btn').forEach(b => b.classList.remove('active'));
        $$('.tab-content').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');
        $('#tab-' + btn.dataset.tab).classList.add('active');
    });
});

// ── Camera Controls ──────────────────────────────────────────────────
$('#startCameraBtn').addEventListener('click', startCamera);
$('#stopCameraBtn').addEventListener('click', stopCamera);

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
        });
        video.srcObject = stream;
        video.play();

        cameraActive = true;
        $('#webcamOverlay').classList.add('hidden');
        $('#startCameraBtn').style.display = 'none';
        $('#stopCameraBtn').style.display = 'inline-flex';

        ensureOverlayCanvas();

        // Send frames as fast as server can process (client-paced)
        sendInterval = setInterval(captureAndSend, 80); // ~12 FPS max attempt
        fpsTimestamp = Date.now();
        frameCount = 0;
    } catch (err) {
        alert('Cannot access camera: ' + err.message);
    }
}

function stopCamera() {
    cameraActive = false;
    if (sendInterval) { clearInterval(sendInterval); sendInterval = null; }
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
    video.srcObject = null;

    $('#webcamOverlay').classList.remove('hidden');
    $('#startCameraBtn').style.display = 'inline-flex';
    $('#stopCameraBtn').style.display = 'none';
    $('#fpsCounter').textContent = '0 FPS';

    // Clear overlay
    if (overlayCtx && overlayCanvas) {
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
}

function captureAndSend() {
    if (!cameraActive || !socket || !socket.connected) return;

    // Client-side pacing: don't send until previous response received
    if (waitingForResponse) return;

    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;

    // Capture at smaller resolution for faster transfer
    const scale = 0.75;
    canvas.width = Math.round(vw * scale);
    canvas.height = Math.round(vh * scale);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to JPEG base64 (strip prefix)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.65);
    const base64 = dataUrl.split(',')[1];

    waitingForResponse = true;
    socket.emit('frame', { data: base64 });

    // FPS counter
    frameCount++;
    const elapsed = (Date.now() - fpsTimestamp) / 1000;
    if (elapsed >= 1) {
        $('#fpsCounter').textContent = Math.round(frameCount / elapsed) + ' FPS';
        frameCount = 0;
        fpsTimestamp = Date.now();
    }
}

// ── Draw Hand Landmarks ──────────────────────────────────────────────
const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17],
];

function drawLandmarks(data) {
    if (!overlayCanvas || !overlayCtx) return;

    const wrapper = $('#webcamWrapper');
    const rect = wrapper.getBoundingClientRect();
    overlayCanvas.width = rect.width;
    overlayCanvas.height = rect.height;
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (!data.landmarks || !data.hand_detected) return;

    const lm = data.landmarks;
    const w = overlayCanvas.width;
    const h = overlayCanvas.height;

    // Draw connections (green lines)
    overlayCtx.strokeStyle = '#00e676';
    overlayCtx.lineWidth = 2.5;
    overlayCtx.lineCap = 'round';
    for (const [i, j] of HAND_CONNECTIONS) {
        overlayCtx.beginPath();
        overlayCtx.moveTo(lm[i].x * w, lm[i].y * h);
        overlayCtx.lineTo(lm[j].x * w, lm[j].y * h);
        overlayCtx.stroke();
    }

    // Draw landmark points
    for (let i = 0; i < lm.length; i++) {
        const x = lm[i].x * w;
        const y = lm[i].y * h;

        // Fingertips in red, others in white
        const isTip = [4, 8, 12, 16, 20].includes(i);

        // Outer ring
        overlayCtx.beginPath();
        overlayCtx.arc(x, y, isTip ? 6 : 4, 0, Math.PI * 2);
        overlayCtx.fillStyle = isTip ? '#ff1744' : '#ffffff';
        overlayCtx.fill();

        // Inner dot
        overlayCtx.beginPath();
        overlayCtx.arc(x, y, isTip ? 3 : 2, 0, Math.PI * 2);
        overlayCtx.fillStyle = isTip ? '#ff8a80' : '#b3e5fc';
        overlayCtx.fill();
    }
}

// ── Update Prediction UI ─────────────────────────────────────────────
function updatePrediction(data) {
    // Hand indicator
    const indicator = $('#handIndicator');
    if (data.hand_detected) {
        indicator.classList.add('detected');
        $('#handStatusText').textContent = 'Hand detected';
    } else {
        indicator.classList.remove('detected');
        $('#handStatusText').textContent = 'No hand detected';
    }

    const activeLetter = data.smoothed_letter || '';
    const conf = data.smoothed_letter ? (data.smoothed_confidence || 0) : 0;

    const charEl = $('#letterChar');
    if (activeLetter) {
        if (activeLetter !== lastLetter) {
            charEl.classList.remove('pop');
            void charEl.offsetWidth; // force reflow
            charEl.classList.add('pop');
            lastLetter = activeLetter;
        }
        charEl.textContent = activeLetter;
    } else {
        charEl.textContent = '?';
        lastLetter = '';
    }

    // Confidence
    $('#confidenceValue').textContent = conf.toFixed(1) + '%';
    const bar = $('#confidenceBar');
    bar.style.width = conf + '%';
    bar.style.background = conf > 80
        ? 'linear-gradient(90deg, #22c55e, #06b6d4)'
        : conf > 50
            ? 'linear-gradient(90deg, #eab308, #f97316)'
            : 'linear-gradient(90deg, #ef4444, #f97316)';

    // Hold progress
    if (data.hold_progress !== undefined) {
        const hp = Math.min(data.hold_progress, 1) * 100;
        $('#holdBar').style.width = hp + '%';
        $('#holdStatus').textContent = hp >= 100 ? 'Locked!' : hp > 0 ? Math.round(hp) + '%' : '--';
    }

    // Locked flash
    if (data.locked) {
        const ld = $('#letterDisplay');
        ld.classList.remove('locked');
        void ld.offsetWidth;
        ld.classList.add('locked');
    }

    // Top 3
    if (data.top3) {
        const items = $$('#top3List .top3-item');
        data.top3.forEach((pred, i) => {
            if (items[i]) {
                items[i].querySelector('.top3-letter').textContent = pred.letter;
                items[i].querySelector('.top3-bar-fill').style.width = pred.confidence + '%';
                items[i].querySelector('.top3-conf').textContent = pred.confidence.toFixed(1) + '%';
            }
        });
    }

    // Sentence
    if (data.sentence !== undefined) {
        const textEl = $('#sentenceText');
        if (data.sentence) {
            textEl.textContent = data.sentence;
            textEl.classList.remove('placeholder');
        } else {
            textEl.textContent = 'Start signing to build a sentence...';
            textEl.classList.add('placeholder');
        }
    }
}

// ── Sentence Controls ────────────────────────────────────────────────
$('#clearBtn').addEventListener('click', () => {
    if (socket) socket.emit('reset_sentence');
});

$('#backspaceBtn').addEventListener('click', () => {
    if (socket) socket.emit('backspace');
});

$('#addSpaceBtn').addEventListener('click', () => {
    if (socket) socket.emit('add_space');
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'Escape') { if (socket) socket.emit('reset_sentence'); e.preventDefault(); }
    if (e.key === 'Backspace') { if (socket) socket.emit('backspace'); e.preventDefault(); }
    if (e.key === ' ') { if (socket) socket.emit('add_space'); e.preventDefault(); }
});

// ── Video Upload ─────────────────────────────────────────────────────
const uploadZone = $('#uploadZone');
const fileInput = $('#videoFileInput');

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleVideoFile(files[0]);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleVideoFile(fileInput.files[0]);
});

async function handleVideoFile(file) {
    if (!file.type.startsWith('video/')) {
        alert('Please select a video file');
        return;
    }

    // Show progress
    $('#uploadFileName').textContent = file.name;
    uploadZone.style.display = 'none';
    $('#uploadProgress').style.display = 'block';
    $('#uploadProgressBar').style.width = '0%';
    $('#uploadPercent').textContent = '0%';
    $('#uploadStatus').textContent = 'Uploading video...';
    $('#transcriptionEmpty').style.display = 'flex';
    $('#transcriptionResults').style.display = 'none';
    $('#downloadBtn').style.display = 'none';

    // Preview
    const previewUrl = URL.createObjectURL(file);
    $('#previewPlayer').src = previewUrl;
    $('#videoPreview').style.display = 'block';

    // Upload
    const formData = new FormData();
    formData.append('video', file);

    try {
        // Simulate progress since fetch doesn't support upload progress easily
        let progressInterval = setInterval(() => {
            const bar = $('#uploadProgressBar');
            const current = parseFloat(bar.style.width) || 0;
            if (current < 85) {
                const next = current + Math.random() * 5;
                bar.style.width = next + '%';
                $('#uploadPercent').textContent = Math.round(next) + '%';
            }
        }, 300);

        $('#uploadStatus').textContent = 'Processing video... This may take a moment.';

        const response = await fetch('/api/upload-video', {
            method: 'POST',
            body: formData,
        });

        clearInterval(progressInterval);
        $('#uploadProgressBar').style.width = '100%';
        $('#uploadPercent').textContent = '100%';

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Upload failed');
        }

        const result = await response.json();
        $('#uploadStatus').textContent = 'Processing complete!';
        displayTranscription(result);

    } catch (err) {
        $('#uploadStatus').textContent = 'Error: ' + err.message;
        $('#uploadProgressBar').style.background = 'var(--red)';
    }
}

function displayTranscription(result) {
    $('#transcriptionEmpty').style.display = 'none';
    $('#transcriptionResults').style.display = 'block';
    $('#downloadBtn').style.display = 'inline-flex';

    // Stats
    $('#videoStats').innerHTML = `
        <div class="stat-card"><div class="stat-value">${result.duration}s</div><div class="stat-label">Duration</div></div>
        <div class="stat-card"><div class="stat-value">${result.detections?.length || 0}</div><div class="stat-label">Signs Detected</div></div>
        <div class="stat-card"><div class="stat-value">${result.frames_processed || 0}</div><div class="stat-label">Frames Analyzed</div></div>
    `;

    // Full transcription
    $('#fullTranscription').textContent = result.transcription || '(No signs detected)';

    // Timeline
    const timeline = $('#timelineList');
    timeline.innerHTML = '';
    if (result.detections) {
        result.detections.forEach(d => {
            const item = document.createElement('div');
            item.className = 'timeline-item';
            item.innerHTML = `
                <span class="timeline-time">${formatTime(d.time)}</span>
                <span class="timeline-letter">${d.letter}</span>
                <span class="timeline-conf">${d.confidence.toFixed(1)}%</span>
            `;
            timeline.appendChild(item);
        });
    }

    // Store for download
    window._lastTranscription = result;
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.round((seconds % 1) * 10);
    return `${m}:${String(s).padStart(2, '0')}.${ms}`;
}

// ── Download Transcription ───────────────────────────────────────────
$('#downloadBtn').addEventListener('click', () => {
    const result = window._lastTranscription;
    if (!result) return;

    let text = 'ASL Sign Language Detection — Transcription\n';
    text += '=' .repeat(50) + '\n\n';
    text += 'Transcription: ' + (result.transcription || '(none)') + '\n\n';
    text += 'Video Duration: ' + result.duration + 's\n';
    text += 'Signs Detected: ' + (result.detections?.length || 0) + '\n';
    text += 'Frames Analyzed: ' + (result.frames_processed || 0) + '\n\n';
    text += 'Timeline:\n';
    text += '-'.repeat(40) + '\n';
    if (result.detections) {
        result.detections.forEach(d => {
            text += `  ${formatTime(d.time)}  →  ${d.letter}  (${d.confidence.toFixed(1)}%)\n`;
        });
    }

    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'transcription.txt';
    a.click();
    URL.revokeObjectURL(url);
});

// ── Initialize ───────────────────────────────────────────────────────
initSocket();
