import { useState, useEffect, useRef, useCallback } from "react";
import FacialEmotionDetection from "./components/FacialEmotionDetection";
import Calibration from "./components/Calibration";

// ─── STYLES ──────────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Literata:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Geist+Mono:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:         #faf9f7;
    --bg2:        #f4f2ee;
    --surface:    #ffffff;
    --surface2:   #f9f7f4;
    --border:     #e8e4dc;
    --border2:    #d6d0c4;
    --text:       #1a1714;
    --text2:      #5a5248;
    --text3:      #9a9088;
    --accent:     #c96a2e;
    --accent-bg:  #fdf3eb;
    --accent-dim: #f0e0d0;
    --green:      #3a7d5a;
    --green-bg:   #edf5f0;
    --red:        #c0392b;
    --red-bg:     #fdf0ee;
    --yellow:     #b5830a;
    --shadow-sm:  0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:  0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-lg:  0 12px 32px rgba(0,0,0,0.1), 0 4px 8px rgba(0,0,0,0.06);
    --radius-sm:  6px;
    --radius-md:  10px;
    --radius-lg:  16px;
    --radius-xl:  24px;
  }

  html, body, #root {
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: 'Literata', Georgia, serif;
    -webkit-font-smoothing: antialiased;
  }

  /* ── LAYOUT ─────────────────────────────────────────────────── */
  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  /* ── HEADER ─────────────────────────────────────────────────── */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    height: 54px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    z-index: 50;
  }

  .logo {
    font-size: 16px;
    font-weight: 600;
    letter-spacing: -0.3px;
    color: var(--text);
  }

  .logo span {
    color: var(--accent);
  }

  .status-pill {
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: 'Geist Mono', monospace;
    font-size: 11px;
    color: var(--text3);
    background: var(--surface2);
    padding: 5px 12px;
    border-radius: 999px;
    border: 1px solid var(--border);
  }

  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--red);
    flex-shrink: 0;
    transition: background 0.3s, box-shadow 0.3s;
  }

  .status-dot.connected {
    background: var(--green);
    box-shadow: 0 0 0 2px var(--green-bg);
  }

  .header-right {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  /* ── TABS ───────────────────────────────────────────────────── */
  .tabs {
    display: flex;
    gap: 4px;
    padding: 0 20px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }

  .tab {
    padding: 12px 20px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text3);
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    transition: all 0.15s ease;
    font-family: 'Literata', serif;
  }

  .tab:hover {
    color: var(--text2);
  }

  .tab.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }

  /* ── MAIN ────────────────────────────────────────────────────── */
  .main {
    display: grid;
    grid-template-columns: 360px 1fr;
    flex: 1;
    overflow: hidden;
  }

  /* ── LEFT PANEL ──────────────────────────────────────────────── */
  .left-panel {
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border);
    background: var(--surface);
    overflow: hidden;
  }

  .panel-label {
    padding: 10px 16px;
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text3);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface2);
  }

  .hand-badge {
    display: flex;
    align-items: center;
    gap: 5px;
    color: var(--green);
    font-size: 10px;
  }

  .hand-badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 1.5s ease infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
  }

  /* ── CAMERA ──────────────────────────────────────────────────── */
  .camera-wrap {
    flex: 1;
    background: #f0ede8;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .camera-canvas {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    transform: scaleX(-1);
  }

  .camera-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    color: var(--text3);
    font-size: 13px;
    text-align: center;
    padding: 24px;
  }

  .camera-placeholder svg {
    opacity: 0.3;
  }

  .rec-badge {
    position: absolute;
    top: 10px;
    left: 10px;
    display: flex;
    align-items: center;
    gap: 7px;
    background: var(--red-bg);
    border: 1px solid rgba(192, 57, 43, 0.25);
    border-radius: var(--radius-sm);
    padding: 5px 11px;
    font-size: 11px;
    font-family: 'Geist Mono', monospace;
    color: var(--red);
    animation: fadeIn 0.2s ease;
  }

  .rec-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--red);
    animation: blink 1s step-end infinite;
  }

  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

  .recog-badge {
    position: absolute;
    bottom: 14px;
    left: 50%;
    transform: translateX(-50%);
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: var(--radius-md);
    padding: 8px 18px;
    font-size: 17px;
    font-weight: 500;
    color: var(--accent);
    white-space: nowrap;
    box-shadow: var(--shadow-md);
    transition: opacity 0.2s, transform 0.2s;
    opacity: 0;
    pointer-events: none;
    letter-spacing: 0.3px;
  }

  .recog-badge.show {
    opacity: 1;
    transform: translateX(-50%) translateY(-2px);
  }

  /* ── RECORD CONTROLS ─────────────────────────────────────────── */
  .record-section {
    padding: 14px;
    border-top: 1px solid var(--border);
    background: var(--surface2);
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .section-title {
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 2px;
  }

  .record-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 7px;
  }

  /* ── RIGHT PANEL ─────────────────────────────────────────────── */
  .right-panel {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--bg);
  }

  .slide-area {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
    padding: 24px;
  }

  .slide-canvas {
    max-width: 100%;
    max-height: 100%;
    border-radius: var(--radius-sm);
    box-shadow: var(--shadow-lg);
    background: white;
  }

  .upload-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 14px;
    text-align: center;
    padding: 32px;
  }

  .upload-empty-icon {
    width: 64px;
    height: 64px;
    border-radius: var(--radius-lg);
    background: var(--accent-bg);
    border: 1px solid var(--accent-dim);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--accent);
    font-size: 24px;
  }

  .upload-empty h3 {
    font-size: 17px;
    font-weight: 500;
    color: var(--text);
    letter-spacing: -0.2px;
  }

  .upload-empty p {
    font-size: 13px;
    color: var(--text3);
    line-height: 1.7;
    max-width: 260px;
    font-style: italic;
  }

  /* ── SLIDE CONTROLS ──────────────────────────────────────────── */
  .slide-controls {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    border-top: 1px solid var(--border);
    background: var(--surface);
  }

  .slide-count {
    font-family: 'Geist Mono', monospace;
    font-size: 12px;
    color: var(--text3);
    min-width: 64px;
    text-align: center;
  }

  .slide-spacer {
    flex: 1;
  }

  .divider {
    width: 1px;
    height: 18px;
    background: var(--border2);
    flex-shrink: 0;
  }

  /* ── CAPTIONS ────────────────────────────────────────────────── */
  .captions {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 18px;
    min-height: 58px;
    background: var(--surface);
    border-top: 1px solid var(--border);
    overflow: hidden;
  }

  .cc-label {
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    color: var(--text3);
    text-transform: uppercase;
    flex-shrink: 0;
    padding-right: 12px;
    border-right: 1px solid var(--border2);
  }

  .caption-text {
    flex: 1;
    font-size: 16px;
    font-weight: 400;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    direction: rtl;
    text-align: left;
    line-height: 1.4;
  }

  .caption-empty {
    color: var(--text3);
    font-size: 13px;
    font-style: italic;
    font-weight: 300;
  }

  @keyframes wordIn {
    from { opacity: 0; transform: translateY(3px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .caption-word {
    animation: wordIn 0.18s ease forwards;
  }

  /* ── BUTTONS ─────────────────────────────────────────────────── */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Literata', serif;
    font-size: 13px;
    font-weight: 500;
    padding: 7px 14px;
    border-radius: var(--radius-md);
    border: 1px solid transparent;
    cursor: pointer;
    transition: all 0.15s ease;
    line-height: 1;
    white-space: nowrap;
    text-decoration: none;
  }

  .btn:disabled { opacity: 0.45; cursor: not-allowed; }

  .btn-default {
    background: var(--surface);
    color: var(--text2);
    border-color: var(--border2);
  }
  .btn-default:hover:not(:disabled) {
    background: var(--surface2);
    border-color: var(--border2);
    color: var(--text);
  }

  .btn-primary {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }
  .btn-primary:hover:not(:disabled) { filter: brightness(0.92); }

  .btn-danger {
    background: transparent;
    color: var(--red);
    border-color: rgba(192, 57, 43, 0.3);
  }
  .btn-danger:hover:not(:disabled) {
    background: var(--red-bg);
  }

  .btn-success {
    background: transparent;
    color: var(--green);
    border-color: rgba(63, 192, 43, 0.3);
  }
  .btn-success:hover:not(:disabled) {
    background: var(--green-bg);
  }

  .btn-ghost {
    background: transparent;
    color: var(--text2);
    border-color: transparent;
  }
  .btn-ghost:hover:not(:disabled) {
    background: var(--surface2);
    color: var(--text);
  }

  .btn-icon {
    padding: 7px;
    background: var(--surface);
    color: var(--text2);
    border-color: var(--border);
    border-radius: var(--radius-sm);
  }
  .btn-icon:hover:not(:disabled) {
    background: var(--surface2);
    border-color: var(--border2);
    color: var(--text);
  }

  /* ── INPUT ───────────────────────────────────────────────────── */
  .input {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: var(--radius-md);
    padding: 8px 12px;
    color: var(--text);
    font-family: 'Literata', serif;
    font-size: 13px;
    outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  .input::placeholder { color: var(--text3); }
  .input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-bg);
  }

  .select {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: var(--radius-md);
    padding: 8px 12px;
    color: var(--text);
    font-family: 'Literata', serif;
    font-size: 13px;
    outline: none;
    cursor: pointer;
    transition: border-color 0.15s;
  }
  .select:focus { border-color: var(--accent); }

  /* ── MODAL ───────────────────────────────────────────────────── */
  .overlay {
    position: fixed;
    inset: 0;
    background: rgba(26, 23, 20, 0.4);
    backdrop-filter: blur(6px);
    z-index: 200;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: fadeIn 0.15s ease;
  }

  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

  .modal {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 28px;
    width: 460px;
    max-height: 80vh;
    overflow-y: auto;
    box-shadow: var(--shadow-lg);
    animation: slideUp 0.2s ease;
  }

  @keyframes slideUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .modal-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.3px;
    margin-bottom: 4px;
  }

  .modal-sub {
    font-size: 13px;
    color: var(--text3);
    font-style: italic;
    margin-bottom: 22px;
  }

  .modal-footer {
    display: flex;
    gap: 8px;
    justify-content: flex-end;
    margin-top: 22px;
    padding-top: 18px;
    border-top: 1px solid var(--border);
  }

  .form-group { margin-bottom: 16px; }

  .form-label {
    display: block;
    font-family: 'Geist Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 6px;
  }

  .range-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  input[type="range"] {
    flex: 1;
    accent-color: var(--accent);
    cursor: pointer;
  }

  .range-val {
    font-family: 'Geist Mono', monospace;
    font-size: 11px;
    color: var(--text3);
    min-width: 36px;
    text-align: right;
  }

  /* ── TEMPLATE LIST ────────────────────────────────────────────── */
  .template-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 320px;
    overflow-y: auto;
  }

  .template-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 10px 14px;
    gap: 12px;
  }

  .template-info {
    display: flex;
    align-items: center;
    gap: 10px;
    flex: 1;
    overflow: hidden;
  }

  .template-name {
    font-size: 14px;
    font-weight: 500;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .tag {
    font-family: 'Geist Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.5px;
    padding: 3px 7px;
    border-radius: 999px;
    border: 1px solid;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .tag-static {
    background: var(--accent-bg);
    color: var(--accent);
    border-color: var(--accent-dim);
  }

  .tag-dynamic {
    background: #edeaf8;
    color: #6c47c9;
    border-color: #d6cff0;
  }

  .tag-ready {
    background: var(--green-bg);
    color: var(--green);
    border-color: rgba(58, 125, 90, 0.2);
  }

  .tag-pending {
    background: #fef9ec;
    color: var(--yellow);
    border-color: rgba(181, 131, 10, 0.2);
  }

  .empty-state {
    text-align: center;
    padding: 32px 20px;
    color: var(--text3);
    font-size: 14px;
    font-style: italic;
  }

  /* ── SCROLLBAR ────────────────────────────────────────────────── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text3); }

  /* ── UTILS ────────────────────────────────────────────────────── */
  .sr-only {
    position: absolute; width: 1px; height: 1px;
    padding: 0; margin: -1px; overflow: hidden;
    clip: rect(0,0,0,0); border: 0;
  }

  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
`;

// ─── INJECT MEDIAPIPE SCRIPTS ─────────────────────────────────────────────────
function loadScript(src) {
  return new Promise((res, rej) => {
    if (document.querySelector(`script[src="${src}"]`)) return res();
    const s = document.createElement("script");
    s.src = src;
    s.crossOrigin = "anonymous";
    s.onload = res;
    s.onerror = rej;
    document.head.appendChild(s);
  });
}

async function initMP(videoEl, canvasEl, onResults) {
  await loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js");
  await loadScript(
    "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"
  );
  await loadScript(
    "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"
  );

  const { Hands, HAND_CONNECTIONS } = window;
  const { Camera } = window;
  const { drawConnectors, drawLandmarks } = window;

  const hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });
  hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.5,
  });
  hands.onResults((results) => {
    // Check if canvas is still valid before drawing
    if (!canvasEl || !canvasEl.getContext) return;
    const ctx = canvasEl.getContext("2d");
    canvasEl.width = results.image.width;
    canvasEl.height = results.image.height;
    ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);

    const hasHands = results.multiHandLandmarks?.length > 0;
    if (hasHands) {
      for (const lm of results.multiHandLandmarks) {
        drawConnectors(ctx, lm, HAND_CONNECTIONS, {
          color: "rgba(0, 0, 0, 0.7)",
          lineWidth: 5,
        });
        drawLandmarks(ctx, lm, {
          color: "#e1e1e1",
          fillColor: "#e1e1e1",
          lineWidth: 1,
          radius: 6,
        });
      }
    }
    onResults(results, hasHands);
  });

  const camera = new Camera(videoEl, {
    onFrame: async () => {
      // Check if video element is still valid
      if (!videoEl || videoEl.readyState < 2) return;
      await hands.send({ image: videoEl });
    },
    width: 640,
    height: 480,
  });
  camera.start();
  return camera;
}

// ─── PDF.JS LOADER ────────────────────────────────────────────────────────────
async function loadPdfJs() {
  if (window.pdfjsLib) return window.pdfjsLib;
  await loadScript(
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"
  );
  window.pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
  return window.pdfjsLib;
}

// ─── ICONS ───────────────────────────────────────────────────────────────────
const ChevronLeft = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2.5"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M15 18l-6-6 6-6" />
  </svg>
);
const ChevronRight = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2.5"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M9 18l6-6-6-6" />
  </svg>
);
const CameraIcon = () => (
  <svg
    width="32"
    height="32"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
  >
    <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" />
    <circle cx="12" cy="13" r="4" />
  </svg>
);

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function SignSpeak() {
  // WebSocket
  const [wsStatus, setWsStatus] = useState("disconnected");
  const wsRef = useRef(null);

  // Camera / MediaPipe
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  //const cameraRef = useRef(null);
  const [handDetected, setHandDetected] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const currentLandmarksRef = useRef(null);
  const landmarkThrottleRef = useRef(0);

  // Recording
  const [signName, setSignName] = useState("");
  const [isRecordingStatic, setIsRecordingStatic] = useState(false);
  const [isRecordingDynamic, setIsRecordingDynamic] = useState(false);
  const [recLabel, setRecLabel] = useState("");

  // Recognition
  const [recognized, setRecognized] = useState("");
  const recognizedTimerRef = useRef(null);

  // Emotion detection
  const [currentEmotion, setCurrentEmotion] = useState({
    emotion: "neutral",
    confidence: 0,
  });

  // Captions
  const [captionWords, setCaptionWords] = useState([]);

  // Presenting / TTS
  const [isPresenting, setIsPresenting] = useState(false);

  // PDF
  const [pdfDoc, setPdfDoc] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const slideCanvasRef = useRef(null);

  // Navigation
  const [activeTab, setActiveTab] = useState("dashboard"); // "dashboard" | "calibration"

  // Modals
  const [voiceOpen, setVoiceOpen] = useState(false);
  const [signsOpen, setSignsOpen] = useState(false);
  const [templates, setTemplates] = useState({});

  // Voice settings
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState("");
  const [rate, setRate] = useState(1);
  const [pitch, setPitch] = useState(1);
  const [volume, setVolume] = useState(1);

  const [editingName, setEditingName] = useState(null); // which sign is being edited
  const [editValue, setEditValue] = useState(""); // what value is being edited

  // ── STYLES INJECT ─────────────────────────────────────────────
  useEffect(() => {
    const el = document.createElement("style");
    el.textContent = css;
    document.head.appendChild(el);
    return () => el.remove();
  }, []);

  // ── VOICES ────────────────────────────────────────────────────
  useEffect(() => {
    const load = () => {
      const v = window.speechSynthesis.getVoices();
      setVoices(v);
      if (v.length && !selectedVoice) setSelectedVoice(v[0].name);
    };
    load();
    window.speechSynthesis.onvoiceschanged = load;
  }, []);

  // ── WEBSOCKET ─────────────────────────────────────────────────
  const connectWs = useCallback(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;
    ws.onopen = () => setWsStatus("connected");
    ws.onclose = () => {
      setWsStatus("disconnected");
      setTimeout(connectWs, 2000);
    };
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      handleMessage(data);
    };
  }, []);

  useEffect(() => {
    connectWs();
  }, [connectWs]);

  // ── MESSAGE HANDLER ────────────────────────────────────────────
  const isPresentingRef = useRef(false);
  useEffect(() => {
    isPresentingRef.current = isPresenting;
  }, [isPresenting]);

  function handleMessage(data) {
    if (data.type === "match") {
      if (data.recording) {
        setRecLabel(`Recording — ${data.frame_count} frames`);
      }
      if (data.recognized && isPresentingRef.current) {
        showRecog(data.recognized);
        setCaptionWords((w) => [...w, data.recognized]);
        speak(data.recognized);
      }
    }
    if (data.type === "recording_started") {
      setIsRecordingStatic(data.gesture_type === "static" ? true : false);
      setRecLabel(
        data.gesture_type === "static"
          ? "Hold sign steady…"
          : "Recording dynamic sign…"
      );
    }
    if (data.type === "recording_done") {
      setIsRecordingStatic(false);
      setIsRecordingDynamic(false);
      setRecLabel("");
      if (data.success) {
        const msg = data.ready
          ? `✓ ${data.name} ready`
          : `${data.name}: ${data.count}/${data.needed} — record again`;
        showRecog(msg);
      }
    }
    if (data.type === "templates") setTemplates(data.templates);
    if (data.type === "deleted") sendWs({ type: "get_templates" });
    if (data.type === "renamed") {
      setTemplates(prev => {
        const next = { ...prev };
        next[data.new_name] = next[data.old_name];
        delete next[data.old_name];
        return next;
      });
      setEditingName(null);
    }
  }

  function sendWs(payload) {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    }
  }

  // ── MEDIAPIPE ─────────────────────────────────────────────────
  const cameraRef = useRef(null);
  useEffect(() => {
    const vid = videoRef.current;
    const cvs = canvasRef.current;
    if (!vid || !cvs) return;

    const handleLoadedMetadata = () => {
      setVideoReady(true);
    };

    vid.addEventListener("loadedmetadata", handleLoadedMetadata);
    if (vid.readyState >= 2) {
      setVideoReady(true);
    }

    let cameraInstance = null;
    initMP(vid, cvs, (results, hasHands) => {
      setHandDetected(hasHands);
      if (hasHands) {
        currentLandmarksRef.current = results.multiHandLandmarks;
        const now = Date.now();
        if (now - landmarkThrottleRef.current < 66) return;
        landmarkThrottleRef.current = now;
        const lmData = results.multiHandLandmarks.map((hand) =>
          hand.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z }))
        );
        sendWs({ type: "landmarks", landmarks: lmData });
      } else {
        currentLandmarksRef.current = null;
      }
    })
      .then((camera) => {
        cameraInstance = camera;
        cameraRef.current = camera;
      })
      .catch((err) => {
        console.error("MediaPipe initialization error:", err);
      });

    return () => {
      vid.removeEventListener("loadedmetadata", handleLoadedMetadata);
      // Stop the camera when component unmounts or when switching away
      if (cameraInstance) {
        try {
          cameraInstance.stop();
        } catch (e) {
          // Ignore errors when stopping
        }
        cameraRef.current = null;
      }
    };
  }, []);

  // ── RECOGNITION BADGE ─────────────────────────────────────────
  function showRecog(text) {
    setRecognized(text);
    clearTimeout(recognizedTimerRef.current);
    recognizedTimerRef.current = setTimeout(() => setRecognized(""), 1600);
  }

  // ── RECORDING ─────────────────────────────────────────────────
  function recordStatic() {
    if (!signName.trim()) return alert("Enter a sign name first.");
    if (!currentLandmarksRef.current)
      return alert("Make sure your hand is visible.");
    sendWs({ type: "start_static_recording", name: signName.trim() });
  }

  function toggleDynamic() {
    if (!isRecordingDynamic) {
      if (!signName.trim()) return alert("Enter a sign name first.");
      setIsRecordingDynamic(true);
      sendWs({ type: "start_dynamic_recording", name: signName.trim() });
    } else {
      sendWs({ type: "stop_dynamic_recording" });
    }
  }

  // ── PDF ───────────────────────────────────────────────────────
  async function handlePdfLoad(e) {
    const file = e.target.files[0];
    if (!file) return;
    const pdfjsLib = await loadPdfJs();
    const buf = await file.arrayBuffer();
    const doc = await pdfjsLib.getDocument({ data: buf }).promise;
    setPdfDoc(doc);
    setCurrentPage(1);
    renderPage(doc, 1);
  }

  async function renderPage(doc, pageNum) {
    const page = await doc.getPage(pageNum);
    const container = slideCanvasRef.current?.parentElement;
    if (!container) return;
    const scale = Math.min(
      (container.clientWidth - 48) / page.getViewport({ scale: 1 }).width,
      (container.clientHeight - 48) / page.getViewport({ scale: 1 }).height
    );
    const vp = page.getViewport({ scale });
    const cvs = slideCanvasRef.current;
    cvs.width = vp.width;
    cvs.height = vp.height;
    await page.render({ canvasContext: cvs.getContext("2d"), viewport: vp })
      .promise;
  }

  function prevSlide() {
    if (!pdfDoc || currentPage <= 1) return;
    const p = currentPage - 1;
    setCurrentPage(p);
    renderPage(pdfDoc, p);
  }

  function nextSlide() {
    if (!pdfDoc || currentPage >= pdfDoc.numPages) return;
    const p = currentPage + 1;
    setCurrentPage(p);
    renderPage(pdfDoc, p);
  }

  // ── KEYBOARD ──────────────────────────────────────────────────
  useEffect(() => {
    function onKey(e) {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.key === "ArrowRight") nextSlide();
      if (e.key === "ArrowLeft") prevSlide();
      if (e.key === " ") {
        e.preventDefault();
        setIsPresenting((p) => !p);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pdfDoc, currentPage]);

  // ── TTS ───────────────────────────────────────────────────────
  function speak(text) {
    const utt = new SpeechSynthesisUtterance(text);
    const v = window.speechSynthesis
      .getVoices()
      .find((v) => v.name === selectedVoice);
    if (v) utt.voice = v;
    utt.rate = rate;
    utt.pitch = pitch;
    utt.volume = volume;
    window.speechSynthesis.speak(utt);
  }

  // ─────────────────────────────────────────────────────────────
  const showRec = isRecordingStatic || isRecordingDynamic;

  return (
    <div className="app">
      {/* HEADER */}
      <header className="header">
        <div className="logo">
          Sign<span>Speak</span>
        </div>

        <div className="status-pill">
          <div
            className={`status-dot ${
              wsStatus === "connected" ? "connected" : ""
            }`}
          />
          {wsStatus === "connected" ? "Connected" : "Disconnected"}
        </div>

        <div className="header-right">
          {activeTab === "dashboard" && (
            <>
              <button
                className="btn btn-default"
                onClick={() => setVoiceOpen(true)}
              >
                Voice
              </button>
              <button
                className="btn btn-default"
                onClick={() => {
                  setSignsOpen(true);
                  sendWs({ type: "get_templates" });
                }}
              >
                Saved Signs
              </button>
              <label className="btn btn-primary" style={{ cursor: "pointer" }}>
                Load PDF
                <input
                  className="sr-only"
                  type="file"
                  accept=".pdf"
                  onChange={handlePdfLoad}
                />
              </label>
            </>
          )}
        </div>
      </header>

      {/* TABS */}
      <div className="tabs">
        <button
          className={`tab ${activeTab === "dashboard" ? "active" : ""}`}
          onClick={() => setActiveTab("dashboard")}
        >
          Dashboard
        </button>
        <button
          className={`tab ${activeTab === "calibration" ? "active" : ""}`}
          onClick={() => setActiveTab("calibration")}
        >
          Calibration
        </button>
      </div>

      {/* MAIN */}
      {/* Keep Dashboard always mounted but hidden when Calibration is active */}
      <div
        className="main"
        style={{ display: activeTab === "calibration" ? "none" : "grid" }}
      >
        {/* LEFT: CAMERA */}
        <div className="left-panel">
          <div className="panel-label">
            <span>Live Feed</span>
            {handDetected && (
              <div className="hand-badge">
                <div className="hand-badge-dot" />
                Hand detected
              </div>
            )}
          </div>

          <div className="camera-wrap">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              style={{ display: "none" }}
            />
            <canvas ref={canvasRef} className="camera-canvas" />

            {/* Face emotion detection - processes video in background */}
            {videoReady && videoRef.current && (
              <FacialEmotionDetection
                videoElement={videoRef.current}
                onEmotionChange={(emotionData) => {
                  setCurrentEmotion(emotionData);
                }}
              />
            )}

            {recognized && (
              <div className={`recog-badge ${recognized ? "show" : ""}`}>
                {recognized}
              </div>
            )}

            {/* Emotion display - now includes neutral */}
            {currentEmotion.emotion && currentEmotion.confidence > 0 && (
              <div
                style={{
                  position: "absolute",
                  top: "10px",
                  right: "10px",
                  background: "var(--accent-bg)",
                  border: "1px solid var(--accent-dim)",
                  borderRadius: "var(--radius-sm)",
                  padding: "6px 12px",
                  fontSize: "12px",
                  fontFamily: "'Geist Mono', monospace",
                  color: "var(--accent)",
                  fontWeight: 500,
                  boxShadow: "var(--shadow-sm)",
                }}
              >
                {currentEmotion.emotion} ({currentEmotion.confidence}%)
              </div>
            )}

            {showRec && (
              <div className="rec-badge">
                <div className="rec-dot" />
                {recLabel || "Recording"}
              </div>
            )}
          </div>

          {/* RECORD CONTROLS */}
          <div className="record-section">
            <div className="section-title">Record New Sign</div>
            <input
              className="input"
              type="text"
              placeholder="Sign name (e.g. hello, next…)"
              value={signName}
              onChange={(e) => setSignName(e.target.value)}
            />
            <div className="record-grid">
              <button className="btn btn-default" onClick={recordStatic}>
                📸 Static
              </button>
              <button
                className={`btn ${
                  isRecordingDynamic ? "btn-danger" : "btn-default"
                }`}
                onClick={toggleDynamic}
              >
                {isRecordingDynamic ? "⏹ Stop" : "⏺ Dynamic"}
              </button>
            </div>
          </div>
        </div>

        {/* RIGHT: SLIDES + CAPTIONS */}
        <div className="right-panel">
          <div className="slide-area">
            {!pdfDoc ? (
              <div className="upload-empty">
                <div className="upload-empty-icon">
                  <CameraIcon />
                </div>
                <h3>Load your presentation</h3>
                <p>
                  Upload a PDF to display slides. Use the camera to sign and
                  speak your content.
                </p>
                <label
                  className="btn btn-primary"
                  style={{ cursor: "pointer", marginTop: 4 }}
                >
                  Choose PDF
                  <input
                    className="sr-only"
                    type="file"
                    accept=".pdf"
                    onChange={handlePdfLoad}
                  />
                </label>
              </div>
            ) : (
              <canvas ref={slideCanvasRef} className="slide-canvas" />
            )}
          </div>

          {/* SLIDE NAV */}
          <div className="slide-controls">
            <button
              className="btn btn-icon"
              onClick={prevSlide}
              disabled={!pdfDoc || currentPage <= 1}
            >
              <ChevronLeft />
            </button>
            <span className="slide-count">
              {pdfDoc ? `${currentPage} / ${pdfDoc.numPages}` : "— / —"}
            </span>
            <button
              className="btn btn-icon"
              onClick={nextSlide}
              disabled={!pdfDoc || currentPage >= (pdfDoc?.numPages ?? 1)}
            >
              <ChevronRight />
            </button>
            <div className="slide-spacer" />
            <button
              className="btn btn-ghost"
              style={{ fontSize: 12 }}
              onClick={() => setCaptionWords([])}
            >
              Clear captions
            </button>
            <div className="divider" />
            <button
              className={`btn ${isPresenting ? "btn-danger" : "btn-success"}`}
              onClick={() => setIsPresenting((p) => !p)}
            >
              {isPresenting ? "Stop Presenting" : "Start Presenting"}
            </button>
          </div>

          {/* CAPTIONS */}
          <div className="captions">
            <span className="cc-label">CC</span>
            <div className="caption-text">
              {captionWords.length === 0 ? (
                <span className="caption-empty">
                  Captions will appear here as you sign…
                </span>
              ) : (
                captionWords.map((w, i) => (
                  <span key={i} className="caption-word">
                    {w}{" "}
                  </span>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* CALIBRATION TAB - Rendered separately to avoid interfering with Dashboard video */}
      {activeTab === "calibration" && (
        <div
          style={{
            flex: 1,
            overflowY: "auto",
            overflowX: "hidden",
            padding: "24px",
            maxWidth: "800px",
            margin: "0 auto",
            width: "100%",
          }}
        >
          <Calibration />
        </div>
      )}

      {/* VOICE MODAL */}
      {voiceOpen && (
        <div
          className="overlay"
          onClick={(e) => e.target === e.currentTarget && setVoiceOpen(false)}
        >
          <div className="modal">
            <div className="modal-title">Voice Settings</div>
            <div className="modal-sub">
              Adjust how SignSpeak reads your signs aloud.
            </div>

            <div className="form-group">
              <label className="form-label">Voice</label>
              <select
                className="select"
                value={selectedVoice}
                onChange={(e) => setSelectedVoice(e.target.value)}
              >
                {voices.map((v) => (
                  <option key={v.name} value={v.name}>
                    {v.name} ({v.lang})
                  </option>
                ))}
              </select>
            </div>

            {[
              {
                label: "Speed",
                val: rate,
                set: setRate,
                min: 0.5,
                max: 2,
                step: 0.1,
                fmt: (v) => `${parseFloat(v).toFixed(1)}×`,
              },
              {
                label: "Pitch",
                val: pitch,
                set: setPitch,
                min: 0.5,
                max: 2,
                step: 0.1,
                fmt: (v) => parseFloat(v).toFixed(1),
              },
              {
                label: "Volume",
                val: volume,
                set: setVolume,
                min: 0,
                max: 1,
                step: 0.05,
                fmt: (v) => `${Math.round(v * 100)}%`,
              },
            ].map(({ label, val, set, min, max, step, fmt }) => (
              <div className="form-group" key={label}>
                <label className="form-label">{label}</label>
                <div className="range-row">
                  <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={val}
                    onChange={(e) => set(parseFloat(e.target.value))}
                  />
                  <span className="range-val">{fmt(val)}</span>
                </div>
              </div>
            ))}

            <button
              className="btn btn-default"
              onClick={() =>
                speak("Hello, my name is SignSpeak. I am ready to present.")
              }
            >
              🔊 Test Voice
            </button>

            <div className="modal-footer">
              <button
                className="btn btn-primary"
                onClick={() => setVoiceOpen(false)}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      {/* SIGNS MODAL */}
      {signsOpen && (
        <div
          className="overlay"
          onClick={(e) => e.target === e.currentTarget && setSignsOpen(false)}
        >
          <div className="modal">
            <div className="modal-title">Saved Signs</div>
            <div className="modal-sub">
              All your recorded gestures and their training status.
            </div>

            <div className="template-list">
              {Object.keys(templates).length === 0 ? (
                <div className="empty-state">
                  No signs recorded yet. Use the camera panel to get started.
                </div>
              ) : (
                Object.entries(templates).map(([name, info]) => (
                  <div className="template-row" key={name}>
                    <div className="template-left">
                      <div className="template-top">
                        {editingName === name ? (
                          <input
                            className="input"
                            style={{ fontSize: 13, padding: "4px 8px", width: "auto", flex: 1 }}
                            value={editValue}
                            autoFocus
                            onChange={(e) => setEditValue(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" && editValue.trim()) {
                                sendWs({ type: "rename_template", old_name: name, new_name: editValue.trim() });
                              }
                              if (e.key === "Escape") setEditingName(null);
                            }}
                          />
                        ) : (
                          <span className="template-name">{name}</span>
                        )}
                        <span className={`tag tag-${info.type}`}>{info.type}</span>
                        <span className={`tag ${info.ready ? "tag-ready" : "tag-pending"}`}>
                          {info.ready ? "✓ ready" : "training"}
                        </span>
                      </div>
                      <TrainingProgress recordings={info.recordings} needed={info.needed} />
                    </div>
                    <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
                      {editingName === name ? (
                        <>
                          <button
                            className="btn btn-primary"
                            style={{ fontSize: 11, padding: "4px 10px" }}
                            onClick={() => {
                              if (editValue.trim()) {
                                sendWs({ type: "rename_template", old_name: name, new_name: editValue.trim() });
                              }
                            }}
                          >
                            Save
                          </button>
                          <button
                            className="btn btn-ghost"
                            style={{ fontSize: 11, padding: "4px 10px" }}
                            onClick={() => setEditingName(null)}
                          >
                            Cancel
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            className="btn btn-default"
                            style={{ fontSize: 11, padding: "4px 10px" }}
                            onClick={() => { setEditingName(name); setEditValue(name); }}
                          >
                            Rename
                          </button>
                          <button
                            className="btn btn-danger"
                            style={{ fontSize: 11, padding: "4px 10px" }}
                            onClick={() => sendWs({ type: "delete_template", name })}
                          >
                            Delete
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>

            <div className="modal-footer">
              <button
                className="btn btn-primary"
                onClick={() => setSignsOpen(false)}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
