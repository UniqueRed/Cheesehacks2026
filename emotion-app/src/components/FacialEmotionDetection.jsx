/**
 * HOW IT WORKS
 * ------------
 * 1. MediaPipe Face Landmarker runs on each video frame → 52 blendshape scores (0–1).
 * 2. Calibration: first 5 seconds we average those scores as your "neutral baseline".
 * 3. Every frame we normalize: score - baseline (so your neutral face → values near 0).
 * 4. classifyEmotion(normalized) picks one of 5 emotions from those normalized scores.
 * 5. We only update the displayed emotion after the SAME emotion wins STABILITY_FRAMES in a row.
 *
 * EMOTIONS (5):
 *   happiness, neutral, surprise, sadness, anger
 *
 * KEY BLENDSHAPES USED:
 *   mouthSmileLeft/Right     → happiness
 *   cheekSquintLeft/Right    → happiness
 *   browInnerUp              → surprise, sadness
 *   eyeWideLeft/Right        → surprise
 *   jawOpen                  → surprise
 *   browDownLeft/Right       → anger
 *   eyeSquintLeft/Right      → anger
 *   jawForward               → anger
 *   mouthFrownLeft/Right     → sadness
 */

import { useState, useRef, useEffect, useCallback } from "react";
import {
  FilesetResolver,
  FaceLandmarker,
} from "@mediapipe/tasks-vision";
import "./FacialEmotionDetection.css";

// ============ THRESHOLDS — tune these for your face ============
// All names now match the FER7 emotions used in classifyEmotion below.
// Lower values = more sensitive. Increase if jumping too much.
const THRESHOLDS = {
  // Stability: how many consecutive frames must agree before switching emotion
  STABILITY_FRAMES: 8,

  // ── HAPPINESS: cheeks up, lip corners pulled up, brows relaxed ──
  HAPPINESS_SMILE_MIN: 0.25,        // mouthSmile average must exceed this
  HAPPINESS_CHEEK_SQUINT_MIN: 0.15, // cheekSquint average must exceed this
  HAPPINESS_BROW_DOWN_MAX: 0.20,    // browDown must stay below this (not angry)

  // ── NEUTRAL: very little movement across all blendshapes ──
  NEUTRAL_MAX_VARIANCE: 0.018,      // overall blendshape variance must be low
  NEUTRAL_SMILE_MAX: 0.20,          // not smiling
  NEUTRAL_BROW_DOWN_MAX: 0.25,      // not frowning
  NEUTRAL_JAW_OPEN_MAX: 0.15,       // mouth mostly closed

  // ── SURPRISE: brows shoot up, eyes go wide, jaw drops ──
  SURPRISE_BROW_INNER_MIN: 0.30,    // browInnerUp high
  SURPRISE_EYE_WIDE_MIN: 0.25,      // eyeWide average high
  SURPRISE_JAW_OPEN_MIN: 0.20,      // jaw noticeably open

  // ── SADNESS: inner brows pull up/together, lip corners down ──
  SADNESS_BROW_INNER_MIN: 0.20,     // browInnerUp elevated
  SADNESS_MOUTH_FROWN_MIN: 0.10,    // mouthFrown average elevated
  SADNESS_SMILE_MAX: 0.15,          // not smiling
  SADNESS_EYE_SQUINT_MAX: 0.15,     // eyes not squinting (not angry)

  // ── ANGER: brows pulled hard down, eyes squint, jaw pushes forward ──
  ANGER_BROW_DOWN_MIN: 0.30,        // browDown average high
  ANGER_EYE_SQUINT_MIN: 0.20,       // eyeSquint average elevated
  ANGER_SMILE_MAX: 0.15,            // not smiling
  ANGER_JAW_FORWARD_MIN: 0.05,      // jaw slightly forward
};

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const CALIBRATION_DURATION_MS = 5000;
const EMOTION_CALIBRATION_DURATION_MS = 3000;
const CALIBRATION_REST_DURATION_MS = 1500;
const ROUNDS_PER_EMOTION = 3;

// 5 emotion names — used throughout, must stay consistent
const EMOTION_NAMES = [
  "happiness",
  "neutral",
  "surprise",
  "sadness",
  "anger",
];

/** Short instruction shown next to each calibration button */
const EMOTION_INSTRUCTIONS = {
  happiness: "genuine smile, cheeks up",
  neutral: "relaxed natural resting face",
  surprise: "eyebrows up, eyes wide, mouth open",
  sadness: "inner brows raised, corners of mouth pulled down",
  anger: "brows pulled hard down, eyes narrowed",
};

// ——— Helpers ———
function getScore(blendshapes, name) {
  if (!blendshapes || blendshapes[name] == null) return 0;
  return blendshapes[name];
}

function variance(values) {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / values.length;
}

/**
 * Given averaged normalized blendshapes recorded during threshold calibration
 * for one emotion, compute tight threshold updates for that emotion.
 */
function thresholdsFromAverages(emotion, averages) {
  const pad = 0.05;
  const padLo = (x) => Math.max(0, x - pad);
  const padHi = (x) => Math.min(1, x + pad);

  const mouthSmile =
    ((averages.mouthSmileLeft ?? 0) + (averages.mouthSmileRight ?? 0)) / 2;
  const browInnerUp = averages.browInnerUp ?? 0;
  const browDown =
    ((averages.browDownLeft ?? 0) + (averages.browDownRight ?? 0)) / 2;
  const eyeWide =
    ((averages.eyeWideLeft ?? 0) + (averages.eyeWideRight ?? 0)) / 2;
  const eyeSquint =
    ((averages.eyeSquintLeft ?? 0) + (averages.eyeSquintRight ?? 0)) / 2;
  const cheekSquint =
    ((averages.cheekSquintLeft ?? 0) + (averages.cheekSquintRight ?? 0)) / 2;
  const jawOpen = averages.jawOpen ?? 0;
  const jawForward = averages.jawForward ?? 0;
  const mouthFrown =
    ((averages.mouthFrownLeft ?? 0) + (averages.mouthFrownRight ?? 0)) / 2;
  const v = variance(Object.values(averages));

  switch (emotion) {
    case "happiness":
      return {
        HAPPINESS_SMILE_MIN: padLo(mouthSmile),
        HAPPINESS_CHEEK_SQUINT_MIN: padLo(cheekSquint),
        HAPPINESS_BROW_DOWN_MAX: padHi(browDown),
      };
    case "neutral":
      return {
        NEUTRAL_MAX_VARIANCE: Math.min(0.05, v * 1.5),
        NEUTRAL_SMILE_MAX: padHi(mouthSmile),
        NEUTRAL_BROW_DOWN_MAX: padHi(browDown),
        NEUTRAL_JAW_OPEN_MAX: padHi(jawOpen),
      };
    case "surprise":
      return {
        SURPRISE_BROW_INNER_MIN: padLo(browInnerUp),
        SURPRISE_EYE_WIDE_MIN: padLo(eyeWide),
        SURPRISE_JAW_OPEN_MIN: padLo(jawOpen),
      };
    case "sadness":
      return {
        SADNESS_BROW_INNER_MIN: padLo(browInnerUp),
        SADNESS_MOUTH_FROWN_MIN: padLo(mouthFrown),
        SADNESS_SMILE_MAX: padHi(mouthSmile),
        SADNESS_EYE_SQUINT_MAX: padHi(eyeSquint),
      };
    case "anger":
      return {
        ANGER_BROW_DOWN_MIN: padLo(browDown),
        ANGER_EYE_SQUINT_MIN: padLo(eyeSquint),
        ANGER_SMILE_MAX: padHi(mouthSmile),
        ANGER_JAW_FORWARD_MIN: padLo(jawForward),
      };
    default:
      return {};
  }
}

/**
 * Classify one of the 5 emotions from normalized blendshape scores.
 * Uses a weighted scoring system so the best match always wins.
 *
 * @param {Object} blendshapes - normalized blendshape scores (score - baseline)
 * @param {Object} [thresholds] - optional override (e.g. from per-face calibration)
 * @returns {string} one of: happiness | neutral | surprise | sadness | anger
 */
export function classifyEmotion(blendshapes, thresholds) {
  if (!blendshapes || typeof blendshapes !== "object") return "neutral";

  const t = thresholds || THRESHOLDS;
  const g = (name) => getScore(blendshapes, name);
  const values = Object.values(blendshapes);
  const v = variance(values);

  // ── Aggregate bilateral blendshapes ──
  const mouthSmile   = (g("mouthSmileLeft")    + g("mouthSmileRight"))    / 2;
  const cheekSquint  = (g("cheekSquintLeft")   + g("cheekSquintRight"))   / 2;
  const browDown     = (g("browDownLeft")      + g("browDownRight"))      / 2;
  const eyeWide      = (g("eyeWideLeft")       + g("eyeWideRight"))       / 2;
  const eyeSquint    = (g("eyeSquintLeft")     + g("eyeSquintRight"))     / 2;
  const mouthFrown   = (g("mouthFrownLeft")    + g("mouthFrownRight"))    / 2;
  const browInnerUp  = g("browInnerUp");
  const jawOpen      = g("jawOpen");
  const jawForward   = g("jawForward");

  // ── Score each emotion (0.0 – 1.0) ──

  // HAPPINESS: smile is the dominant signal, cheek squint confirms it
  const scoreHappiness =
    (mouthSmile  >= t.HAPPINESS_SMILE_MIN        ? 1 : 0) * 0.50 +
    (cheekSquint >= t.HAPPINESS_CHEEK_SQUINT_MIN  ? 1 : 0) * 0.30 +
    (browDown    <= t.HAPPINESS_BROW_DOWN_MAX     ? 1 : 0) * 0.20;

  // NEUTRAL: low variance across everything is the key signal
  const scoreNeutral =
    (v          <= t.NEUTRAL_MAX_VARIANCE   ? 1 : 0) * 0.40 +
    (mouthSmile <= t.NEUTRAL_SMILE_MAX      ? 1 : 0) * 0.20 +
    (browDown   <= t.NEUTRAL_BROW_DOWN_MAX  ? 1 : 0) * 0.20 +
    (jawOpen    <= t.NEUTRAL_JAW_OPEN_MAX   ? 1 : 0) * 0.20;

  // SURPRISE: all three signals (brow, eye, jaw) must fire together
  const scoreSurprise =
    (browInnerUp >= t.SURPRISE_BROW_INNER_MIN ? 1 : 0) * 0.35 +
    (eyeWide     >= t.SURPRISE_EYE_WIDE_MIN   ? 1 : 0) * 0.35 +
    (jawOpen     >= t.SURPRISE_JAW_OPEN_MIN   ? 1 : 0) * 0.30;

  // SADNESS: inner brow raise + lip corners down + no smile
  const scoreSadness =
    (browInnerUp >= t.SADNESS_BROW_INNER_MIN       ? 1 : 0) * 0.35 +
    (mouthFrown  >= t.SADNESS_MOUTH_FROWN_MIN       ? 1 : 0) * 0.35 +
    (mouthSmile  <= t.SADNESS_SMILE_MAX             ? 1 : 0) * 0.15 +
    (eyeSquint   <= t.SADNESS_EYE_SQUINT_MAX        ? 1 : 0) * 0.15;

  // ANGER: heavy brow furrow + eye squint are the strongest signals
  const scoreAnger =
    (browDown    >= t.ANGER_BROW_DOWN_MIN    ? 1 : 0) * 0.40 +
    (eyeSquint   >= t.ANGER_EYE_SQUINT_MIN   ? 1 : 0) * 0.30 +
    (mouthSmile  <= t.ANGER_SMILE_MAX        ? 1 : 0) * 0.15 +
    (jawForward  >= t.ANGER_JAW_FORWARD_MIN  ? 1 : 0) * 0.15;

  const scores = [
    { emotion: "happiness", score: scoreHappiness },
    { emotion: "neutral",   score: scoreNeutral   },
    { emotion: "surprise",  score: scoreSurprise  },
    { emotion: "sadness",   score: scoreSadness   },
    { emotion: "anger",     score: scoreAnger     },
  ];

  const best = scores.reduce((a, b) => (b.score > a.score ? b : a));
  if (best.score >= 0.25) return best.emotion;
  return "neutral";
}

// ——— Main Component ———
export function FacialEmotionDetection({ onEmotionChange }) {
  const videoRef       = useRef(null);
  const streamRef      = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const animationRef   = useRef(null);
  const lastVideoTimeRef = useRef(-1);

  const [phase, setPhase]               = useState("loading"); // loading | calibrating | running
  const [error, setError]               = useState(null);
  const [currentEmotion, setCurrentEmotion] = useState("neutral");
  const [confidence, setConfidence]     = useState(0);
  const [emotionLog, setEmotionLog]     = useState([]);
  const [debugOpen, setDebugOpen]       = useState(false);
  const [blendshapesDebug, setBlendshapesDebug] = useState({});

  const baselineRef              = useRef(null);
  const sameEmotionCountRef      = useRef(0);
  const lastDetectedEmotionRef   = useRef("neutral");

  // Per-face threshold calibration state (initial from localStorage if present)
  const [calibratedThresholds, setCalibratedThresholds] = useState(() => {
    try {
      const s = localStorage.getItem("facialEmotionThresholds");
      if (s) {
        const parsed = JSON.parse(s);
        if (parsed && typeof parsed === "object")
          return { ...THRESHOLDS, ...parsed };
      }
    } catch (_) {}
    return { ...THRESHOLDS };
  });
  const [usingSavedCalibration, setUsingSavedCalibration] = useState(() => {
    try {
      return !!localStorage.getItem("facialEmotionThresholds");
    } catch (_) {
      return false;
    }
  });
  const [calibratingEmotion, setCalibratingEmotion]     = useState(null);
  const [calibrationPhase, setCalibrationPhase]         = useState(null); // 'record' | 'rest'
  const [calibrationRound, setCalibrationRound]         = useState(1);
  const [calibratedEmotions, setCalibratedEmotions]     = useState(new Set());
  const calibrationStartTimeRef  = useRef(null);
  const calibrationRestStartRef  = useRef(null);
  const calibrationSamplesRef    = useRef([]);
  const [calibrationRemainingSec, setCalibrationRemainingSec] = useState(3);
  const [calibrationProgress, setCalibrationProgress]   = useState(0);

  // Countdown + progress during per-face threshold calibration (record or rest)
  useEffect(() => {
    if (!calibratingEmotion) return;
    const interval = setInterval(() => {
      if (calibrationPhase === "record") {
        const elapsed = Date.now() - calibrationStartTimeRef.current;
        const remaining = Math.max(
          0,
          (EMOTION_CALIBRATION_DURATION_MS - elapsed) / 1000
        );
        setCalibrationRemainingSec(remaining);
        setCalibrationProgress(Math.min(1, elapsed / EMOTION_CALIBRATION_DURATION_MS));
      } else if (calibrationPhase === "rest") {
        const restElapsed = Date.now() - calibrationRestStartRef.current;
        const remaining = Math.max(
          0,
          (CALIBRATION_REST_DURATION_MS - restElapsed) / 1000
        );
        setCalibrationRemainingSec(remaining);
        setCalibrationProgress(0);
      }
    }, 100);
    return () => clearInterval(interval);
  }, [calibratingEmotion, calibrationPhase]);

  // Notify parent component when emotion changes
  useEffect(() => {
    if (onEmotionChange && phase === "running") {
      onEmotionChange({ emotion: currentEmotion, confidence });
    }
  }, [currentEmotion, confidence, phase, onEmotionChange]);

  // ── Step 1: Initialize MediaPipe + webcam ──
  useEffect(() => {
    let cancelled = false;
    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL },
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 1,
        });
        if (cancelled) return;
        faceLandmarkerRef.current = faceLandmarker;

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
        });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        setPhase("calibrating");
      } catch (e) {
        if (!cancelled) setError(e.message || "Failed to initialize camera");
      }
    }
    init();
    return () => {
      cancelled = true;
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // ── Step 2: Baseline calibration (5 seconds, neutral face) ──
  useEffect(() => {
    if (phase !== "calibrating" || !faceLandmarkerRef.current || !videoRef.current)
      return;
    const video = videoRef.current;
    const samples = [];
    const numSamples = 50;
    const intervalMs = CALIBRATION_DURATION_MS / numSamples;
    let count = 0;

    const timer = setInterval(() => {
      if (video.readyState < 2) return;
      const result = faceLandmarkerRef.current.detectForVideo(video, performance.now());
      if (result.faceBlendshapes?.[0]) {
        const map = {};
        result.faceBlendshapes[0].categories.forEach((c) => {
          map[c.categoryName] = c.score;
        });
        samples.push(map);
      }
      if (++count >= numSamples) {
        clearInterval(timer);
        const baseline = {};
        const keys = samples.length > 0 ? Object.keys(samples[0]) : [];
        keys.forEach((key) => {
          baseline[key] =
            samples.reduce((s, sample) => s + (sample[key] ?? 0), 0) /
            samples.length;
        });
        baselineRef.current = baseline;
        setPhase("running");
      }
    }, intervalMs);

    return () => clearInterval(timer);
  }, [phase]);

  // ── Step 3: Main detection loop ──
  const runDetection = useCallback(() => {
    const video = videoRef.current;
    const faceLandmarker = faceLandmarkerRef.current;

    if (phase !== "running" || !video || !faceLandmarker || video.readyState < 2) {
      animationRef.current = requestAnimationFrame(runDetection);
      return;
    }

    const now = performance.now();
    if (now - lastVideoTimeRef.current < 1000 / 30) {
      animationRef.current = requestAnimationFrame(runDetection);
      return;
    }
    lastVideoTimeRef.current = now;

    const result = faceLandmarker.detectForVideo(video, now);
    const baseline = baselineRef.current || {};

    if (!result.faceBlendshapes?.[0]) {
      setBlendshapesDebug({});
      animationRef.current = requestAnimationFrame(runDetection);
      return;
    }

    const raw = {};
    const normalized = {};
    result.faceBlendshapes[0].categories.forEach((c) => {
      raw[c.categoryName] = c.score;
      normalized[c.categoryName] = Math.max(0, c.score - (baseline[c.categoryName] ?? 0));
    });
    setBlendshapesDebug(raw);

    if (import.meta.env.DEV) {
      console.log("blendshapes (raw)", raw);
    }

    // ── Per-face threshold calibration: 3 rounds of record, rest between ──
    if (calibratingEmotion) {
      if (calibrationPhase === "record") {
        calibrationSamplesRef.current.push({ ...normalized });
        const elapsed = Date.now() - calibrationStartTimeRef.current;
        if (elapsed >= EMOTION_CALIBRATION_DURATION_MS) {
          if (calibrationRound < ROUNDS_PER_EMOTION) {
            setCalibrationPhase("rest");
            calibrationRestStartRef.current = Date.now();
          } else {
            // Round 3 just finished — pool all samples from all 3 rounds and average
            const samples = calibrationSamplesRef.current;
            if (samples.length > 0) {
              const keys = Object.keys(samples[0]);
              const averages = {};
              keys.forEach((key) => {
                averages[key] =
                  samples.reduce((s, sample) => s + (sample[key] ?? 0), 0) /
                  samples.length;
              });
              const updates = thresholdsFromAverages(calibratingEmotion, averages);
              const next = { ...calibratedThresholds, ...updates };
              setCalibratedThresholds(next);
              try {
                localStorage.setItem("facialEmotionThresholds", JSON.stringify(next));
              } catch (_) {}
              setCalibratedEmotions((prev) => new Set([...prev, calibratingEmotion]));
            }
            setCalibratingEmotion(null);
            setCalibrationPhase(null);
            setCalibrationRound(1);
            calibrationSamplesRef.current = [];
          }
        }
      } else if (calibrationPhase === "rest") {
        const restElapsed = Date.now() - calibrationRestStartRef.current;
        if (restElapsed >= CALIBRATION_REST_DURATION_MS) {
          setCalibrationRound((r) => r + 1);
          setCalibrationPhase("record");
          calibrationStartTimeRef.current = Date.now();
        }
      }
    }

    // ── Classify + stability check ──
    const emotion = classifyEmotion(normalized, calibratedThresholds);

    if (emotion === lastDetectedEmotionRef.current) {
      sameEmotionCountRef.current += 1;
      if (sameEmotionCountRef.current >= calibratedThresholds.STABILITY_FRAMES) {
        setCurrentEmotion(emotion);
        const conf = Math.min(100, Math.round(50 + sameEmotionCountRef.current * 5));
        setConfidence(conf);
        setEmotionLog((prev) =>
          [
            ...prev,
            { emotion, confidence: conf, time: new Date().toLocaleTimeString() },
          ].slice(-5)
        );
      }
    } else {
      lastDetectedEmotionRef.current = emotion;
      sameEmotionCountRef.current = 1;
    }

    animationRef.current = requestAnimationFrame(runDetection);
  }, [phase, calibratingEmotion, calibrationPhase, calibrationRound, calibratedThresholds]);

  useEffect(() => {
    animationRef.current = requestAnimationFrame(runDetection);
    return () => { if (animationRef.current) cancelAnimationFrame(animationRef.current); };
  }, [runDetection]);

  // ── Error state ──
  if (error) {
    return (
      <div className="facial-emotion-error">
        <p>Error: {error}</p>
        <p>Make sure you allow camera access and use Chrome on localhost.</p>
      </div>
    );
  }

  // ── Render ──
  return (
    <div className="facial-emotion">

      {/* Camera feed + overlays */}
      <div className="facial-emotion-video-wrap">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="facial-emotion-video"
        />

        {phase === "calibrating" && (
          <div className="facial-emotion-overlay calibrating">
            Look at the camera with a neutral face for 5 seconds…
          </div>
        )}

        {phase === "running" && (
          <>
            <div className="facial-emotion-label">{currentEmotion}</div>
            <div className="facial-emotion-confidence">
              Confidence: {confidence}%
            </div>
            {calibratingEmotion && (
              <div className="facial-emotion-overlay calibrating-threshold">
                {calibrationPhase === "rest" ? (
                  <>
                    Relax your face…
                    <span className="facial-emotion-calibration-remaining">
                      {calibrationRemainingSec.toFixed(1)}s until next round
                    </span>
                  </>
                ) : (
                  <>
                    Round {calibrationRound} of {ROUNDS_PER_EMOTION} — hold your &quot;{calibratingEmotion}&quot; expression naturally, slightly looking up as if presenting
                    <div className="facial-emotion-calibration-progress-wrap">
                      <div
                        className="facial-emotion-calibration-progress-bar"
                        style={{ width: `${calibrationProgress * 100}%` }}
                      />
                    </div>
                    <span className="facial-emotion-calibration-remaining">
                      {calibrationRemainingSec.toFixed(1)}s remaining
                    </span>
                  </>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* Emotion history log */}
      {phase === "running" && (
        <div className="facial-emotion-log">
          <strong>Last 5 emotions</strong>
          <ul>
            {emotionLog
              .slice()
              .reverse()
              .map((entry, i) => (
                <li key={i}>
                  {entry.time} — {entry.emotion} ({entry.confidence}%)
                </li>
              ))}
          </ul>
        </div>
      )}

      {/* Per-face threshold calibration panel */}
      {phase === "running" && (
        <div className="facial-emotion-calibration">
          {usingSavedCalibration && (
            <div className="facial-emotion-saved-banner">
              Using your saved face calibration
            </div>
          )}
          <strong>Per-face calibration (optional but recommended)</strong>
          <p className="facial-emotion-calibration-tip">
            Tip: look slightly up as if presenting, not straight at the camera. Make each expression naturally — slight variation between rounds is good.
          </p>
          <p className="facial-emotion-calibration-hint">
            Each emotion records 3 rounds (3s each) with a short rest between. Hold the expression, then click the button.
          </p>
          <div className="facial-emotion-calibration-buttons">
            {EMOTION_NAMES.map((emotion) => (
              <button
                key={emotion}
                type="button"
                className="facial-emotion-calibration-btn"
                disabled={!!calibratingEmotion}
                onClick={() => {
                  calibrationStartTimeRef.current = Date.now();
                  calibrationSamplesRef.current = [];
                  setCalibrationRemainingSec(3);
                  setCalibrationProgress(0);
                  setCalibrationPhase("record");
                  setCalibrationRound(1);
                  setCalibratingEmotion(emotion);
                }}
              >
                <span className="facial-emotion-calibration-btn-label">
                  Calibrate: {emotion}
                  {calibratedEmotions.has(emotion) && " ✓"}
                </span>
                <span className="facial-emotion-calibration-btn-hint">
                  {EMOTION_INSTRUCTIONS[emotion]}
                </span>
              </button>
            ))}
          </div>

          <button
            type="button"
            className="facial-emotion-reset-btn"
            onClick={() => {
              try {
                localStorage.removeItem("facialEmotionThresholds");
              } catch (_) {}
              setCalibratedThresholds({ ...THRESHOLDS });
              setCalibratedEmotions(new Set());
              setUsingSavedCalibration(false);
            }}
          >
            Reset to defaults
          </button>

          {calibratedEmotions.size === EMOTION_NAMES.length && (
            <button
              type="button"
              className="facial-emotion-export-btn"
              onClick={() => {
                const str = JSON.stringify(calibratedThresholds, null, 2);
                console.log("═══ Paste the block below into THRESHOLDS in FacialEmotionDetection.jsx ═══");
                console.log(`const THRESHOLDS = ${str};`);
                alert("Thresholds exported to console (F12 → Console).");
              }}
            >
              Export Thresholds to Console
            </button>
          )}
        </div>
      )}

      {/* Debug panel — shows all 52 raw blendshape values */}
      <div className="facial-emotion-debug">
        <button
          type="button"
          className="facial-emotion-debug-toggle"
          onClick={() => setDebugOpen((o) => !o)}
        >
          {debugOpen ? "Hide" : "Show"} debug panel (52 blendshapes)
        </button>
        {debugOpen && (
          <div className="facial-emotion-debug-panel">
            {Object.entries(blendshapesDebug)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([name, score]) => (
                <div key={name} className="facial-emotion-debug-row">
                  <span className="facial-emotion-debug-name">{name}</span>
                  <div className="facial-emotion-debug-bar-wrap">
                    <div
                      className="facial-emotion-debug-bar"
                      style={{ width: `${Math.round(score * 100)}%` }}
                    />
                  </div>
                  <span className="facial-emotion-debug-value">
                    {score.toFixed(3)}
                  </span>
                </div>
              ))}
          </div>
        )}
      </div>

    </div>
  );
}

export default FacialEmotionDetection;