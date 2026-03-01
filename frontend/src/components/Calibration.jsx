/**
 * Calibration Component
 * Handles facial emotion calibration with 3-round flow per emotion
 */

import { useState, useRef, useEffect, useCallback } from "react";
import {
  FilesetResolver,
  FaceLandmarker,
} from "@mediapipe/tasks-vision";
import { classifyEmotion, THRESHOLDS, EMOTION_NAMES, EMOTION_INSTRUCTIONS } from "./FacialEmotionDetection";
import "./FacialEmotionDetection.css";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const CALIBRATION_DURATION_MS = 5000;
const EMOTION_CALIBRATION_DURATION_MS = 3000;
const CALIBRATION_REST_DURATION_MS = 1500;
const ROUNDS_PER_EMOTION = 3;

// Helper function to compute thresholds from averages
function thresholdsFromAverages(emotion, averages) {
  const pad = 0.05;
  const padLo = (x) => Math.max(0, x - pad);
  const padHi = (x) => Math.min(1, x + pad);

  const variance = (values) => {
    if (values.length === 0) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / values.length;
  };

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

export default function Calibration() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const animationRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);

  const [phase, setPhase] = useState("loading"); // loading | calibrating | running
  const [error, setError] = useState(null);
  const [currentEmotion, setCurrentEmotion] = useState("neutral");
  const [confidence, setConfidence] = useState(0);
  const [debugOpen, setDebugOpen] = useState(false);
  const [blendshapesDebug, setBlendshapesDebug] = useState({});

  const baselineRef = useRef(null);
  const sameEmotionCountRef = useRef(0);
  const lastDetectedEmotionRef = useRef("neutral");

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
  const [calibratingEmotion, setCalibratingEmotion] = useState(null);
  const [calibrationPhase, setCalibrationPhase] = useState(null); // 'record' | 'rest'
  const [calibrationRound, setCalibrationRound] = useState(1);
  const [calibratedEmotions, setCalibratedEmotions] = useState(new Set());
  const calibrationStartTimeRef = useRef(null);
  const calibrationRestStartRef = useRef(null);
  const calibrationSamplesRef = useRef([]);
  const [calibrationRemainingSec, setCalibrationRemainingSec] = useState(3);
  const [calibrationProgress, setCalibrationProgress] = useState(0);

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

  // ── Step 1: Initialize MediaPipe + webcam ──
  useEffect(() => {
    let cancelled = false;
    let currentStream = null;
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
        if (cancelled) { 
          stream.getTracks().forEach((t) => t.stop()); 
          return; 
        }
        currentStream = stream;
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setPhase("calibrating");
      } catch (e) {
        if (!cancelled) setError(e.message || "Failed to initialize camera");
      }
    }
    init();
    return () => {
      cancelled = true;
      // Only stop tracks from our own stream, not any shared streams
      if (currentStream) {
        currentStream.getTracks().forEach((t) => {
          if (t.readyState === 'live') {
            t.stop();
          }
        });
      }
      // Clear video srcObject to release the stream reference
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      // Stop animation frame loop
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
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
              setUsingSavedCalibration(true);
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
