/**
 * Calibration Component
 * Handles facial emotion calibration with 3-round flow per emotion
 */

import { useState, useRef, useEffect } from "react";
import {
  THRESHOLDS,
  EMOTION_NAMES,
  EMOTION_INSTRUCTIONS,
} from "./FacialEmotionDetection";
import { useFacialEmotion } from "../context/FacialEmotionContext";
import "./FacialEmotionDetection.css";

const EMOTION_CALIBRATION_DURATION_MS = 3000;
const CALIBRATION_REST_DURATION_MS = 1500;
const ROUNDS_PER_EMOTION = 3;
const CALIBRATED_SET_KEY = "facialEmotionCalibratedSet";

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
  const animationRef = useRef(null);
  const normalizedRef = useRef({});
  const {
    stream,
    currentEmotion,
    confidence,
    blendshapesDebug,
    normalizedBlendshapes,
    calibratedThresholds,
    isRunning,
    updateThresholds,
  } = useFacialEmotion();

  const [debugOpen, setDebugOpen] = useState(false);
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
  const [calibratedEmotions, setCalibratedEmotions] = useState(() => {
    try {
      const raw = localStorage.getItem(CALIBRATED_SET_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return new Set(Array.isArray(parsed) ? parsed : []);
    } catch (_) {
      return new Set();
    }
  });
  const calibrationStartTimeRef = useRef(null);
  const calibrationRestStartRef = useRef(null);
  const calibrationSamplesRef = useRef([]);
  const [calibrationRemainingSec, setCalibrationRemainingSec] = useState(3);
  const [calibrationProgress, setCalibrationProgress] = useState(0);

  useEffect(() => {
    normalizedRef.current = normalizedBlendshapes || {};
  }, [normalizedBlendshapes]);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.srcObject = stream || null;
      if (stream) {
        videoRef.current.play().catch(() => {});
      }
    }
  }, [stream]);

  useEffect(() => {
    if (!calibratingEmotion) return;
    const interval = setInterval(() => {
      if (calibrationPhase === "record") {
        const elapsed = Date.now() - calibrationStartTimeRef.current;
        const remaining = Math.max(
          0,
          (EMOTION_CALIBRATION_DURATION_MS - elapsed) / 1000,
        );
        setCalibrationRemainingSec(remaining);
        setCalibrationProgress(
          Math.min(1, elapsed / EMOTION_CALIBRATION_DURATION_MS),
        );
      } else if (calibrationPhase === "rest") {
        const restElapsed = Date.now() - calibrationRestStartRef.current;
        const remaining = Math.max(
          0,
          (CALIBRATION_REST_DURATION_MS - restElapsed) / 1000,
        );
        setCalibrationRemainingSec(remaining);
        setCalibrationProgress(0);
      }
    }, 100);
    return () => clearInterval(interval);
  }, [calibratingEmotion, calibrationPhase]);

  useEffect(() => {
    try {
      localStorage.setItem(
        CALIBRATED_SET_KEY,
        JSON.stringify(Array.from(calibratedEmotions)),
      );
    } catch (_) {}
  }, [calibratedEmotions]);

  useEffect(() => {
    if (!calibratingEmotion || !isRunning) return;
    let cancelled = false;

    const loop = () => {
      if (cancelled) return;
      const normalized = normalizedRef.current || {};

      if (calibrationPhase === "record" && Object.keys(normalized).length > 0) {
        calibrationSamplesRef.current.push({ ...normalized });
        const elapsed = Date.now() - calibrationStartTimeRef.current;
        if (elapsed >= EMOTION_CALIBRATION_DURATION_MS) {
          if (calibrationRound < ROUNDS_PER_EMOTION) {
            setCalibrationPhase("rest");
            calibrationRestStartRef.current = Date.now();
          } else {
            const samples = calibrationSamplesRef.current;
            if (samples.length > 0) {
              const keys = Object.keys(samples[0]);
              const averages = {};
              keys.forEach((key) => {
                averages[key] =
                  samples.reduce((s, sample) => s + (sample[key] ?? 0), 0) /
                  samples.length;
              });
              const updates = thresholdsFromAverages(
                calibratingEmotion,
                averages,
              );
              const next = { ...calibratedThresholds, ...updates };
              updateThresholds(next);
              setCalibratedEmotions(
                (prev) => new Set([...prev, calibratingEmotion]),
              );
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

      animationRef.current = requestAnimationFrame(loop);
    };

    animationRef.current = requestAnimationFrame(loop);
    return () => {
      cancelled = true;
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [
    calibratingEmotion,
    calibrationPhase,
    calibrationRound,
    calibratedThresholds,
    isRunning,
    updateThresholds,
  ]);

  const phase = isRunning ? "running" : "calibrating";

  return (
    <div className="facial-emotion">
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
                    Round {calibrationRound} of {ROUNDS_PER_EMOTION} — hold your
                    &quot;{calibratingEmotion}&quot; expression naturally,
                    slightly looking up as if presenting
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

      {phase === "running" && (
        <div className="facial-emotion-calibration">
          {usingSavedCalibration && (
            <div className="facial-emotion-saved-banner">
              Using your saved face calibration
            </div>
          )}
          <strong>Per-face calibration (optional but recommended)</strong>
          <div className="card" style={{ marginBottom: 10 }}>
            <span className="badge badge-ready">
              {calibratedEmotions.size} of {EMOTION_NAMES.length} emotions
              calibrated
            </span>
          </div>
          <p className="facial-emotion-calibration-tip">
            Tip: look slightly up as if presenting, not straight at the camera.
            Make each expression naturally — slight variation between rounds is
            good.
          </p>
          <div className="calibration-minimal">
            {EMOTION_NAMES.map((emotion) => {
              const isDone = calibratedEmotions.has(emotion);
              return (
                <div key={emotion} className="cal-row">
                  <div>
                    <strong>{emotion}</strong>
                    <small>{EMOTION_INSTRUCTIONS[emotion]}</small>
                  </div>
                  <span
                    className={`badge ${isDone ? "badge-ready" : "badge-draft"}`}
                  >
                    {isDone ? "Calibrated" : "Not calibrated"}
                  </span>
                  <button
                    type="button"
                    className="btn btn-default"
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
                    {isDone ? "Recalibrate" : "Calibrate"}
                  </button>
                </div>
              );
            })}
          </div>

          <button
            type="button"
            className="facial-emotion-reset-btn"
            onClick={() => {
              try {
                localStorage.removeItem("facialEmotionThresholds");
                localStorage.removeItem(CALIBRATED_SET_KEY);
              } catch (_) {}
              updateThresholds({ ...THRESHOLDS });
              setCalibratedEmotions(new Set());
              setUsingSavedCalibration(false);
            }}
          >
            Reset to defaults
          </button>
        </div>
      )}

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
