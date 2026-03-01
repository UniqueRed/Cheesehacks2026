import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { FilesetResolver, FaceLandmarker } from "@mediapipe/tasks-vision";
import { classifyEmotion, THRESHOLDS } from "../components/FacialEmotionDetection";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const CALIBRATION_DURATION_MS = 5000;

let sharedStreamPromise = null;
let sharedStream = null;

async function getSharedStream() {
  if (sharedStream && sharedStream.getTracks().some((t) => t.readyState === "live")) {
    return sharedStream;
  }
  if (!sharedStreamPromise) {
    sharedStreamPromise = navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } })
      .then((stream) => {
        sharedStream = stream;
        return stream;
      })
      .finally(() => {
        sharedStreamPromise = null;
      });
  }
  return sharedStreamPromise;
}

const FacialEmotionContext = createContext(null);

export function FacialEmotionProvider({ children }) {
  const [stream, setStream] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState("neutral");
  const [detectedEmotion, setDetectedEmotion] = useState("neutral"); // Most recent detected emotion (not waiting for stability)
  const [confidence, setConfidence] = useState(0);
  const [blendshapesDebug, setBlendshapesDebug] = useState({});
  const [normalizedBlendshapes, setNormalizedBlendshapes] = useState({});
  const [calibratedThresholds, setCalibratedThresholds] = useState(() => {
    try {
      const raw = localStorage.getItem("facialEmotionThresholds");
      const parsed = raw ? JSON.parse(raw) : null;
      if (parsed && typeof parsed === "object") return { ...THRESHOLDS, ...parsed };
    } catch (_) {}
    return { ...THRESHOLDS };
  });

  const faceLandmarkerRef = useRef(null);
  const processingVideoRef = useRef(null);
  const animationRef = useRef(null);
  const thresholdsRef = useRef(calibratedThresholds);
  const baselineRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const sameEmotionCountRef = useRef(0);
  const lastDetectedEmotionRef = useRef("neutral");

  const updateThresholds = useCallback((nextThresholds) => {
    const merged = { ...THRESHOLDS, ...nextThresholds };
    setCalibratedThresholds(merged);
    try {
      localStorage.setItem("facialEmotionThresholds", JSON.stringify(merged));
    } catch (_) {}
  }, []);

  useEffect(() => {
    thresholdsRef.current = calibratedThresholds;
  }, [calibratedThresholds]);

  useEffect(() => {
    const onStorage = (e) => {
      if (e.key !== "facialEmotionThresholds") return;
      try {
        const parsed = e.newValue ? JSON.parse(e.newValue) : null;
        setCalibratedThresholds(parsed && typeof parsed === "object" ? { ...THRESHOLDS, ...parsed } : { ...THRESHOLDS });
      } catch (_) {
        setCalibratedThresholds({ ...THRESHOLDS });
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const init = async () => {
      try {
        const [vision, mediaStream] = await Promise.all([
          FilesetResolver.forVisionTasks(WASM_URL),
          getSharedStream(),
        ]);
        if (cancelled) return;

        setStream(mediaStream);

        const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL },
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 1,
        });
        if (cancelled) return;
        faceLandmarkerRef.current = faceLandmarker;

        const processingVideo = document.createElement("video");
        processingVideo.autoplay = true;
        processingVideo.muted = true;
        processingVideo.playsInline = true;
        processingVideo.srcObject = mediaStream;
        processingVideoRef.current = processingVideo;
        await processingVideo.play().catch(() => {});

        const samples = [];
        const numSamples = 50;
        const intervalMs = CALIBRATION_DURATION_MS / numSamples;
        let count = 0;

        await new Promise((resolve) => {
          const timer = setInterval(() => {
            if (processingVideo.readyState < 2 || !faceLandmarkerRef.current) return;
            const result = faceLandmarkerRef.current.detectForVideo(processingVideo, performance.now());
            if (result.faceBlendshapes?.[0]) {
              const map = {};
              result.faceBlendshapes[0].categories.forEach((c) => {
                map[c.categoryName] = c.score;
              });
              samples.push(map);
            }
            if (++count >= numSamples) {
              clearInterval(timer);
              resolve();
            }
          }, intervalMs);
        });

        const baseline = {};
        const keys = samples.length > 0 ? Object.keys(samples[0]) : [];
        keys.forEach((key) => {
          baseline[key] = samples.reduce((s, sample) => s + (sample[key] ?? 0), 0) / samples.length;
        });
        baselineRef.current = baseline;
        setIsRunning(true);

        const run = () => {
          if (cancelled) return;
          const lm = faceLandmarkerRef.current;
          const vid = processingVideoRef.current;
          if (!lm || !vid || vid.readyState < 2) {
            animationRef.current = requestAnimationFrame(run);
            return;
          }

          const now = performance.now();
          if (now - lastVideoTimeRef.current < 1000 / 30) {
            animationRef.current = requestAnimationFrame(run);
            return;
          }
          lastVideoTimeRef.current = now;

          const result = lm.detectForVideo(vid, now);
          const baselineLocal = baselineRef.current || {};
          if (!result.faceBlendshapes?.[0]) {
            setBlendshapesDebug({});
            setNormalizedBlendshapes({});
            animationRef.current = requestAnimationFrame(run);
            return;
          }

          const raw = {};
          const normalized = {};
          result.faceBlendshapes[0].categories.forEach((c) => {
            raw[c.categoryName] = c.score;
            normalized[c.categoryName] = Math.max(0, c.score - (baselineLocal[c.categoryName] ?? 0));
          });
          setBlendshapesDebug(raw);
          setNormalizedBlendshapes(normalized);

          const emotion = classifyEmotion(normalized, thresholdsRef.current);
          // Always update detectedEmotion immediately (for TTS)
          // Debug: log detected emotion before setting
          if (emotion !== "neutral") {
            console.log(`[Context] ✅ Detected NON-NEUTRAL emotion: '${emotion}' (updating detectedEmotion state)`);
          }
          setDetectedEmotion(emotion);
          
          if (emotion === lastDetectedEmotionRef.current) {
            sameEmotionCountRef.current += 1;
            if (sameEmotionCountRef.current >= thresholdsRef.current.STABILITY_FRAMES) {
              setCurrentEmotion(emotion);
              const conf = Math.min(100, Math.round(50 + sameEmotionCountRef.current * 5));
              setConfidence(conf);
            }
          } else {
            lastDetectedEmotionRef.current = emotion;
            sameEmotionCountRef.current = 1;
          }

          animationRef.current = requestAnimationFrame(run);
        };

        animationRef.current = requestAnimationFrame(run);
      } catch (err) {
        console.error("FacialEmotionProvider init failed:", err);
      }
    };

    init();
    return () => {
      cancelled = true;
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      if (faceLandmarkerRef.current?.close) {
        try {
          faceLandmarkerRef.current.close();
        } catch (_) {}
      }
      faceLandmarkerRef.current = null;
      if (processingVideoRef.current) {
        processingVideoRef.current.srcObject = null;
        processingVideoRef.current = null;
      }
    };
  }, []);

  const value = useMemo(
    () => ({
      stream,
      currentEmotion,
      detectedEmotion, // Most recent detected emotion (for TTS)
      confidence,
      blendshapesDebug,
      normalizedBlendshapes,
      calibratedThresholds,
      isRunning,
      updateThresholds,
    }),
    [
      stream,
      currentEmotion,
      detectedEmotion,
      confidence,
      blendshapesDebug,
      normalizedBlendshapes,
      calibratedThresholds,
      isRunning,
      updateThresholds,
    ]
  );

  return <FacialEmotionContext.Provider value={value}>{children}</FacialEmotionContext.Provider>;
}

export function useFacialEmotion() {
  const ctx = useContext(FacialEmotionContext);
  if (!ctx) {
    throw new Error("useFacialEmotion must be used within FacialEmotionProvider");
  }
  return ctx;
}
