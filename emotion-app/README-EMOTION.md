# Facial Emotion Detection (Local)

React web app that detects 7 emotions from your webcam using MediaPipe Face Landmarker. Runs **entirely in the browser** on localhost — no backend, no API keys.

## Quick start

```bash
cd emotion-app
npm install
npm run dev
```

Then open **Chrome** and go to:

**http://localhost:5173/**

Allow camera access when prompted.

## What you need to run

- **Node.js** (v18+)
- **Chrome** (recommended; webcam works on localhost)
- **Camera** (built-in or USB)

## Commands

| Command | Description |
|--------|-------------|
| `npm install` | Install dependencies (including `@mediapipe/tasks-vision`) |
| `npm run dev` | Start dev server at http://localhost:5173/ |
| `npm run build` | Production build |
| `npm run preview` | Preview production build |

## What you see on screen

1. **Loading** — MediaPipe and webcam are initializing.
2. **Calibration (5 seconds)** — Message: *"Look at the camera naturally for 5 seconds"*. Stay still and neutral. This records your personal baseline for the 52 blendshapes.
3. **Running** — Live camera feed with:
   - **Large label** — Current emotion: one of `confident`, `warm`, `excited`, `emphatic`, `serious`, `passionate`, `reflective`.
   - **Confidence** — 0–100% for the current emotion.
   - **Last 5 emotions** — Log with timestamps.
   - **Debug panel** — Collapsible list of all 52 blendshape values (see below).

Emotion only updates after the **same** emotion is detected for **10 consecutive frames** (configurable in `THRESHOLDS`).

## Tuning thresholds for your face

All thresholds live in **one place** at the top of:

`src/components/FacialEmotionDetection.jsx`

Look for the **`THRESHOLDS`** object. Examples:

- **STABILITY_FRAMES** — Frames the same emotion must be detected before switching (default: 10).
- **CONFIDENT_*** — Rules for “confident” (e.g. max variance, smile range).
- **WARM_***, **EXCITED_***, **EMPHATIC_***, **SERIOUS_***, **PASSIONATE_***, **REFLECTIVE_*** — Same idea for each emotion.

**Using the debug panel:**

1. Click **“Show debug panel (52 blendshapes)”**.
2. Make different faces and watch how values change (e.g. `mouthSmileLeft`, `browInnerUp`, `eyeWideLeft`).
3. Note the ranges you see for expressions you care about.
4. Edit the `THRESHOLDS` constants in `FacialEmotionDetection.jsx` to match those ranges and save; the app will hot-reload.

Blendshape scores are **normalized** using your 5-second calibration baseline, so thresholds are relative to your neutral face.

## Using the emotion in other components

**Option 1 — Callback prop**

```jsx
<FacialEmotionDetection
  onEmotionChange={({ emotion, confidence }) => {
    console.log(emotion, confidence);
    // use emotion and confidence
  }}
/>
```

**Option 2 — React context**

Wrap the app (or a subtree) with `EmotionProvider` (already done in `App.jsx`). Then in any child:

```jsx
import { useEmotion } from "./context/EmotionContext";

function MyComponent() {
  const { emotion, confidence } = useEmotion();
  return <span>{emotion} ({confidence}%)</span>;
}
```

Output shape: **`{ emotion: string, confidence: number }`**.

## Tech stack

- **React** (hooks only)
- **Vite** (dev server + build)
- **@mediapipe/tasks-vision** — Face Landmarker with 52 blendshapes, `runningMode: "VIDEO"` (live stream)
- No backend, no extra ML libraries — only MediaPipe + hand-coded rules in `classifyEmotion(blendshapes)`

## File layout

- `src/components/FacialEmotionDetection.jsx` — Main component (MediaPipe, calibration, classifier, UI, debug panel).
- `src/components/FacialEmotionDetection.css` — Styles for the component.
- `src/context/EmotionContext.jsx` — Optional context for `{ emotion, confidence }`.
- `src/App.jsx` — Uses the component and shows exported emotion.

Blendshape scores are logged to the **browser console** every frame in dev so you can watch values while tuning.
