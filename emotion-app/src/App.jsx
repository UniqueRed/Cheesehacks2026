import { useState } from "react";
import { EmotionProvider, useEmotion, useSetEmotion } from "./context/EmotionContext";
import FacialEmotionDetection from "./components/FacialEmotionDetection";
import "./App.css";

function EmotionDemo() {
  const [emotionState, setEmotionState] = useState({ emotion: "confident", confidence: 0 });
  const setContextEmotion = useSetEmotion();

  return (
    <div className="app">
      <h1>Facial Emotion Detection</h1>
      <p className="subtitle">Local webcam + MediaPipe — no backend</p>

      <FacialEmotionDetection
        onEmotionChange={(payload) => {
          setEmotionState(payload);
          if (setContextEmotion) setContextEmotion(payload);
        }}
      />

      <div className="export-box">
        <strong>Export for other components</strong>
        <pre>{JSON.stringify({ emotion: emotionState.emotion, confidence: emotionState.confidence }, null, 2)}</pre>
      </div>
    </div>
  );
}

function App() {
  return (
    <EmotionProvider>
      <EmotionDemo />
    </EmotionProvider>
  );
}

export default App;
