import { createContext, useContext, useState } from "react";

const defaultState = { emotion: "confident", confidence: 0 };
const EmotionContext = createContext([defaultState, () => {}]);

export function EmotionProvider({ children }) {
  const [state, setState] = useState(defaultState);
  return (
    <EmotionContext.Provider value={[state, setState]}>
      {children}
    </EmotionContext.Provider>
  );
}

export function useEmotion() {
  const [state] = useContext(EmotionContext);
  return state;
}

export function useSetEmotion() {
  const ctx = useContext(EmotionContext);
  return ctx[1];
}

export { EmotionContext };
