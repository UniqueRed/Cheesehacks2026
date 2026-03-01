import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { FacialEmotionProvider } from './context/FacialEmotionContext.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <FacialEmotionProvider>
      <App />
    </FacialEmotionProvider>
  </StrictMode>,
)
