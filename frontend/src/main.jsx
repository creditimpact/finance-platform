import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './devtools/setupFrontendReviewMock'
import './index.css'
import App from './App.jsx'
import { ToastProvider } from './components/ToastProvider'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ToastProvider>
      <App />
    </ToastProvider>
  </StrictMode>,
)
