import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './devtools/setupFrontendReviewMock'
import './index.css'
import App from './App.jsx'
import AppErrorBoundary from './components/AppErrorBoundary'
import { ToastProvider } from './components/ToastProvider'

if (import.meta.env?.VITE_API_BASE_URL) {
  console.info('[finance-platform] API base URL:', import.meta.env.VITE_API_BASE_URL)
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <AppErrorBoundary>
      <ToastProvider>
        <App />
      </ToastProvider>
    </AppErrorBoundary>
  </StrictMode>,
)
