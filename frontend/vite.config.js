import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const rawProxyTarget =
  process.env.VITE_DEV_PROXY_TARGET ||
  process.env.VITE_API_BASE_URL ||
  process.env.VITE_API_URL ||
  'http://127.0.0.1:5000'

const proxyTarget = rawProxyTarget.replace(/\/+$/, '') || 'http://127.0.0.1:5000'

const createProxyConfig = () => ({
  target: proxyTarget,
  changeOrigin: true,
  secure: false,
})

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': createProxyConfig(),
      '/runs': createProxyConfig(),
    },
    hmr: {
      overlay: true,
    },
  },
})
