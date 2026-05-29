import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],

  base: '/aquatune/',

  build: {
    outDir: 'docs',
    target: 'esnext',
    sourcemap: true,
    modulePreload: { polyfill: false },
  },

  // ORT uses dynamic imports internally — don't pre-bundle it
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },

  // ES module workers (required for comlink + ORT in worker)
  worker: {
    format: 'es',
    rollupOptions: {
      external: ['onnxruntime-web', 'onnxruntime-web/all'],
    },
  },

  server: {
    headers: {
      // Required for SharedArrayBuffer (ORT multi-threaded WASM)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },

  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
