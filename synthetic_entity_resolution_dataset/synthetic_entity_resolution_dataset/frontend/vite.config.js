import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'

// Ensure process.cwd matches real canonical path when invoked through directory junctions on Windows
try {
  const realCwd = fs.realpathSync(process.cwd())
  if (realCwd !== process.cwd()) {
    process.chdir(realCwd)
  }
} catch (e) {
  // Ignore if realpath fails
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: 'localhost',
    open: false,
    fs: {
      strict: false,
    },
  },
})
