import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    server: {
        port: 3000,
        proxy: {
            '/api': {
                target: 'http://localhost:7272',
                changeOrigin: true,
            },
            '/ws': {
                target: 'ws://localhost:7272',
                ws: true,
            }
        }
    }
})
