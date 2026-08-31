import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dev server proxies /api to the backend, so the browser sees one origin
// and there is no CORS configuration to get wrong. In production nginx does
// the same (see docker/nginx.conf).
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": {
        // H3_API points the dev server at a backend on another port.
        target: process.env.H3_API ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
