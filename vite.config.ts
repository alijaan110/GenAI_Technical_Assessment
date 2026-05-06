import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
    plugins: [react(), tailwindcss()],
    server: {
        port: 5173,
        proxy: {
            "/api": {
                target: "http://127.0.0.1:8001",
                changeOrigin: true,
                // SSE-friendly: don't buffer responses for the streaming chat.
                ws: false,
                configure: (proxy) => {
                    proxy.on("proxyRes", (proxyRes) => {
                        proxyRes.headers["x-accel-buffering"] = "no";
                    });
                },
            },
        },
    },
});
