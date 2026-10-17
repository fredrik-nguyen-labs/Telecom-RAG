import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Apply a restrictive CSP only to the production HTML. Vite dev mode needs
// websocket/React-refresh allowances that should not reach production.
const productionCsp = [
  "default-src 'self'",
  "base-uri 'self'",
  "form-action 'self'",
  "object-src 'none'",
  "script-src 'self'",
  "style-src 'self'",
  "img-src 'self' data:",
  "font-src 'self'",
  "connect-src 'self' https://telecom-rag-api.onrender.com https://api.telecom.fnsystems.dev",
  "frame-src 'none'",
  "upgrade-insecure-requests",
].join("; ");

export default defineConfig({
  plugins: [
    react(),
    {
      name: "production-csp",
      apply: "build",
      transformIndexHtml(html: string) {
        return html.replace(
          "</head>",
          `<meta http-equiv="Content-Security-Policy" content="${productionCsp}" />\n  </head>`,
        );
      },
    },
  ],
  server: {
    port: 5173,
  },
});
