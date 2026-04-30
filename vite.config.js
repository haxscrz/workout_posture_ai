import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

export default defineConfig({
  plugins: [
    basicSsl()
  ],
  server: {
    host: '0.0.0.0', // Listen on all local IPs so mobile can connect
    port: 5173,
    strictPort: true,
    https: true, // Required for mobile browsers to allow camera access
  }
});
