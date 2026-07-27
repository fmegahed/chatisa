import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname),
      // Modules holding server-side credentials import "server-only" so that
      // importing them from a client component is a build error. That package
      // throws unless resolved under the "react-server" condition, and Vitest
      // externalizes node_modules so Node's resolver picks the throwing entry.
      // Point at the package's own no-op entry, which is what Next resolves to
      // on the server. This keeps the client-import guard intact in the app
      // while letting these Node tests exercise the modules it protects.
      "server-only": path.resolve(__dirname, "node_modules/server-only/empty.js"),
    },
  },
  test: {
    environment: "node",
    include: ["tests/unit/**/*.test.ts"],
  },
});
