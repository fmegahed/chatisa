import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Self-hosted WASM runtimes are vendored third-party assets, not our code.
    "public/runtimes/**",
    // Generated deploy bundles (make-deploy-bundle.mjs): built output, not source.
    "deploy/**",
  ]),
]);

export default eslintConfig;
