import { defineConfig, type Plugin } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

const PKG_DIR = new URL("./pkg", import.meta.url).pathname;

// Force a full browser reload whenever `cargo watch` rebuilds the WASM package.
// Vite's HMR keys off the JS module graph, but a Rust *logic* change may rewrite
// only `app_wasm_bg.wasm` (the JS bindings stay byte-identical), which Vite would
// otherwise not pick up. Watch the generated pkg/ explicitly and full-reload on
// any .wasm/.js change so `npm run dev` live-reloads on Rust edits.
function reloadOnWasm(): Plugin {
  return {
    name: "reload-on-wasm-rebuild",
    apply: "serve",
    configureServer(server) {
      server.watcher.add(PKG_DIR);
      // A single wasm-pack run rewrites both app_wasm.js and app_wasm_bg.wasm;
      // debounce so the browser reloads once per rebuild, not per file.
      let timer: ReturnType<typeof setTimeout> | undefined;
      const onChange = (file: string) => {
        if (!file.startsWith(PKG_DIR) || !(file.endsWith(".wasm") || file.endsWith(".js"))) {
          return;
        }
        clearTimeout(timer);
        timer = setTimeout(() => {
          server.config.logger.info("[wasm] rebuilt → full reload", { timestamp: true });
          server.ws.send({ type: "full-reload", path: "*" });
        }, 150);
      };
      server.watcher.on("change", onChange);
      server.watcher.on("add", onChange);
    },
  };
}

// The Rust/WASM package is generated into ./pkg by `wasm-pack build`.
// We alias it as `pkg` so the TS shell can import from `pkg` regardless of depth.
export default defineConfig({
  base: "./",
  plugins: [wasm(), topLevelAwait(), reloadOnWasm()],
  resolve: {
    alias: {
      pkg: new URL("./pkg", import.meta.url).pathname,
    },
  },
  server: {
    fs: {
      // Allow serving the sibling `pkg` directory and the .wasm asset.
      allow: [".."],
    },
  },
  build: {
    target: "esnext",
  },
  optimizeDeps: {
    // The generated wasm glue should not be pre-bundled by esbuild.
    exclude: ["pkg"],
  },
});
