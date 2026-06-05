/**
 * Typed loader for the Rust/WASM package.
 *
 * Imports from the `pkg` alias (aliased to `web/pkg` in vite.config.ts and
 * tsconfig paths). The package's generated types are described in
 * `wasm-types.d.ts` until the real `pkg/` is built by the integrate phase.
 */
import init, { WasmApp } from "pkg";

export { WasmApp };

let initialized: Promise<void> | null = null;

/**
 * Initialize the wasm module exactly once. Safe to call repeatedly; the
 * underlying init runs a single time and subsequent calls await the same
 * promise.
 */
export function initWasm(): Promise<void> {
  if (initialized === null) {
    initialized = init().then(() => undefined);
  }
  return initialized;
}

/**
 * Convenience: ensure the module is initialized, then construct a WasmApp
 * bound to the given canvas.
 */
export async function createApp(
  canvas: HTMLCanvasElement
): Promise<WasmApp> {
  await initWasm();
  return WasmApp.new(canvas);
}
