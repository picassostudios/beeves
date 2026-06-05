/**
 * Minimal toolbar wiring. All actual tool behavior lives in Rust (WasmApp).
 * This helper only:
 *   - reads `data-tool` from toolbar buttons,
 *   - forwards the selected tool to a callback (which calls `set_tool`),
 *   - tracks/reflects the active tool via an `is-active` class.
 */

export type ToolName =
  | "brush"
  | "vectordraw"
  | "bezier"
  | "edit"
  | "sculpt"
  | "blend"
  | "select";

export const TOOLS: readonly ToolName[] = [
  "brush",
  "vectordraw",
  "bezier",
  "edit",
  "sculpt",
  "blend",
  "select",
] as const;

function isToolName(value: string): value is ToolName {
  return (TOOLS as readonly string[]).includes(value);
}

export interface Toolbar {
  /** The currently active tool. */
  readonly active: ToolName;
  /** Programmatically select a tool (also fires the onSelect callback). */
  select(tool: ToolName): void;
}

/**
 * Wire all `button[data-tool]` elements within `container`.
 *
 * @param container  the toolbar element holding the tool buttons.
 * @param onSelect   invoked with the tool name whenever the active tool
 *                   changes (forward this to `app.set_tool`).
 * @param initial    the tool to start on (default "brush").
 */
export function createToolbar(
  container: HTMLElement,
  onSelect: (tool: ToolName) => void,
  initial: ToolName = "brush"
): Toolbar {
  const buttons = Array.from(
    container.querySelectorAll<HTMLButtonElement>("button[data-tool]")
  );

  let active: ToolName = initial;

  function reflect(): void {
    for (const btn of buttons) {
      const isActive = btn.dataset.tool === active;
      btn.classList.toggle("is-active", isActive);
      btn.setAttribute("aria-pressed", String(isActive));
    }
  }

  function select(tool: ToolName): void {
    active = tool;
    reflect();
    onSelect(tool);
  }

  for (const btn of buttons) {
    const tool = btn.dataset.tool ?? "";
    if (!isToolName(tool)) continue;
    btn.addEventListener("click", () => select(tool));
  }

  // Apply the initial selection (reflect + notify).
  select(active);

  return {
    get active() {
      return active;
    },
    select,
  };
}
