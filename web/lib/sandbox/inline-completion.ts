/**
 * A minimal inline (ghost-text) code-completion extension for CodeMirror 6.
 *
 * On a typing pause it asks a source for a suggestion, shows it as dimmed text
 * after the cursor, and lets the student accept it with Tab or dismiss it with
 * Escape. Tab only accepts when a suggestion is showing; otherwise it falls
 * through to its normal behaviour (moving focus), so there is no keyboard trap,
 * and Escape always clears a suggestion.
 *
 * Client-only: uses the DOM and timers, and is imported lazily with CodeMirror.
 */
import {
  Decoration,
  EditorView,
  ViewPlugin,
  WidgetType,
  keymap,
  type ViewUpdate,
} from "@codemirror/view";
import { Prec, StateEffect, StateField } from "@codemirror/state";

export interface CompletionContext {
  prefix: string;
  suffix: string;
  signal: AbortSignal;
}

export type CompletionSource = (ctx: CompletionContext) => Promise<string>;

interface Suggestion {
  text: string;
  pos: number;
}

const setSuggestion = StateEffect.define<Suggestion | null>();

const suggestionField = StateField.define<Suggestion | null>({
  create: () => null,
  update(value, tr) {
    for (const effect of tr.effects) {
      if (effect.is(setSuggestion)) return effect.value;
    }
    // Any edit, or a cursor move away from the anchor, invalidates it.
    if (tr.docChanged) return null;
    if (value && tr.state.selection.main.head !== value.pos) return null;
    return value;
  },
  provide: (field) =>
    EditorView.decorations.from(field, (value) =>
      value && value.text
        ? Decoration.set([
            Decoration.widget({
              widget: new GhostWidget(value.text),
              side: 1,
            }).range(value.pos),
          ])
        : Decoration.none,
    ),
});

class GhostWidget extends WidgetType {
  constructor(readonly text: string) {
    super();
  }
  eq(other: GhostWidget) {
    return other.text === this.text;
  }
  toDOM() {
    const span = document.createElement("span");
    span.className = "cm-ghost-text";
    span.setAttribute("aria-hidden", "true");
    span.textContent = this.text;
    return span;
  }
}

function currentSuggestion(view: EditorView): Suggestion | null {
  return view.state.field(suggestionField, false) ?? null;
}

const acceptKeymap = Prec.highest(
  keymap.of([
    {
      key: "Tab",
      run(view) {
        const suggestion = currentSuggestion(view);
        if (!suggestion || !suggestion.text) return false;
        view.dispatch({
          changes: { from: suggestion.pos, insert: suggestion.text },
          selection: { anchor: suggestion.pos + suggestion.text.length },
          effects: setSuggestion.of(null),
        });
        return true;
      },
    },
    {
      key: "Escape",
      run(view) {
        if (!currentSuggestion(view)) return false;
        view.dispatch({ effects: setSuggestion.of(null) });
        return true;
      },
    },
  ]),
);

const DEBOUNCE_MS = 200;

function completionPlugin(getSource: () => CompletionSource | null) {
  return ViewPlugin.fromClass(
    class {
      timer = 0;
      controller: AbortController | null = null;

      update(update: ViewUpdate) {
        // Re-request only after a real edit; setting the suggestion itself is
        // not a doc change, so this does not loop.
        if (update.docChanged) this.schedule(update.view);
      }

      schedule(view: EditorView) {
        window.clearTimeout(this.timer);
        this.controller?.abort();
        this.timer = window.setTimeout(
          () => void this.request(view),
          DEBOUNCE_MS,
        );
      }

      async request(view: EditorView) {
        const source = getSource();
        if (!source) return;
        const sel = view.state.selection.main;
        if (!sel.empty) return;
        const pos = sel.head;
        const prefix = view.state.sliceDoc(0, pos);
        const suffix = view.state.sliceDoc(pos);
        if (prefix.trim().length === 0) return;

        this.controller = new AbortController();
        let text: string;
        try {
          text = await source({
            prefix,
            suffix,
            signal: this.controller.signal,
          });
        } catch {
          return;
        }
        if (!text) return;
        // Only show it if the cursor has not moved since we asked.
        if (view.state.selection.main.head !== pos) return;
        view.dispatch({ effects: setSuggestion.of({ text, pos }) });
      }

      destroy() {
        window.clearTimeout(this.timer);
        this.controller?.abort();
      }
    },
  );
}

const ghostTheme = EditorView.baseTheme({
  // Colour is set by the editor's light/dark theme so it stays visible on both
  // backgrounds; here just the layout and a fallback style.
  ".cm-ghost-text": {
    whiteSpace: "pre",
    fontStyle: "italic",
  },
});

/**
 * The full inline-completion extension. `getSource` is read fresh on each
 * request, so the language and model can change without rebuilding the editor.
 */
export function inlineCompletion(getSource: () => CompletionSource | null) {
  return [
    suggestionField,
    acceptKeymap,
    completionPlugin(getSource),
    ghostTheme,
  ];
}
