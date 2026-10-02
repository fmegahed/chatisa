"use client";

import { useEffect, useId, useRef, useState, useSyncExternalStore } from "react";
import type { EditorState, Extension } from "@codemirror/state";
import type { EditorView } from "@codemirror/view";
import type { CompletionSource } from "@/lib/sandbox/inline-completion";
import type { CompletionResult } from "@/lib/run/manager";
import type { HelpRequest } from "@/lib/sandbox/help-docs";
import {
  CARET_POSITION_KEY,
  buildPipeInsertion,
  caretAnnouncement,
  caretPositionAt,
  caretPositionKeysLabel,
  caretStatusText,
  type CaretPosition,
} from "@/lib/sandbox/editor-keys";
import { detectIsMac } from "@/lib/sandbox/shortcuts";
import { announce } from "@/lib/a11y/announce";

/** A store that never changes, so useSyncExternalStore reads its snapshot once. */
function subscribeNever(): () => void {
  return () => {};
}

/** The caret position of a CodeMirror state's main selection. */
function caretOf(state: EditorState): CaretPosition {
  return caretPositionAt(
    (pos) => state.doc.lineAt(pos),
    state.doc.lines,
    state.selection.main.head,
  );
}

/** Queries the runtime for autocomplete candidates for the text before the cursor. */
export type RuntimeCompleteSource = (
  prefix: string,
) => Promise<CompletionResult | null>;

/**
 * A code editor that upgrades to CodeMirror, lazily.
 *
 * CodeMirror is never in the initial bundle: it is imported only when this
 * component mounts (Customize inline, or the Sandbox). Until it loads (and if it
 * ever fails to), a plain textarea stands in, so editing works immediately and
 * degrades gracefully. Both editors carry the same accessible name.
 *
 * Controlled: `value` is the source of truth and `onChange` reports every edit.
 */
export function CodeEditor(props: {
  value: string;
  onChange: (next: string) => void;
  /** The runnable language id, used to pick a syntax mode. */
  languageId: string;
  /** Accessible name for the editor. */
  label: string;
  /** Dark syntax theme (approximating Tomorrow Night Bright). */
  dark?: boolean;
  /** Fill the parent's height instead of the capped inline height. */
  fillHeight?: boolean;
  /** When set, the editor offers inline (ghost-text) AI completions. */
  completionSource?: CompletionSource;
  /** When set, the editor offers a runtime autocomplete popup (member/name list). */
  completeSource?: RuntimeCompleteSource;
  /** Run the current statement or selection (Mod-Enter), advancing the cursor. */
  onRunLine?: (code: string) => void;
  /** Run the whole script, echoed to the console (Mod-Shift-Enter). */
  onRunAll?: () => void;
  /** Source the whole script silently (Mod-Shift-s). */
  onSource?: () => void;
  /** Resolve the symbol under a Ctrl/Cmd+Click (or F1 at the cursor) and open it
   *  in the HELP tab. Does not move the caret. */
  onHelp?: (req: HelpRequest) => void;
  /** Focus the editor once it loads (the student asked to edit). Otherwise it
   *  takes focus only if focus was already inside it, so switching language
   *  or theme leaves focus on the control the student used (#17). */
  autoFocus?: boolean;
}) {
  const host = useRef<HTMLDivElement | null>(null);
  // Whether focus was inside the editor when it was torn down for a rebuild.
  const hadFocusRef = useRef(false);
  // The plain textarea shown while CodeMirror loads; text typed there carries
  // over, and so should focus.
  const fallbackRef = useRef<HTMLTextAreaElement | null>(null);
  const viewRef = useRef<EditorView | null>(null);
  // Keep the latest onChange without re-creating the editor on every keystroke.
  const onChangeRef = useRef(props.onChange);
  // The latest text, so an editor that finishes loading after the student
  // has typed into the stand-in textarea starts from what they typed, not
  // from the text as it was when loading began.
  const valueRef = useRef(props.value);
  // The completion sources are read fresh on each request, so switching them (or
  // turning them off) does not rebuild the editor; only their presence does.
  const completionSourceRef = useRef(props.completionSource);
  const completeSourceRef = useRef(props.completeSource);
  // Run handlers, read fresh so the editor is not rebuilt when they change.
  const runRef = useRef({
    onRunLine: props.onRunLine,
    onRunAll: props.onRunAll,
    onSource: props.onSource,
  });
  // Help handler, read fresh so the editor is not rebuilt when it changes.
  const helpRef = useRef(props.onHelp);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  // The caret's line and column, shown in the status line under the editor
  // (#21). Not a live region: the Alt+Shift+L shortcut announces it on demand.
  const [caret, setCaret] = useState<CaretPosition>({ line: 1, column: 1, lines: 1 });
  const hintId = useId();
  // Client-only platform fact (Option vs Alt in the hint), hydration-safe.
  const isMac = useSyncExternalStore(subscribeNever, detectIsMac, () => false);
  const { dark = false, fillHeight = false } = props;
  const hasCompletion = props.completionSource != null;
  const hasComplete = props.completeSource != null;
  const hasRun = props.onRunAll != null;
  const hasHelp = props.onHelp != null;

  useEffect(() => {
    onChangeRef.current = props.onChange;
  }, [props.onChange]);

  useEffect(() => {
    valueRef.current = props.value;
  }, [props.value]);

  useEffect(() => {
    completionSourceRef.current = props.completionSource;
  }, [props.completionSource]);

  useEffect(() => {
    completeSourceRef.current = props.completeSource;
  }, [props.completeSource]);

  useEffect(() => {
    runRef.current = {
      onRunLine: props.onRunLine,
      onRunAll: props.onRunAll,
      onSource: props.onSource,
    };
  }, [props.onRunLine, props.onRunAll, props.onSource]);

  useEffect(() => {
    helpRef.current = props.onHelp;
  }, [props.onHelp]);

  useEffect(() => {
    let cancelled = false;
    const hostEl = host.current;
    loadCodeMirror(props.languageId)
      .then(({ view, cm, lang, state, autocomplete, tags, commands, langExt, inline, langStructure, helpDocs, lint }) => {
        if (cancelled || !host.current) return;
        const editor = new view.EditorView({
          doc: valueRef.current,
          parent: host.current,
          extensions: [
            cm.basicSetup,
            langExt,
            ...indentExtensions(lang, langStructure, props.languageId),
            view.EditorView.updateListener.of((update) => {
              if (update.docChanged) {
                onChangeRef.current(update.state.doc.toString());
              }
              if (update.docChanged || update.selectionSet) {
                setCaret(caretOf(update.state));
              }
            }),
            // The contenteditable gets the accessible name; CodeMirror already
            // gives it role="textbox" and aria-multiline. The description names
            // the caret-position shortcut (#21). An explicit tabindex makes the
            // scroller's focusable content visible to checkers (axe
            // scrollable-region-focusable, #14) and keeps the content reachable
            // if it is ever made read-only (contenteditable=false). It is the
            // same single tab stop: a contenteditable is already in tab order.
            view.EditorView.contentAttributes.of({
              "aria-label": props.label,
              "aria-describedby": hintId,
              tabindex: "0",
            }),
            ...themeExtensions(view, lang, tags, dark, fillHeight, props.languageId),
            editorKeymap(view, state, commands, props.languageId),
            caretKeymap(view, state, lint),
            ...(hasRun
              ? [runKeymap(view, state, runRef, langStructure.statementRangeAt, props.languageId)]
              : []),
            ...(hasHelp
              ? [
                  helpMouse(view, helpRef, helpDocs.symbolAt, props.languageId),
                  helpKeymap(view, state, helpRef, helpDocs.symbolAt, props.languageId),
                ]
              : []),
            lintExtension(lint, langStructure, props.languageId),
            ...(hasCompletion
              ? [inline.inlineCompletion(() => completionSourceRef.current ?? null)]
              : []),
            ...(hasComplete
              ? [
                  runtimeAutocomplete(
                    autocomplete,
                    () => completeSourceRef.current ?? null,
                  ),
                ]
              : []),
          ],
        });
        viewRef.current = editor;
        setCaret(caretOf(editor.state));
        if (
          props.autoFocus ||
          hadFocusRef.current ||
          host.current?.contains(document.activeElement) ||
          (fallbackRef.current !== null &&
            fallbackRef.current === document.activeElement)
        ) {
          editor.focus();
        }
        hadFocusRef.current = false;
        setReady(true);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
      hadFocusRef.current = !!hostEl?.contains(document.activeElement);
      viewRef.current?.destroy();
      viewRef.current = null;
    };
    // Rebuilt when the language, theme, completions or run keys are toggled;
    // value is seeded once and then kept in sync by the effect below (so a
    // rebuild preserves the text).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.languageId, dark, fillHeight, hasCompletion, hasComplete, hasRun, hasHelp]);

  // Reflect external value changes (for example Reset) into the live editor,
  // without clobbering edits the student is making.
  useEffect(() => {
    const editor = viewRef.current;
    if (!editor) return;
    const current = editor.state.doc.toString();
    if (props.value !== current) {
      editor.dispatch({
        changes: { from: 0, to: current.length, insert: props.value },
      });
    }
  }, [props.value]);

  const usingCodeMirror = ready && !failed;
  const rows = Math.min(30, Math.max(6, props.value.split("\n").length + 1));
  const wrapBg = dark ? "bg-[#0a0a0a]" : "bg-light-tan";
  // Status line colours match the gutter: >=5.7:1 on light tan, 7:1 on #0a0a0a.
  const statusTone = dark
    ? "border-[#2c2c2c] text-[#9a9a9a]"
    : "border-medium-tan text-[#5f5a50]";

  return (
    <div className={fillHeight ? "h-full" : undefined}>
      {/* The editor box, kept in the DOM but hidden until ready so CodeMirror
          has a parent to attach to. */}
      <div
        className={
          usingCodeMirror
            ? `flex flex-col overflow-hidden rounded-card border border-medium-tan ${wrapBg} focus-within:border-miami-red ${fillHeight ? "h-full" : ""}`
            : "hidden"
        }
      >
        {/* CodeMirror mounts here. */}
        <div ref={host} className={fillHeight ? "min-h-0 flex-1" : undefined} />
        {/* Caret position, VS Code style (#21). Plain text, not a live region,
            so it does not chatter on every keystroke; screen reader users hear
            it on demand with the shortcut named in the hint below. */}
        <div
          className={`shrink-0 border-t px-2 py-0.5 text-right font-mono text-xs ${statusTone}`}
          data-testid="editor-caret-status"
        >
          <span aria-hidden="true" data-testid="editor-caret-status-text">
            {caretStatusText(caret)}
          </span>
          <span className="sr-only">
            Line {caret.line}, column {caret.column}
          </span>
        </div>
      </div>
      {usingCodeMirror ? (
        <p id={hintId} className="sr-only">
          Press {caretPositionKeysLabel(isMac)} to hear the line and column.
        </p>
      ) : null}
      {!usingCodeMirror ? (
        <textarea
          ref={fallbackRef}
          aria-label={props.label}
          value={props.value}
          onChange={(e) => props.onChange(e.target.value)}
          spellCheck={false}
          rows={fillHeight ? undefined : rows}
          className={`w-full resize-y rounded-card border border-medium-tan p-3 font-mono text-sm focus:border-miami-red ${dark ? "bg-[#0a0a0a] text-[#eaeaea]" : "bg-light-tan text-ink"} ${fillHeight ? "h-full resize-none" : ""}`}
        />
      ) : null}
    </div>
  );
}

interface LoadedEditor {
  view: typeof import("@codemirror/view");
  cm: typeof import("codemirror");
  lang: typeof import("@codemirror/language");
  state: typeof import("@codemirror/state");
  autocomplete: typeof import("@codemirror/autocomplete");
  tags: typeof import("@lezer/highlight")["tags"];
  commands: typeof import("@codemirror/commands");
  inline: typeof import("@/lib/sandbox/inline-completion");
  langStructure: typeof import("@/lib/sandbox/lang-structure");
  helpDocs: typeof import("@/lib/sandbox/help-docs");
  lint: typeof import("@codemirror/lint");
  langExt: Extension;
}

/** Dynamically imports CodeMirror and the syntax mode for `languageId`. */
async function loadCodeMirror(languageId: string): Promise<LoadedEditor> {
  const [cm, view, lang, state, autocomplete, highlight, commands, inline, langStructure, helpDocs, lint, langExt] =
    await Promise.all([
      import("codemirror"),
      import("@codemirror/view"),
      import("@codemirror/language"),
      import("@codemirror/state"),
      import("@codemirror/autocomplete"),
      import("@lezer/highlight"),
      import("@codemirror/commands"),
      import("@/lib/sandbox/inline-completion"),
      import("@/lib/sandbox/lang-structure"),
      import("@/lib/sandbox/help-docs"),
      import("@codemirror/lint"),
      loadLanguageMode(languageId),
    ]);
  return {
    view,
    cm,
    lang,
    state,
    autocomplete,
    tags: highlight.tags,
    commands,
    inline,
    langStructure,
    helpDocs,
    lint,
    langExt,
  };
}

/** A CodeMirror completion source backed by the runtime (via `getSource`). */
function runtimeAutocomplete(
  autocomplete: LoadedEditor["autocomplete"],
  getSource: () => RuntimeCompleteSource | null,
): Extension {
  return autocomplete.autocompletion({
    activateOnTyping: true,
    override: [
      async (ctx) => {
        const source = getSource();
        if (!source) return null;
        const before = ctx.matchBefore(/[\w.$:]+/);
        if (!ctx.explicit && !before) return null;
        let result: CompletionResult | null;
        try {
          result = await source(ctx.state.sliceDoc(0, ctx.pos));
        } catch {
          return null;
        }
        if (!result || result.options.length === 0) return null;
        return {
          from: ctx.pos - (result.partial?.length ?? 0),
          validFor: /^[\w.]*$/,
          options: result.options.map((o) => ({
            label: o.label,
            type: o.type,
            detail: o.detail || undefined,
          })),
        };
      },
    ],
  });
}

/** The syntax mode for a language. */
async function loadLanguageMode(languageId: string): Promise<Extension> {
  if (languageId === "python") {
    return (await import("@codemirror/lang-python")).python();
  }
  if (languageId === "sql") {
    return (await import("@codemirror/lang-sql")).sql();
  }
  if (languageId === "r") {
    // R has no first-party CodeMirror 6 grammar; the legacy stream mode gives
    // comment/keyword/string/function highlighting, which is what students expect.
    const [{ StreamLanguage }, { r }] = await Promise.all([
      import("@codemirror/language"),
      import("@codemirror/legacy-modes/mode/r"),
    ]);
    const rLang = StreamLanguage.define(r);
    // Own the comment token rather than relying on the legacy grammar to keep
    // providing it, so Mod-/ toggles `#` comments in R regardless of the mode's
    // version. `toggleComment` reads this via state.languageDataAt.
    return [rLang, rLang.data.of({ commentTokens: { line: "#" } })];
  }
  return [];
}

const MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

interface RunHandlers {
  onRunLine?: (code: string) => void;
  onRunAll?: () => void;
  onSource?: () => void;
}

/** RStudio-style run keys, reading the latest handlers from a ref:
 *  Mod-Enter runs the current line/selection and advances; Mod-Shift-Enter runs
 *  the whole script (echoed); Mod-Shift-s sources it silently. */
function runKeymap(
  view: LoadedEditor["view"],
  cmState: LoadedEditor["state"],
  runRef: { current: RunHandlers },
  statementRangeAt: LoadedEditor["langStructure"]["statementRangeAt"],
  languageId: string,
): Extension {
  const lang: "r" | "python" | "sql" =
    languageId === "python" ? "python" : languageId === "sql" ? "sql" : "r";
  return cmState.Prec.high(
    view.keymap.of([
      {
        key: "Mod-Enter",
        preventDefault: true,
        run: (editor) => {
          const handler = runRef.current.onRunLine;
          if (!handler) return false;
          const state = editor.state;
          const sel = state.selection.main;
          let code: string;
          let nextPos: number;
          if (!sel.empty) {
            // Selection-first: run exactly what is selected (unchanged).
            code = state.sliceDoc(sel.from, sel.to);
            const endLine = state.doc.lineAt(sel.to).number;
            nextPos =
              endLine < state.doc.lines
                ? state.doc.line(endLine + 1).from
                : state.doc.line(endLine).to;
          } else {
            // Run the complete logical statement containing the cursor.
            const r = statementRangeAt(state.doc.toString(), sel.head, lang);
            code = state.sliceDoc(r.from, r.to);
            nextPos = r.nextPos;
          }
          editor.dispatch({ selection: { anchor: nextPos }, scrollIntoView: true });
          if (code.trim()) handler(code);
          return true;
        },
      },
      {
        key: "Mod-Shift-Enter",
        preventDefault: true,
        run: () => {
          runRef.current.onRunAll?.();
          return true;
        },
      },
      {
        key: "Mod-Shift-s",
        preventDefault: true,
        run: () => {
          runRef.current.onSource?.();
          return true;
        },
      },
    ]),
  );
}

/**
 * Editor edit-keys, added at Prec.high so they win over basicSetup:
 *  - Mod-/ toggles a line comment in every language. R, Python and SQL all carry
 *    a commentTokens line token, so `toggleComment` knows the prefix. Ours runs
 *    first and returns true, so basicSetup's default Mod-/ never double-toggles.
 *  - Mod-Shift-m inserts the native R pipe ` |> ` in R only. On Python and SQL it
 *    returns false (no preventDefault) so the key falls through to basicSetup's
 *    lint-panel binding exactly as before. It is R-gated the same way runKeymap
 *    derives its language.
 */
function editorKeymap(
  view: LoadedEditor["view"],
  cmState: LoadedEditor["state"],
  commands: LoadedEditor["commands"],
  languageId: string,
): Extension {
  const isR = languageId === "r";
  return cmState.Prec.high(
    view.keymap.of([
      {
        key: "Mod-/",
        preventDefault: true,
        run: commands.toggleComment,
      },
      {
        key: "Mod-Shift-m",
        // No preventDefault: on non-R we must return false AND let the event fall
        // through to basicSetup's lint-panel binding, which requires that this
        // binding not mark the event prevented.
        run: (editor) => {
          if (!isR) return false;
          const sel = editor.state.selection.main;
          const edit = buildPipeInsertion({ from: sel.from, to: sel.to });
          editor.dispatch(
            editor.state.update({
              changes: { from: edit.from, to: edit.to, insert: edit.insert },
              selection: { anchor: edit.anchor },
              scrollIntoView: true,
              userEvent: "input.pipe",
            }),
          );
          return true;
        },
      },
    ]),
  );
}

/**
 * Alt+Shift+L (Option+Shift+L on macOS) announces "Line X of N, column Y"
 * through the app's shared live region, plus any lint problems on that line,
 * since the gutter's numbers and the hover-only lint tooltips are not
 * available to screen readers (#21). Read-only: the caret does not move.
 */
function caretKeymap(
  view: LoadedEditor["view"],
  cmState: LoadedEditor["state"],
  lintMod: LoadedEditor["lint"],
): Extension {
  return cmState.Prec.high(
    view.keymap.of([
      {
        key: CARET_POSITION_KEY,
        preventDefault: true,
        run: (editor) => {
          const state = editor.state;
          const line = state.doc.lineAt(state.selection.main.head);
          const problems: string[] = [];
          lintMod.forEachDiagnostic(state, (d, from, to) => {
            if (from <= line.to && to >= line.from) {
              const kind = d.severity === "error" ? "Error" : "Warning";
              problems.push(`${kind}: ${d.message}`);
            }
          });
          announce(caretAnnouncement(caretOf(state), problems));
          return true;
        },
      },
    ]),
  );
}

type HelpLang = "r" | "python" | "sql";
type SymbolAt = LoadedEditor["helpDocs"]["symbolAt"];

function toHelpLang(languageId: string): HelpLang {
  return languageId === "python" ? "python" : languageId === "sql" ? "sql" : "r";
}

/**
 * Ctrl+Click (Windows/Linux) or Cmd+Click (macOS) on a symbol opens its docs in
 * the HELP tab. `preventDefault()` stops CodeMirror moving the caret, and we
 * never focus or scroll, so the script position and cursor are preserved.
 */
function helpMouse(
  view: LoadedEditor["view"],
  helpRef: { current: ((req: HelpRequest) => void) | undefined },
  symbolAt: SymbolAt,
  languageId: string,
): Extension {
  const lang = toHelpLang(languageId);
  return view.EditorView.domEventHandlers({
    mousedown(event, editor) {
      if (!(event.metaKey || event.ctrlKey)) return false;
      if (event.button !== 0) return false;
      const pos = editor.posAtCoords({ x: event.clientX, y: event.clientY });
      if (pos == null) return false;
      const req = symbolAt(editor.state.doc.toString(), pos, lang);
      // Prevent the caret move and text selection whether or not we found a
      // symbol, so a modified click never disturbs the cursor.
      event.preventDefault();
      if (req) helpRef.current?.(req);
      return true;
    },
  });
}

/**
 * F1 opens docs for the symbol at the cursor, the keyboard equivalent of a
 * modified click. It reads the caret position and does not move it.
 */
function helpKeymap(
  view: LoadedEditor["view"],
  cmState: LoadedEditor["state"],
  helpRef: { current: ((req: HelpRequest) => void) | undefined },
  symbolAt: SymbolAt,
  languageId: string,
): Extension {
  const lang = toHelpLang(languageId);
  return cmState.Prec.high(
    view.keymap.of([
      {
        key: "F1",
        preventDefault: true,
        run: (editor) => {
          const handler = helpRef.current;
          if (!handler) return false;
          const pos = editor.state.selection.main.head;
          const req = symbolAt(editor.state.doc.toString(), pos, lang);
          if (req) handler(req);
          return true;
        },
      },
    ]),
  );
}

/** Per-language indentation on Enter. `indentUnit` is four spaces for all three
 *  languages so indentation is never a tab and Python is a consistent 4-space
 *  policy. R and SQL add a custom indentService backed by the pure column helpers;
 *  Python relies on the @lezer/python tree indentation already in `python()`. */
function indentExtensions(
  lang: LoadedEditor["lang"],
  langStructure: LoadedEditor["langStructure"],
  languageId: string,
): Extension[] {
  const exts: Extension[] = [lang.indentUnit.of("    ")];
  if (languageId === "r") {
    exts.push(
      lang.indentService.of((cx, pos) =>
        langStructure.rIndentColumns(cx.state.doc.toString(), pos),
      ),
    );
  } else if (languageId === "sql") {
    exts.push(
      lang.indentService.of((cx, pos) =>
        langStructure.sqlIndentColumns(cx.state.doc.toString(), pos),
      ),
    );
  }
  return exts;
}

/** An unobtrusive, debounced linter: underlines obvious problems (R bracket/quote
 *  balance; Python and SQL parser error nodes; Python tab/space mixing). It only
 *  produces diagnostics, so it never edits the document, never moves the cursor, and
 *  needs no undo. Heavier truth surfaces in the console on execute. */
function lintExtension(
  lintMod: LoadedEditor["lint"],
  langStructure: LoadedEditor["langStructure"],
  languageId: string,
): Extension {
  const lang: "r" | "python" | "sql" =
    languageId === "python" ? "python" : languageId === "sql" ? "sql" : "r";
  return lintMod.linter(
    (view) =>
      langStructure.lintProblems(view.state.doc.toString(), lang).map((p) => ({
        from: p.from,
        to: p.to,
        severity: p.severity,
        message: p.message,
      })),
    { delay: 400 },
  );
}

/** The editor's look: a light theme (CodeMirror's default highlight, contrast
 * adjusted) or a dark one approximating Tomorrow Night Bright. Every text colour
 * is >=4.5:1 on its editor background, including the active-line band. */
function themeExtensions(
  view: LoadedEditor["view"],
  lang: LoadedEditor["lang"],
  tags: LoadedEditor["tags"],
  dark: boolean,
  fillHeight: boolean,
  languageId: string,
): Extension[] {
  const maxHeight = fillHeight ? "none" : "30rem";
  // In fill-height mode the editor must match its panel so the scroller (which
  // defaults to overflow:auto) scrolls a long or wide script instead of growing
  // past the overflow-hidden wrapper. Inline (capped) mode keeps auto height.
  const height = fillHeight ? "100%" : "auto";
  // R uses a legacy stream grammar that tags every identifier (functions
  // included) as a plain variable, which the default highlight leaves black.
  // Colour identifiers for R so function and package names stand out; the
  // lezer-based Python and SQL grammars already highlight calls, so they are
  // left to the default style.
  const rIdentifiers =
    languageId === "r"
      ? [
          lang.syntaxHighlighting(
            lang.HighlightStyle.define([
              { tag: tags.variableName, color: dark ? "#7aa6da" : "#1f5fa8" },
              {
                tag: tags.function(tags.variableName),
                color: dark ? "#7aa6da" : "#1f5fa8",
              },
            ]),
          ),
        ]
      : [];
  const t = tags;
  if (!dark) {
    // CodeMirror's default highlight colours, measured on the light-tan editor
    // background (#edece2) and kept at >=4.5:1 (WCAG 1.4.3, #13): regexp
    // (#e40, 3.2:1), type (#085, 3.8:1) and invalid (#f00, 3.4:1) are darkened.
    // Registered as a full style rather than relying on the fallback, which
    // CodeMirror drops as soon as any other style (R identifiers) is present,
    // leaving R keywords, strings and comments uncoloured.
    const lightHighlight = lang.HighlightStyle.define([
      { tag: t.meta, color: "#404740" },
      { tag: t.link, textDecoration: "underline" },
      { tag: t.heading, textDecoration: "underline", fontWeight: "bold" },
      { tag: t.emphasis, fontStyle: "italic" },
      { tag: t.strong, fontWeight: "bold" },
      { tag: t.strikethrough, textDecoration: "line-through" },
      { tag: t.keyword, color: "#770088" },
      { tag: [t.atom, t.bool, t.url, t.contentSeparator, t.labelName], color: "#221199" },
      { tag: [t.literal, t.inserted], color: "#116644" },
      { tag: [t.string, t.deleted], color: "#aa1111" },
      { tag: [t.regexp, t.escape, t.special(t.string)], color: "#a63000" },
      { tag: t.definition(t.variableName), color: "#0000ff" },
      { tag: t.local(t.variableName), color: "#3300aa" },
      { tag: [t.typeName, t.namespace], color: "#006b45" },
      { tag: t.className, color: "#116677" },
      { tag: [t.special(t.variableName), t.macroName], color: "#225566" },
      { tag: t.definition(t.propertyName), color: "#0000cc" },
      { tag: t.comment, color: "#994400" },
      { tag: t.invalid, color: "#c41230" },
    ]);
    return [
      view.EditorView.theme({
        "&": { backgroundColor: "transparent", fontSize: "0.875rem", height },
        "&.cm-focused": { outline: "none" },
        ".cm-content": { fontFamily: MONO },
        // Line numbers and fold markers: 5.8:1 on light tan (CodeMirror's
        // #6c6c6c measured 4.4:1); the active line's number is ink on a tan
        // band, 15:1 (#13).
        ".cm-gutters": { backgroundColor: "transparent", color: "#5f5a50", border: "none" },
        ".cm-activeLineGutter": { backgroundColor: "#e2e0d4", color: "#000000" },
        ".cm-foldPlaceholder": {
          backgroundColor: "transparent",
          borderColor: "#ccc9b8",
          color: "#5f5a50",
        },
        // Match highlights as outlines, so token text keeps its contrast.
        ".cm-selectionMatch": { backgroundColor: "transparent", outline: "1px solid #6a8a5a" },
        ".cm-searchMatch-selected": { backgroundColor: "#fde7b0", outline: "1px solid #9a6a00" },
        ".cm-scroller": { maxHeight, overflow: "auto" },
        // Ghost (AI suggestion) text: clearly visible on the light background.
        ".cm-ghost-text": { color: "#6f685c" },
      }),
      lang.syntaxHighlighting(lightHighlight),
      ...rIdentifiers,
    ];
  }
  const darkHighlight = lang.HighlightStyle.define([
    { tag: t.comment, color: "#969896", fontStyle: "italic" },
    { tag: [t.string, t.special(t.string), t.regexp], color: "#b9ca4a" },
    { tag: [t.number, t.bool, t.null, t.atom], color: "#e78c45" },
    { tag: [t.keyword, t.modifier, t.operatorKeyword], color: "#c397d8" },
    {
      tag: [t.function(t.variableName), t.function(t.propertyName)],
      color: "#7aa6da",
    },
    { tag: [t.typeName, t.className, t.namespace], color: "#e7c547" },
    { tag: [t.propertyName, t.attributeName], color: "#7aa6da" },
    // Brightened from #d54e53 (4.4:1 on the active-line band) to 5.6:1 (#13).
    { tag: [t.variableName, t.tagName], color: "#e0686c" },
    {
      tag: [t.operator, t.punctuation, t.separator, t.bracket, t.definition(t.variableName)],
      color: "#eaeaea",
    },
  ]);
  return [
    view.EditorView.theme(
      {
        "&": { backgroundColor: "transparent", color: "#eaeaea", fontSize: "0.875rem", height },
        "&.cm-focused": { outline: "none" },
        ".cm-content": { fontFamily: MONO, caretColor: "#eaeaea" },
        ".cm-cursor, .cm-dropCursor": { borderLeftColor: "#eaeaea" },
        "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection":
          { backgroundColor: "#3a3a3a" },
        // Line numbers and fold markers: the workspace's muted grey, 7:1 on
        // #0a0a0a and 6.4:1 on the active-line band (#5a5a5a measured 2.9:1);
        // the active line's number is the body text colour (#13).
        ".cm-gutters": {
          backgroundColor: "transparent",
          color: "#9a9a9a",
          border: "none",
        },
        ".cm-activeLine": { backgroundColor: "rgba(255,255,255,0.04)" },
        ".cm-activeLineGutter": { backgroundColor: "rgba(255,255,255,0.05)", color: "#eaeaea" },
        ".cm-foldPlaceholder": {
          backgroundColor: "transparent",
          borderColor: "#2c2c2c",
          color: "#9a9a9a",
        },
        // Match highlights as outlines: CodeMirror's bright fills (#99ff77,
        // #00ffff) left light token text near 1.3:1.
        ".cm-selectionMatch": { backgroundColor: "transparent", outline: "1px solid #6a8a5a" },
        ".cm-searchMatch": { backgroundColor: "transparent", outline: "1px solid #7aa6da" },
        ".cm-searchMatch-selected": { backgroundColor: "transparent", outline: "2px solid #e7c547" },
        ".cm-scroller": { maxHeight, overflow: "auto" },
        ".cm-ghost-text": { color: "#8a8a8a" },
      },
      { dark: true },
    ),
    lang.syntaxHighlighting(darkHighlight),
  ];
}
