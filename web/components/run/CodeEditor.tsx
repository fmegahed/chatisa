"use client";

import { useEffect, useRef, useState } from "react";
import type { Extension } from "@codemirror/state";
import type { EditorView } from "@codemirror/view";
import type { CompletionSource } from "@/lib/sandbox/inline-completion";
import type { CompletionResult } from "@/lib/run/manager";
import type { HelpRequest } from "@/lib/sandbox/help-docs";
import { buildPipeInsertion } from "@/lib/sandbox/editor-keys";

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
}) {
  const host = useRef<HTMLDivElement | null>(null);
  const viewRef = useRef<EditorView | null>(null);
  // Keep the latest onChange without re-creating the editor on every keystroke.
  const onChangeRef = useRef(props.onChange);
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
  const { dark = false, fillHeight = false } = props;
  const hasCompletion = props.completionSource != null;
  const hasComplete = props.completeSource != null;
  const hasRun = props.onRunAll != null;
  const hasHelp = props.onHelp != null;

  useEffect(() => {
    onChangeRef.current = props.onChange;
  }, [props.onChange]);

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
    loadCodeMirror(props.languageId)
      .then(({ view, cm, lang, state, autocomplete, tags, commands, langExt, inline, langStructure, helpDocs, lint }) => {
        if (cancelled || !host.current) return;
        const editor = new view.EditorView({
          doc: props.value,
          parent: host.current,
          extensions: [
            cm.basicSetup,
            langExt,
            ...indentExtensions(lang, langStructure, props.languageId),
            view.EditorView.updateListener.of((update) => {
              if (update.docChanged) {
                onChangeRef.current(update.state.doc.toString());
              }
            }),
            // The contenteditable gets the accessible name; CodeMirror already
            // gives it role="textbox" and aria-multiline.
            view.EditorView.contentAttributes.of({ "aria-label": props.label }),
            ...themeExtensions(view, lang, tags, dark, fillHeight, props.languageId),
            editorKeymap(view, state, commands, props.languageId),
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
        editor.focus();
        setReady(true);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
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

  return (
    <div className={fillHeight ? "h-full" : undefined}>
      {/* CodeMirror mounts here; kept in the DOM but hidden until ready so the
          editor has a parent to attach to. */}
      <div
        ref={host}
        className={
          usingCodeMirror
            ? `overflow-hidden rounded-card border border-medium-tan ${wrapBg} focus-within:border-miami-red ${fillHeight ? "h-full" : ""}`
            : "hidden"
        }
      />
      {!usingCodeMirror ? (
        <textarea
          aria-label={props.label}
          value={props.value}
          onChange={(e) => props.onChange(e.target.value)}
          spellCheck={false}
          rows={fillHeight ? undefined : rows}
          className={`w-full resize-y rounded-card border border-medium-tan p-3 font-mono text-sm focus:border-miami-red focus:outline-none ${dark ? "bg-[#0a0a0a] text-[#eaeaea]" : "bg-light-tan text-ink"} ${fillHeight ? "h-full resize-none" : ""}`}
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

/** The editor's look: a light theme (CodeMirror's default highlight) or a dark
 * one approximating Tomorrow Night Bright. */
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
  if (!dark) {
    return [
      view.EditorView.theme({
        "&": { backgroundColor: "transparent", fontSize: "0.875rem", height },
        "&.cm-focused": { outline: "none" },
        ".cm-content": { fontFamily: MONO },
        ".cm-gutters": { backgroundColor: "transparent", border: "none" },
        ".cm-scroller": { maxHeight, overflow: "auto" },
        // Ghost (AI suggestion) text: clearly visible on the light background.
        ".cm-ghost-text": { color: "#6f685c" },
      }),
      ...rIdentifiers,
    ];
  }
  const t = tags;
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
    { tag: [t.variableName, t.tagName], color: "#d54e53" },
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
        ".cm-gutters": {
          backgroundColor: "transparent",
          color: "#5a5a5a",
          border: "none",
        },
        ".cm-activeLine": { backgroundColor: "rgba(255,255,255,0.04)" },
        ".cm-activeLineGutter": { backgroundColor: "rgba(255,255,255,0.05)" },
        ".cm-scroller": { maxHeight, overflow: "auto" },
        ".cm-ghost-text": { color: "#8a8a8a" },
      },
      { dark: true },
    ),
    lang.syntaxHighlighting(darkHighlight),
  ];
}
