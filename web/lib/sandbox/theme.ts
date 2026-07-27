/**
 * The Sandbox light/dark choice, persisted in localStorage and read through
 * useSyncExternalStore so it stays client-only (no hydration mismatch) and
 * never sets React state from inside an effect.
 */
export type SandboxTheme = "light" | "dark";

const KEY = "sb-theme";
const listeners = new Set<() => void>();

export function getSandboxTheme(): SandboxTheme {
  if (typeof window === "undefined") return "light";
  return window.localStorage.getItem(KEY) === "dark" ? "dark" : "light";
}

/** Server render (and the first client snapshot) always starts light. */
export function getServerSandboxTheme(): SandboxTheme {
  return "light";
}

export function setSandboxTheme(theme: SandboxTheme): void {
  window.localStorage.setItem(KEY, theme);
  for (const listener of listeners) listener();
}

export function subscribeSandboxTheme(callback: () => void): () => void {
  listeners.add(callback);
  const onStorage = (e: StorageEvent) => {
    if (e.key === KEY) callback();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(callback);
    window.removeEventListener("storage", onStorage);
  };
}
