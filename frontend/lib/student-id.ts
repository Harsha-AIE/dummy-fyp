import { useSyncExternalStore } from "react";

const STORAGE_KEY = "edukg_student_id";

/**
 * A lightweight, per-browser demo identity — not auth. There's no login
 * anywhere in this app; Knowledge Tracing needs *some* stable id to key
 * mastery on, so this generates one once and reuses it from localStorage.
 * Returns null during SSR (no localStorage there) and if storage is
 * unavailable (private browsing, blocked site data) — callers treat a null
 * student id as "skip Knowledge Tracing for this request," never as an
 * error.
 */
export function getStudentId(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const existing = window.localStorage.getItem(STORAGE_KEY);
    if (existing) return existing;
    const id = crypto.randomUUID();
    window.localStorage.setItem(STORAGE_KEY, id);
    return id;
  } catch {
    return null;
  }
}

const NOOP_SUBSCRIBE = () => () => {};
const SERVER_SNAPSHOT = () => null;

/**
 * Render-safe version of getStudentId() for use directly in a component
 * body. Reading localStorage during render would return different values
 * on the server (always null) vs. the client (the stored/generated id),
 * causing a React hydration mismatch. useSyncExternalStore is the pattern
 * React itself recommends for exactly this — external (non-React) state
 * that must read consistently across server/client — rather than setState
 * inside a useEffect, which works but costs an extra render pass and trips
 * the react-hooks/set-state-in-effect lint rule. The student id never
 * changes once generated, so the "subscribe" callback has nothing to ever
 * notify and is a permanent no-op.
 */
export function useStudentId(): string | null {
  return useSyncExternalStore(NOOP_SUBSCRIBE, getStudentId, SERVER_SNAPSHOT);
}
