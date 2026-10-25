import { useSyncExternalStore } from "react";

const ROLE_KEY = "edukg_role";
const NAME_KEY = "edukg_display_name";
const CHANGE_EVENT = "edukg-identity-changed";

export type Role = "teacher" | "student";

/**
 * The Teacher/Student role picker's persisted identity — deliberately a
 * separate module from student-id.ts, not an extension of it. That file's
 * whole design (no setter, "generates once and reuses it") encodes an
 * invariant that the student id never changes after first read; role and
 * display name are the opposite — unset until an explicit user action
 * (the picker form), so they need a real setter.
 */
export function getRole(): Role | null {
  if (typeof window === "undefined") return null;
  try {
    const value = window.localStorage.getItem(ROLE_KEY);
    return value === "teacher" || value === "student" ? value : null;
  } catch {
    return null;
  }
}

export function getDisplayName(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(NAME_KEY);
  } catch {
    return null;
  }
}

/**
 * Persists the picked role + name, then fires a real event so every
 * useRole()/useDisplayName() consumer re-renders immediately — unlike
 * student-id.ts's permanent no-op subscribe (valid there only because that
 * value never changes post-mount), role/displayName change exactly once at
 * runtime (the moment the picker submits), and every mounted consumer
 * (RoleGate, the sidebar's nav filter, any teacher-only page guard) needs
 * to see that change without a full page reload. The native `storage`
 * event doesn't fire in the tab that made the change — only other tabs —
 * so a custom window event is used instead.
 */
export function setIdentity(role: Role, displayName: string): void {
  try {
    window.localStorage.setItem(ROLE_KEY, role);
    window.localStorage.setItem(NAME_KEY, displayName);
  } catch {
    // Same silent-degrade convention as student-id.ts — private browsing /
    // blocked storage means the role simply won't persist across reloads.
  }
  window.dispatchEvent(new Event(CHANGE_EVENT));
}

/**
 * Clears the picked role + name so RoleGate shows the picker again —
 * "switch role" / "log out", not account deletion: the underlying
 * student id (student-id.ts) is untouched, so switching back to the same
 * role+name re-attaches to the same mastery history.
 */
export function clearIdentity(): void {
  try {
    window.localStorage.removeItem(ROLE_KEY);
    window.localStorage.removeItem(NAME_KEY);
  } catch {
    // Same silent-degrade convention as elsewhere in this file.
  }
  window.dispatchEvent(new Event(CHANGE_EVENT));
}

function subscribe(onStoreChange: () => void): () => void {
  window.addEventListener(CHANGE_EVENT, onStoreChange);
  return () => window.removeEventListener(CHANGE_EVENT, onStoreChange);
}

const SERVER_SNAPSHOT = () => null;

export function useRole(): Role | null {
  return useSyncExternalStore(subscribe, getRole, SERVER_SNAPSHOT);
}

export function useDisplayName(): string | null {
  return useSyncExternalStore(subscribe, getDisplayName, SERVER_SNAPSHOT);
}
