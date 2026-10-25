import { ShieldAlert } from "lucide-react";

/**
 * Shown instead of a page's real content when a Student navigates directly
 * to a teacher-only URL (Dashboard, Review Console, Pipeline Runs, Class
 * Overview) — the sidebar already hides these links, but hiding a link
 * doesn't stop a direct visit. This is UI-only enforcement, matching the
 * project's explicitly-agreed non-secure role model (no passwords, no
 * server-side authorization) — see lib/role.ts.
 */
export function TeacherOnlyNotice() {
  return (
    <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed py-16 text-center text-muted-foreground">
      <ShieldAlert className="size-8" />
      <p>This page is only available to teachers.</p>
    </div>
  );
}
