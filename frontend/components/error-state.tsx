import { AlertTriangle } from "lucide-react";
import { ApiError } from "@/lib/api";

/**
 * Shared failure state for any page backed by a useQuery that can fail —
 * distinct from an empty-but-successful result. A query's own data being
 * undefined on error looks identical to "there's genuinely nothing here"
 * unless a caller checks isError explicitly, which several pages (the
 * Review Console in particular) didn't: an API outage rendered the exact
 * same "0 pending" / empty-queue message a reviewer would see if the queue
 * were actually clear. Surfaces the real ApiError message (CORS, 500,
 * network) instead of a generic string, so the cause is visible without
 * opening devtools.
 */
export function ErrorState({ error, label }: { error: unknown; label?: string }) {
  const message = error instanceof ApiError ? error.message : "Couldn't reach the API — check that the backend is running.";
  return (
    <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-destructive/40 bg-destructive/5 py-16 text-center">
      <AlertTriangle className="size-8 text-destructive" />
      {label && <p className="font-medium text-destructive">{label}</p>}
      <p className="max-w-md text-sm text-destructive/80">{message}</p>
    </div>
  );
}
