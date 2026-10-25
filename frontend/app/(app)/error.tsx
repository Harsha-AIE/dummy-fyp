"use client";

import { useEffect } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";

// Next.js App Router convention: catches any uncaught render-time exception
// thrown by a page/component under this route group and renders this
// instead of the framework's default unstyled error screen. Lives inside
// the (app) layout, so the sidebar stays visible — only the page content
// area is replaced.
export default function AppError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex h-full min-h-[60vh] items-center justify-center p-6">
      <div className="flex max-w-md flex-col items-center gap-3 rounded-lg border border-destructive/40 bg-destructive/5 px-6 py-10 text-center">
        <AlertTriangle className="size-8 text-destructive" />
        <p className="font-medium text-destructive">Something went wrong on this page.</p>
        <p className="text-sm text-destructive/80">{error.message || "An unexpected error occurred."}</p>
        <Button size="sm" variant="outline" onClick={() => reset()} className="mt-2">
          <RotateCcw className="size-3.5" /> Try again
        </Button>
      </div>
    </div>
  );
}
