"use client";

import { useState } from "react";
import Link from "next/link";
import { X } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function AnnouncementBar() {
  const [dismissed, setDismissed] = useState(false);
  if (dismissed) return null;

  return (
    <div className="flex items-center justify-center gap-3 bg-accent px-4 py-2.5 text-sm">
      <span className="flex items-center gap-1.5 font-medium">
        <span className="size-1.5 rounded-full bg-primary" />
        NEW
      </span>
      <span className="hidden text-muted-foreground sm:inline">Two agents are live — Knowledge Manager &amp; Pedagogical Guide</span>
      <Link href="/demo" className={cn(buttonVariants({ size: "sm" }), "h-7 rounded-full px-3 text-xs")}>
        Try the demo
      </Link>
      <button
        onClick={() => setDismissed(true)}
        aria-label="Dismiss"
        className="ml-1 rounded-full p-1 text-muted-foreground hover:bg-black/5 hover:text-foreground"
      >
        <X className="size-3.5" />
      </button>
    </div>
  );
}
