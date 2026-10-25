import Link from "next/link";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Wordmark } from "@/components/wordmark";
import { AnnouncementBar } from "@/components/announcement-bar";

// Only shared (non-teacherOnly) destinations — a visitor who's already
// picked Student (or picks it after clicking) must not land on a "teachers
// only" dead end from the site's own primary nav. See app-sidebar.tsx's
// NAV_ITEMS for which routes are teacher-only.
const LINKS = [
  { href: "/demo", label: "Live Demo" },
  { href: "/study", label: "Study" },
  { href: "/graph", label: "Knowledge Graph" },
];

export function LandingNav() {
  return (
    <header className="sticky top-0 z-20 border-b border-border/70 bg-background/85 backdrop-blur-md">
      <AnnouncementBar />
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
        <Link href="/">
          <Wordmark />
        </Link>

        <nav className="hidden items-center gap-7 md:flex">
          {LINKS.map((link) => (
            <Link key={link.href} href={link.href} className="text-sm text-muted-foreground transition-colors hover:text-foreground">
              {link.label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-2.5">
          <Link href="/study" className={cn(buttonVariants({ size: "sm", variant: "outline" }), "rounded-full")}>
            Study
          </Link>
          <Link href="/demo" className={cn(buttonVariants({ size: "sm" }), "rounded-full")}>
            Talk to the Agents
          </Link>
        </div>
      </div>
    </header>
  );
}
