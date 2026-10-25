"use client";

import { cn } from "@/lib/utils";
import { useQuery } from "@tanstack/react-query";
import {
  LayoutDashboard,
  MessageSquareText,
  Network,
  ClipboardCheck,
  Workflow,
  Gauge,
  CircleDot,
  ArrowLeft,
  Users,
  LogOut,
  BookOpen,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { api } from "@/lib/api";
import { useRole, useDisplayName, clearIdentity } from "@/lib/role";
import { Wordmark } from "@/components/wordmark";

const NAV_ITEMS = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard, teacherOnly: true },
  { href: "/demo", label: "Live Agent Demo", icon: MessageSquareText, teacherOnly: false },
  { href: "/study", label: "Study", icon: BookOpen, teacherOnly: false },
  { href: "/graph", label: "Knowledge Graph", icon: Network, teacherOnly: false },
  { href: "/mastery", label: "Knowledge Tracing", icon: Gauge, teacherOnly: false },
  { href: "/class", label: "Class Overview", icon: Users, teacherOnly: true },
  { href: "/review", label: "Review Console", icon: ClipboardCheck, teacherOnly: true },
  { href: "/pipeline", label: "Pipeline Runs", icon: Workflow, teacherOnly: true },
];

export function AppSidebar() {
  const pathname = usePathname();
  const role = useRole();
  const displayName = useDisplayName();
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15_000,
    retry: 0,
  });

  const items = NAV_ITEMS.filter((item) => !item.teacherOnly || role === "teacher");

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r bg-sidebar text-sidebar-foreground">
      <Link href="/" className="flex h-16 flex-col justify-center gap-0.5 border-b px-6 transition-colors hover:bg-sidebar-accent/40">
        <Wordmark />
        <p className="text-xs text-muted-foreground">Knowledge Graph Tutor</p>
      </Link>

      <nav className="flex-1 space-y-1 p-3">
        {items.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "text-sidebar-foreground/65 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground",
              )}
            >
              <Icon className={cn("size-4", isActive && "text-primary")} />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="space-y-2 border-t p-4">
        <Link href="/" className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground">
          <ArrowLeft className="size-3" /> Back to overview
        </Link>
        {displayName && role && (
          <div className="flex items-center justify-between gap-2">
            <p className="truncate text-xs text-muted-foreground">
              {displayName} <span className="capitalize">({role})</span>
            </p>
            <button
              type="button"
              onClick={clearIdentity}
              className="flex shrink-0 items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
              title="Switch role"
            >
              <LogOut className="size-3" /> Switch
            </button>
          </div>
        )}
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <CircleDot className={cn("size-3", health?.status === "ok" ? "text-emerald-500" : "text-destructive")} />
          {health?.status === "ok" ? "API connected" : "API unreachable"}
        </div>
      </div>
    </aside>
  );
}
