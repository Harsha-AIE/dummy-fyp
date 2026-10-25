import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export function StatCard({
  title,
  value,
  icon: Icon,
  loading,
  accent,
}: {
  title: string;
  value: number | string;
  icon: LucideIcon;
  loading?: boolean;
  accent?: "default" | "amber" | "emerald";
}) {
  return (
    <Card className="card-hover">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="font-sans text-sm font-medium text-muted-foreground">{title}</CardTitle>
        <div
          className={cn(
            "flex size-7 items-center justify-center rounded-md",
            accent === "amber" && "bg-amber-500/15 text-amber-500",
            accent === "emerald" && "bg-emerald-500/15 text-emerald-500",
            (!accent || accent === "default") && "bg-muted text-muted-foreground",
          )}
        >
          <Icon className="size-4" />
        </div>
      </CardHeader>
      <CardContent>
        {loading ? <Skeleton className="h-8 w-16" /> : <div className="font-heading text-3xl font-semibold">{value}</div>}
      </CardContent>
    </Card>
  );
}
