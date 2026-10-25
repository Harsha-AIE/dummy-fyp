"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { Users } from "lucide-react";
import { api } from "@/lib/api";
import { useRole } from "@/lib/role";
import type { RosterEntryOut } from "@/lib/types";
import { Card, CardContent } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { ErrorState } from "@/components/error-state";
import { TeacherOnlyNotice } from "@/components/teacher-only";

export default function ClassOverviewPage() {
  const role = useRole();
  const { data, isLoading, isError, error } = useQuery({ queryKey: ["roster"], queryFn: api.roster });

  if (role !== "teacher") return <TeacherOnlyNotice />;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-heading text-3xl font-semibold tracking-tight">Class Overview</h1>
        <p className="text-muted-foreground">Every student who has picked the Student role, with their mastery so far.</p>
      </div>

      {isLoading && <Skeleton className="h-64 w-full" />}
      {isError && <ErrorState error={error} label="Couldn't load the class roster." />}
      {data && data.students.length === 0 && (
        <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed py-16 text-center text-muted-foreground">
          <Users className="size-8" />
          <p>No students yet — once someone picks &quot;Student&quot; on this app, they&apos;ll show up here.</p>
        </div>
      )}
      {data && data.students.length > 0 && (
        <Card>
          <CardContent className="pt-6">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Concepts engaged</TableHead>
                  <TableHead>Average mastery</TableHead>
                  <TableHead>Last active</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.students.map((s: RosterEntryOut) => (
                  <TableRow key={s.student_id}>
                    <TableCell className="font-medium">
                      <Link
                        href={`/mastery?student=${s.student_id}&name=${encodeURIComponent(s.display_name)}`}
                        className="hover:underline"
                      >
                        {s.display_name}
                      </Link>
                    </TableCell>
                    <TableCell>{s.concept_count}</TableCell>
                    <TableCell className="w-40">
                      {s.average_mastery === null ? (
                        <span className="text-xs text-muted-foreground">no activity yet</span>
                      ) : (
                        <div className="flex items-center gap-2">
                          <Progress value={s.average_mastery * 100} className="w-16" />
                          <span className="text-xs text-muted-foreground">{Math.round(s.average_mastery * 100)}%</span>
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {s.last_active ? new Date(s.last_active).toLocaleString() : "—"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
