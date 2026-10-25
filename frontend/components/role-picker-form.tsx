"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Presentation, GraduationCap, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { api, ApiError } from "@/lib/api";
import { getStudentId } from "@/lib/student-id";
import { setIdentity, type Role } from "@/lib/role";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Wordmark } from "@/components/wordmark";
import { cn } from "@/lib/utils";

export function RolePickerForm() {
  const [role, setRole] = useState<Role | null>(null);
  const [name, setName] = useState("");

  const register = useMutation({
    mutationFn: () => {
      const studentId = getStudentId();
      if (!studentId || !role) throw new Error("missing identity or role");
      return api.registerProfile(studentId, role, name.trim());
    },
    // Only persist the role locally once the backend upsert actually
    // succeeds — persisting it unconditionally would let a browser believe
    // it's "logged in" while never appearing in a teacher's roster, which
    // reads from Postgres, not localStorage.
    onSuccess: () => {
      if (role) setIdentity(role, name.trim());
    },
    onError: (err) => toast.error(err instanceof ApiError ? err.message : "Couldn't save your role — try again."),
  });

  const canSubmit = role !== null && name.trim().length > 0 && !register.isPending;

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-6">
      <Card className="w-full max-w-md">
        <CardHeader className="items-center text-center">
          <Wordmark />
          <CardTitle className="mt-2">Who&apos;s using eduKG?</CardTitle>
          <CardDescription>
            No password needed — this just personalizes what you see. You can switch roles anytime from the
            &quot;Switch&quot; link at the bottom of the sidebar.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={() => setRole("teacher")}
              className={cn(
                "flex flex-col items-center gap-2 rounded-lg border-2 p-4 text-sm font-medium transition-colors",
                role === "teacher" ? "border-primary bg-primary/5" : "border-border hover:bg-accent",
              )}
            >
              <Presentation className="size-6" />
              Teacher
            </button>
            <button
              type="button"
              onClick={() => setRole("student")}
              className={cn(
                "flex flex-col items-center gap-2 rounded-lg border-2 p-4 text-sm font-medium transition-colors",
                role === "student" ? "border-primary bg-primary/5" : "border-border hover:bg-accent",
              )}
            >
              <GraduationCap className="size-6" />
              Student
            </button>
          </div>

          <div className="space-y-1.5">
            <label htmlFor="display-name" className="text-xs font-medium text-muted-foreground">
              Your name
            </label>
            <Input
              id="display-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && canSubmit) register.mutate();
              }}
            />
          </div>

          <Button className="w-full" disabled={!canSubmit} onClick={() => register.mutate()}>
            {register.isPending ? <Loader2 className="size-4 animate-spin" /> : null}
            Continue
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
