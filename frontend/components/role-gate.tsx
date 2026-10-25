"use client";

import { useRole, useDisplayName } from "@/lib/role";
import { RolePickerForm } from "@/components/role-picker-form";

export function RoleGate({ children }: { children: React.ReactNode }) {
  const role = useRole();
  const displayName = useDisplayName();

  if (!role || !displayName) {
    return <RolePickerForm />;
  }
  return <>{children}</>;
}
