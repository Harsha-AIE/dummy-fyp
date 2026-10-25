import { AppSidebar } from "@/components/app-sidebar";
import { RoleGate } from "@/components/role-gate";

export default function AppShellLayout({ children }: { children: React.ReactNode }) {
  return (
    <RoleGate>
      <div className="flex h-screen overflow-hidden bg-background">
        <AppSidebar />
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-6xl p-8">{children}</div>
        </main>
      </div>
    </RoleGate>
  );
}
