import Link from "next/link";
import { Compass } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";

// Next.js App Router convention: renders for any unmatched route under this
// layout, instead of the framework's default unstyled 404 screen.
export default function AppNotFound() {
  return (
    <div className="flex h-full min-h-[60vh] flex-col items-center justify-center gap-3 p-6 text-center">
      <Compass className="size-8 text-muted-foreground" />
      <p className="font-medium">Page not found.</p>
      <p className="max-w-md text-sm text-muted-foreground">
        There&apos;s nothing at this address. Use the sidebar to get back to a real page.
      </p>
      <Link href="/study" className={buttonVariants({ variant: "outline", size: "sm", className: "mt-2" })}>
        Back to Study
      </Link>
    </div>
  );
}
