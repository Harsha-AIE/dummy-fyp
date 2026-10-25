"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "next-themes";
import { useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // Applied to every query by default — this is a live console
            // over data that changes from other tabs/other people uploading
            // documents, not a static site. Polling here beats patching a
            // refetchInterval onto each individual useQuery call one at a
            // time (and forgetting one). Any screen that genuinely wants a
            // different cadence — or none, e.g. to avoid the force-graph
            // re-laying-out mid-drag — overrides it locally.
            staleTime: 5_000,
            refetchInterval: 8_000,
            refetchOnWindowFocus: true,
            retry: 1,
          },
        },
      }),
  );

  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false}>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </ThemeProvider>
  );
}
