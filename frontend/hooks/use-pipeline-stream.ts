"use client";

import { useCallback, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { PipelineRunStatus, PipelineStageName } from "@/lib/types";

export type StageStatus = "pending" | "started" | "completed" | "failed" | "skipped";

export interface PipelineStreamState {
  runId: string | null;
  stageStatuses: Partial<Record<PipelineStageName, StageStatus>>;
  isRunning: boolean;
  finalStatus: PipelineRunStatus | null;
  streamError: string | null;
}

const initialState: PipelineStreamState = {
  runId: null,
  stageStatuses: {},
  isRunning: false,
  finalStatus: null,
  streamError: null,
};

export function usePipelineStream() {
  const [state, setState] = useState<PipelineStreamState>(initialState);
  const eventSourceRef = useRef<EventSource | null>(null);

  const start = useCallback((runId: string) => {
    eventSourceRef.current?.close();
    setState({ runId, stageStatuses: {}, isRunning: true, finalStatus: null, streamError: null });

    const source = new EventSource(api.pipelineRunStreamUrl(runId));
    eventSourceRef.current = source;

    source.addEventListener("stage", (event: MessageEvent<string>) => {
      const payload = JSON.parse(event.data) as { stage: PipelineStageName; status: StageStatus };
      setState((prev) => ({
        ...prev,
        stageStatuses: { ...prev.stageStatuses, [payload.stage]: payload.status },
      }));
    });

    source.addEventListener("done", (event: MessageEvent<string>) => {
      const finalStatus = JSON.parse(event.data) as PipelineRunStatus;
      setState((prev) => ({ ...prev, isRunning: false, finalStatus }));
      source.close();
    });

    source.onerror = () => {
      source.close();
      // The live feed dropping doesn't mean the run itself died (network
      // blip, dev-server restart, CORS) — getPipelineRun exists precisely
      // for this and was previously never called anywhere, so a dropped
      // connection just went silent forever with no recovery path. Try it
      // once: if the run actually finished, recover the real result instead
      // of leaving the UI showing whatever stage was last marked "started".
      api
        .getPipelineRun(runId)
        .then((status) => {
          if (status.state === "running") {
            setState((prev) => ({ ...prev, isRunning: false, streamError: "Live updates disconnected — the run may still be in progress." }));
          } else {
            setState((prev) => ({ ...prev, isRunning: false, finalStatus: status }));
          }
        })
        .catch(() => {
          setState((prev) => ({ ...prev, isRunning: false, streamError: "Live updates disconnected and the run's status couldn't be checked." }));
        });
    };
  }, []);

  const reset = useCallback(() => {
    eventSourceRef.current?.close();
    setState(initialState);
  }, []);

  return { ...state, start, reset };
}
