import { useEffect, useState } from "react";

import type { JobSpec } from "./types";

export interface Estimates {
  seconds: number | null;
  variants: (number | null)[];
  learnedFrom: number;
}

/** One request labels the current settings and every alternative on screen. */
export function useEstimates(
  spec: JobSpec,
  variants: Partial<JobSpec>[],
): Estimates {
  const [estimates, setEstimates] = useState<Estimates>({
    seconds: null,
    variants: [],
    learnedFrom: 0,
  });
  const key = JSON.stringify([spec, variants]);

  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(() => {
      fetch("/api/jobs/estimate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spec, variants }),
        signal: controller.signal,
      })
        .then((response) => (response.ok ? response.json() : null))
        .then((body) => {
          if (!body) return;
          setEstimates({
            seconds: body.seconds ?? null,
            variants: (body.variants ?? []).map(
              (variant: { seconds?: number }) => variant.seconds ?? null,
            ),
            learnedFrom: body.learned_from ?? 0,
          });
        })
        .catch(() => undefined);
    }, 250);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
    // `key` is the serialised request: it changes exactly when the answer would.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return estimates;
}
