import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Play } from "lucide-react";
import { useMemo, useState } from "react";

import { api } from "@/api/endpoints";
import type { DriftReport } from "@/api/schemas";
import { StatusBadge } from "@/components/badges";
import { PsiBars, PsiOverTime } from "@/components/charts";
import { Button, Card, CardTitle, ErrorBox, PageHeader, Select, Spinner } from "@/components/ui";
import { useAuth } from "@/lib/auth";
import { formatDate } from "@/lib/utils";

export function psiSeries(reports: DriftReport[]) {
  const usable = reports.filter((r) => r.status === "ok" || r.status === "drift").slice().reverse(); // oldest first
  const latest = usable[usable.length - 1];
  const top = latest
    ? Object.entries(latest.feature_psi).sort((a, b) => b[1] - a[1]).map(([f]) => f).slice(0, 3)
    : [];
  const points: Array<Record<string, number | string>> = usable.map((r) => ({
    date: new Date(r.window_end).toLocaleDateString(undefined, { month: "short", day: "numeric" }),
    ...Object.fromEntries(top.map((f) => [f, r.feature_psi[f] ?? 0])),
  }));
  return { points, top, latest };
}

export function MonitoringPage() {
  const [task, setTask] = useState("credit");
  const { can } = useAuth();
  const qc = useQueryClient();
  const drift = useQuery({ queryKey: ["drift", task], queryFn: () => api.drift(task) });
  const run = useMutation({ mutationFn: () => api.runDrift(task, 24 * 30), onSuccess: () => qc.invalidateQueries({ queryKey: ["drift"] }) });
  const { points, top, latest } = useMemo(() => psiSeries(drift.data?.results ?? []), [drift.data]);

  return (
    <div>
      <PageHeader
        title="Monitoring"
        subtitle="Population Stability Index of production inputs vs. the training reference. PSI > 0.2 = significant shift (banking standard)."
        actions={
          <>
            <Select value={task} onChange={(e) => setTask(e.target.value)} className="w-36" aria-label="Model">
              <option value="credit">Credit</option>
              <option value="fraud">Fraud</option>
            </Select>
            {can("admin") && (
              <Button variant="secondary" onClick={() => run.mutate()} loading={run.isPending}>
                <Play className="h-4 w-4" aria-hidden /> Run drift check
              </Button>
            )}
          </>
        }
      />
      {drift.isLoading && <Spinner />}
      {drift.error && <ErrorBox error={drift.error} />}
      {run.error && <ErrorBox error={run.error} />}
      {drift.data && (
        <div className="grid gap-4 xl:grid-cols-2">
          <Card className="xl:col-span-2">
            <CardTitle hint="top 3 features by latest PSI">PSI over time</CardTitle>
            {points.length >= 2 ? <PsiOverTime points={points} features={top} /> : points.length === 1 ? (
              <p className="text-sm text-ink-muted">One report so far — the trend appears after the next nightly run. Latest values are below.</p>
            ) : (
              <p className="text-sm text-ink-muted">No drift reports yet. The nightly Celery job creates one every day (or run it now).</p>
            )}
          </Card>
          {latest && (
            <Card>
              <CardTitle hint={`${latest.n_samples} predictions · ${formatDate(latest.window_end)}`}>Latest PSI per feature</CardTitle>
              <PsiBars psi={latest.feature_psi} />
              {latest.prediction_psi != null && (
                <p className="mt-2 text-xs text-ink-2">Prediction (score) PSI: <strong className="text-ink">{latest.prediction_psi.toFixed(3)}</strong></p>
              )}
            </Card>
          )}
          <Card>
            <CardTitle>Reports</CardTitle>
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-ink-2">
                <tr className="border-b border-[var(--border)]">
                  <th className="py-2 font-medium">Window end</th><th className="py-2 font-medium">Model</th>
                  <th className="py-2 text-right font-medium">n</th><th className="py-2 text-right font-medium">Max PSI</th>
                  <th className="py-2 pl-3 font-medium">Status</th><th className="py-2 font-medium"></th>
                </tr>
              </thead>
              <tbody className="tabular">
                {drift.data.results.map((r) => (
                  <tr key={r.id} className="border-b border-[var(--border)]">
                    <td className="py-2 text-xs text-ink-2">{formatDate(r.window_end)}</td>
                    <td className="py-2">{r.model_version ? `v${r.model_version}` : "—"}</td>
                    <td className="py-2 text-right">{r.n_samples}</td>
                    <td className="py-2 text-right">{r.max_psi?.toFixed(3) ?? "—"}</td>
                    <td className="py-2 pl-3"><StatusBadge status={r.status} />{r.retrain_triggered && <span className="ml-1 text-xs text-ink-muted">retrain ↻</span>}</td>
                    <td className="py-2 text-right">{r.report_url && <a className="text-xs text-brand hover:underline" href={r.report_url} target="_blank" rel="noreferrer">Evidently</a>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
    </div>
  );
}
