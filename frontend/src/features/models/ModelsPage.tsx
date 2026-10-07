import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";

import { api } from "@/api/endpoints";
import type { ModelVersion } from "@/api/schemas";
import { Button, Card, CardTitle, ErrorBox, PageHeader, Spinner } from "@/components/ui";
import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";

const METRICS: Array<[string, string, "higher" | "lower", (v: number) => string]> = [
  ["roc_auc", "ROC-AUC", "higher", (v) => v.toFixed(4)],
  ["pr_auc", "PR-AUC", "higher", (v) => v.toFixed(4)],
  ["ks", "KS", "higher", (v) => v.toFixed(4)],
  ["brier", "Brier", "lower", (v) => v.toFixed(4)],
  ["expected_cost", "Expected cost", "lower", (v) => v.toLocaleString(undefined, { maximumFractionDigits: 0 })],
  ["loss_saved_pct", "Loss saved vs. approve-all", "higher", (v) => `${(v * 100).toFixed(1)}%`],
  ["recall_at_1pct_fpr", "Recall @ 1% FPR", "higher", (v) => v.toFixed(4)],
  ["fairness_max_dp_gap", "Max approval-rate gap", "lower", (v) => v.toFixed(4)],
  ["oot_roc_auc", "Out-of-time AUC", "higher", (v) => v.toFixed(4)],
];

function Comparison({ versions }: { versions: ModelVersion[] }) {
  const champion = versions.find((v) => v.aliases.includes("champion"));
  const challenger = versions.find((v) => v.aliases.includes("challenger"));
  const cols = [champion, challenger, ...versions.filter((v) => v !== champion && v !== challenger).slice(0, 2)].filter(
    Boolean,
  ) as ModelVersion[];
  if (!cols.length) return <p className="text-sm text-ink-muted">No registered versions yet — run <code>make train</code>.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-ink-2">
          <tr className="border-b border-[var(--border)]">
            <th className="py-2 pr-4 font-medium">Metric</th>
            {cols.map((v) => (
              <th key={v.version} className="py-2 pr-4 text-right font-medium">
                v{v.version} {v.aliases.map((a) => <span key={a} className="ml-1 rounded bg-[var(--gridline)] px-1.5 py-0.5 text-[10px] uppercase text-ink">{a}</span>)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="tabular">
          {METRICS.filter(([k]) => cols.some((c) => c.metrics[k] !== undefined)).map(([key, label, better, fmt]) => {
            const champ = champion?.metrics[key];
            return (
              <tr key={key} className="border-b border-[var(--border)]">
                <td className="py-2 pr-4 text-ink-2">{label}</td>
                {cols.map((v) => {
                  const val = v.metrics[key];
                  const delta = champ !== undefined && val !== undefined && v !== champion ? val - champ : null;
                  const good = delta !== null && (better === "higher" ? delta > 0 : delta < 0);
                  return (
                    <td key={v.version} className="py-2 pr-4 text-right text-ink">
                      {val === undefined ? "—" : fmt(val)}
                      {delta !== null && delta !== 0 && (
                        <span className={cn("ml-1 text-xs", good ? "text-[var(--success-text)]" : "text-[var(--status-critical)]")}>
                          {good ? "▲" : "▼"} {Math.abs(delta) < 1 ? Math.abs(delta).toFixed(4) : Math.abs(delta).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function ModelsPage() {
  const { can } = useAuth();
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const reload = useMutation({ mutationFn: api.reloadModels, onSuccess: () => qc.invalidateQueries({ queryKey: ["models"] }) });

  return (
    <div>
      <PageHeader
        title="Model registry"
        subtitle="Champion vs. challenger from the MLflow Model Registry. Promotion = better AUC AND no worse expected cost AND fairness gap within limits, then human approval."
        actions={can("admin") && (
          <Button variant="secondary" onClick={() => reload.mutate()} loading={reload.isPending}>
            <RefreshCw className="h-4 w-4" aria-hidden /> Hot-reload serving
          </Button>
        )}
      />
      {models.isLoading && <Spinner />}
      {models.error && <ErrorBox error={models.error} />}
      {reload.error && <ErrorBox error={reload.error} />}
      {models.data && (
        <div className="space-y-4">
          {Object.entries(models.data.registry).map(([task, reg]) => {
            const serving = models.data.serving[task] as { version?: string; loaded?: boolean } | undefined;
            return (
              <Card key={task}>
                <CardTitle hint={serving?.loaded ? `serving v${serving.version}` : "not loaded in serving"}>
                  {task === "credit" ? "Credit risk" : "Fraud"} · {reg.name}
                </CardTitle>
                {reg.error ? <ErrorBox error={reg.error} /> : <Comparison versions={reg.versions} />}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
