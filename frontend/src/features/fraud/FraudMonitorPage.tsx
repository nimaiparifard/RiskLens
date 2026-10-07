import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pause, Play } from "lucide-react";
import { useState } from "react";

import { api } from "@/api/endpoints";
import type { Transaction } from "@/api/schemas";
import { GradeBadge, StatusBadge } from "@/components/badges";
import { Button, Card, ErrorBox, PageHeader, Spinner, StatTile } from "@/components/ui";
import { useAuth } from "@/lib/auth";
import { cn, formatMoney, formatPct } from "@/lib/utils";

const POLL_MS = 2000;

function TxRow({ tx, onReview, canReview }: { tx: Transaction; onReview: (label: "fraud" | "legit") => void; canReview: boolean }) {
  const top = tx.reasons[0];
  return (
    <tr className={cn("border-b border-[var(--border)]", tx.is_flagged && "bg-[color-mix(in_srgb,var(--status-critical)_10%,transparent)]")}>
      <td className="px-3 py-2 text-xs text-ink-2">{new Date(tx.created_at).toLocaleTimeString()}</td>
      <td className="px-3 py-2">{tx.transaction_id}</td>
      <td className="px-3 py-2">{tx.card1}</td>
      <td className="px-3 py-2 text-right">{formatMoney(tx.amount)}</td>
      <td className="px-3 py-2 text-xs text-ink-2">{tx.product_cd} · {tx.p_emaildomain || "—"} · {tx.device_type || "—"}</td>
      <td className="px-3 py-2 text-right font-medium">{formatPct(tx.probability, 2)}</td>
      <td className="px-3 py-2"><GradeBadge grade={tx.grade} /></td>
      <td className="px-3 py-2"><StatusBadge status={tx.status} /></td>
      <td className="max-w-xs truncate px-3 py-2 text-xs text-ink-2" title={top?.text}>{top?.text ?? "—"}</td>
      <td className="px-3 py-2 text-right text-xs text-ink-muted">{tx.latency_ms?.toFixed(1)} ms</td>
      <td className="px-3 py-2">
        {canReview && tx.is_flagged && !tx.reviewed_by && (
          <div className="flex gap-1">
            <Button size="sm" variant="danger" onClick={() => onReview("fraud")}>Fraud</Button>
            <Button size="sm" variant="secondary" onClick={() => onReview("legit")}>Legit</Button>
          </div>
        )}
        {tx.reviewed_by && <span className="text-xs text-ink-muted">by {tx.reviewed_by}</span>}
      </td>
    </tr>
  );
}

export function FraudMonitorPage() {
  const [live, setLive] = useState(true);
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const { can } = useAuth();
  const qc = useQueryClient();
  const feed = useQuery({
    queryKey: ["transactions", flaggedOnly],
    queryFn: () => api.transactions({ flagged: flaggedOnly ? true : undefined, page_size: 50 }),
    refetchInterval: live ? POLL_MS : false,
  });
  const stats = useQuery({ queryKey: ["txStats"], queryFn: api.txStats, refetchInterval: live ? POLL_MS * 2 : false });
  const review = useMutation({
    mutationFn: (v: { id: number; label: "fraud" | "legit" }) => api.reviewTransaction(v.id, v.label),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["transactions"] }),
  });

  return (
    <div>
      <PageHeader
        title="Fraud monitor"
        subtitle={<>Live card transactions scored in real time. Run <code className="text-xs">make simulate</code> to stream demo traffic.</>}
        actions={
          <>
            <Button variant="secondary" onClick={() => setFlaggedOnly((f) => !f)}>{flaggedOnly ? "Show all" : "Flagged queue"}</Button>
            <Button variant="secondary" onClick={() => setLive((l) => !l)} aria-pressed={live}>
              {live ? <Pause className="h-4 w-4" aria-hidden /> : <Play className="h-4 w-4" aria-hidden />} {live ? "Pause" : "Resume"}
            </Button>
          </>
        }
      />
      <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Transactions scored" value={stats.data?.total ?? "—"} />
        <StatTile label="Flagged (review + block)" value={stats.data?.flagged ?? "—"}
          sub={stats.data?.total ? formatPct(stats.data.flagged / stats.data.total) : undefined} />
        <StatTile label="Confirmed fraud" value={stats.data?.confirmed ?? "—"} />
        <StatTile label="Avg model latency" value={stats.data?.avg_latency_ms != null ? `${stats.data.avg_latency_ms.toFixed(1)} ms` : "—"} sub={`last ${stats.data?.latency_window_minutes ?? 60} min · budget 50 ms`} />
      </div>
      <Card className="p-0">
        {feed.isLoading && <Spinner />}
        {feed.error && <div className="p-3"><ErrorBox error={feed.error} /></div>}
        {review.error && <div className="p-3"><ErrorBox error={review.error} /></div>}
        {feed.data && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-ink-2">
                <tr className="border-b border-[var(--border)]">
                  {["Time", "Tx ID", "Card", "Amount", "Product · e-mail · device", "P(fraud)", "Risk", "Status", "Top reason", "Latency", ""].map((h) => (
                    <th key={h} className={cn("px-3 py-2 font-medium", ["Amount", "P(fraud)", "Latency"].includes(h) && "text-right")}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="tabular">
                {feed.data.results.map((tx) => (
                  <TxRow key={tx.id} tx={tx} canReview={can("decide")} onReview={(label) => review.mutate({ id: tx.id, label })} />
                ))}
                {feed.data.results.length === 0 && (
                  <tr><td colSpan={11} className="px-4 py-8 text-center text-ink-muted">No transactions yet.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
