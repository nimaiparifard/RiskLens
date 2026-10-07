import type { AuditEntry } from "@/api/schemas";
import { Card, CardTitle } from "@/components/ui";
import { formatDate, formatPct } from "@/lib/utils";

const LABEL: Record<string, string> = {
  "application.created": "Application created",
  "application.scored": "Scored by model",
  "decision.made": "Decision recorded",
  "decision.overridden": "Final decision overridden",
};

function detail(e: AuditEntry): string {
  const p = e.payload as Record<string, unknown>;
  if (e.action === "application.scored")
    return `PD ${formatPct(Number(p.probability))}, grade ${p.grade}, recommends ${p.recommendation} (model v${p.model_version})`;
  if (e.action === "decision.made")
    return `${p.decision}${p.is_override ? " — override of " + p.recommendation : ""}${p.comment ? `: “${p.comment}”` : ""}`;
  if (e.action === "decision.overridden") return `${p.previous_status} → ${p.decision}: “${p.comment}”`;
  return "";
}

export function AuditTimeline({ entries }: { entries: AuditEntry[] }) {
  return (
    <Card>
      <CardTitle hint="immutable">Audit history</CardTitle>
      <ol className="space-y-3">
        {entries.map((e) => (
          <li key={e.id} className="border-l-2 border-[var(--gridline)] pl-3">
            <div className="text-sm font-medium text-ink">{LABEL[e.action] ?? e.action}</div>
            <div className="text-xs text-ink-2">{detail(e)}</div>
            <div className="text-xs text-ink-muted">{formatDate(e.created_at)} · {e.actor ?? "system"}</div>
          </li>
        ))}
        {entries.length === 0 && <li className="text-sm text-ink-muted">No events yet.</li>}
      </ol>
    </Card>
  );
}
