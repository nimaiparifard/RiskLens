import { ArrowLeft, Gauge } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import { GradeBadge, StatusBadge } from "@/components/badges";
import { ScoreGauge, ShapWaterfall } from "@/components/charts";
import { Button, Card, CardTitle, ErrorBox, PageHeader, Spinner } from "@/components/ui";
import { AuditTimeline } from "@/features/decision/AuditTimeline";
import { DecisionPanel } from "@/features/decision/DecisionPanel";
import { useAuth } from "@/lib/auth";
import { displayField, FIELD_LABELS, formatDate, formatMoney, formatValue } from "@/lib/utils";

import { useApplication, useAudit, useScore } from "./hooks";

export function ApplicationDetailPage() {
  const { id = "" } = useParams();
  const app = useApplication(id);
  const audit = useAudit(id);
  const score = useScore(id);
  const { can } = useAuth();

  if (app.isLoading) return <Spinner />;
  if (app.error) return <ErrorBox error={app.error} />;
  const a = app.data!;
  const pred = a.latest_prediction;
  const final = a.status === "approved" || a.status === "rejected";

  return (
    <div>
      <Link to="/applications" className="mb-3 inline-flex items-center gap-1 text-sm text-ink-2 hover:text-ink">
        <ArrowLeft className="h-4 w-4" aria-hidden /> Applications
      </Link>
      <PageHeader
        title={`${a.reference} · ${a.applicant.full_name}`}
        subtitle={<>Created {formatDate(a.created_at)} · <StatusBadge status={a.status} /></>}
        actions={can("decide") && !final && (
          <Button onClick={() => score.mutate()} loading={score.isPending} variant={pred ? "secondary" : "primary"}>
            <Gauge className="h-4 w-4" aria-hidden /> {pred ? "Re-score" : "Score application"}
          </Button>
        )}
      />
      {score.error && <div className="mb-4"><ErrorBox error={score.error} /></div>}

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="space-y-4 xl:col-span-2">
          <Card>
            <CardTitle hint={pred && `model ${pred.model_name} v${pred.model_version} · ${pred.latency_ms.toFixed(1)} ms`}>Risk assessment</CardTitle>
            {pred ? (
              <div className="flex flex-wrap items-center justify-around gap-6">
                <ScoreGauge probability={pred.probability} threshold={pred.threshold} points={pred.score_points} />
                <GradeBadge grade={pred.grade} size="lg" />
                <div className="text-center">
                  <div className="text-xs text-ink-2">Recommendation</div>
                  <div className="mt-1"><StatusBadge status={pred.decision} /></div>
                  {pred.shadow && (
                    <div className="mt-2 text-xs text-ink-muted">
                      challenger v{pred.shadow.model_version}: {(pred.shadow.probability * 100).toFixed(1)}% ({pred.shadow.decision})
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <p className="text-sm text-ink-muted">Not scored yet.</p>
            )}
          </Card>

          {pred && (
            <Card>
              <CardTitle hint="SHAP, log-odds vs. an average applicant">Top {pred.shap_reasons.length} reasons</CardTitle>
              <ol className="mb-4 space-y-1.5">
                {pred.shap_reasons.map((r, i) => (
                  <li key={r.feature} className="flex items-start gap-2 text-sm">
                    <span className="mt-0.5 w-4 text-xs text-ink-muted">{i + 1}.</span>
                    <span className="mt-1.5 inline-block h-2 w-2 shrink-0 rounded-full"
                      style={{ background: r.direction === "increases_risk" ? "var(--div-pos)" : "var(--div-neg)" }} aria-hidden />
                    <span className="text-ink">
                      {r.text} <span className="text-xs text-ink-muted">({r.feature} = {formatValue(r.value)}, {r.direction === "increases_risk" ? "↑ risk" : "↓ risk"})</span>
                    </span>
                  </li>
                ))}
              </ol>
              <ShapWaterfall reasons={pred.shap_reasons} />
            </Card>
          )}

          <Card>
            <CardTitle>Application data</CardTitle>
            <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
              <div className="flex justify-between gap-2"><dt className="text-ink-2">Loan amount</dt><dd className="tabular text-ink">{formatMoney(a.amt_credit)}</dd></div>
              <div className="flex justify-between gap-2"><dt className="text-ink-2">Annual income</dt><dd className="tabular text-ink">{formatMoney(a.amt_income_total)}</dd></div>
              {Object.entries(a.features)
                .filter(([k]) => FIELD_LABELS[k] && !["AMT_CREDIT", "AMT_INCOME_TOTAL"].includes(k))
                .map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-2">
                    <dt className="text-ink-2">{FIELD_LABELS[k]}</dt>
                    <dd className="tabular text-right text-ink">{displayField(k, v)}</dd>
                  </div>
                ))}
            </dl>
            {Object.keys(a.history).length > 0 && (
              <p className="mt-3 text-xs text-ink-muted">+ {Object.keys(a.history).length} credit-history features from bureau / previous loans.</p>
            )}
          </Card>
        </div>

        <div className="space-y-4">
          <DecisionPanel app={a} />
          {audit.data && <AuditTimeline entries={audit.data} />}
        </div>
      </div>
    </div>
  );
}
