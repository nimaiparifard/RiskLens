import { CheckCircle2, XCircle } from "lucide-react";
import { useState } from "react";

import type { ApplicationDetail } from "@/api/schemas";
import { StatusBadge } from "@/components/badges";
import { Button, Card, CardTitle, ErrorBox, Label, Textarea } from "@/components/ui";
import { useDecide } from "@/features/applications/hooks";
import { useAuth } from "@/lib/auth";
import { formatDate } from "@/lib/utils";

/** Approve / reject. Going against the model recommendation (an override) requires a comment;
 *  changing a final decision is admin-only - both enforced server-side, mirrored here. */
export function DecisionPanel({ app }: { app: ApplicationDetail }) {
  const { can } = useAuth();
  const decide = useDecide(app.id);
  const [comment, setComment] = useState("");
  const final = app.status === "approved" || app.status === "rejected";
  const recommendation = app.latest_prediction?.decision;
  const allowed = final ? can("admin") : can("decide");

  const needsComment = (d: "approve" | "reject") => final || d !== recommendation;
  const submit = (d: "approve" | "reject") => decide.mutate({ decision: d, comment });

  return (
    <Card>
      <CardTitle hint={recommendation && <>model recommends <strong className="text-ink">{recommendation}</strong></>}>
        Decision
      </CardTitle>
      {final && (
        <div className="mb-3 text-sm text-ink-2">
          <StatusBadge status={app.status} /> by <strong className="text-ink">{app.decided_by}</strong> · {formatDate(app.decided_at)}
          {app.is_override && <span className="ml-1">(override)</span>}
          {app.decision_comment && <p className="mt-2 italic">“{app.decision_comment}”</p>}
        </div>
      )}
      {!app.latest_prediction ? (
        <p className="text-sm text-ink-muted">Score the application before deciding.</p>
      ) : !allowed ? (
        <p className="text-sm text-ink-muted">{final ? "Only admins can change a final decision." : "Your role is read-only."}</p>
      ) : (
        <div className="space-y-3">
          <div>
            <Label htmlFor="comment">Comment {final ? "(required to override)" : "(required when overriding the model)"}</Label>
            <Textarea id="comment" rows={3} value={comment} onChange={(e) => setComment(e.target.value)}
              placeholder="Justification, documents checked, …" />
          </div>
          {decide.error && <ErrorBox error={decide.error} />}
          <div className="flex gap-2">
            <Button variant="success" loading={decide.isPending && decide.variables?.decision === "approve"}
              disabled={needsComment("approve") && !comment.trim()} onClick={() => submit("approve")}>
              <CheckCircle2 className="h-4 w-4" aria-hidden /> Approve
            </Button>
            <Button variant="danger" loading={decide.isPending && decide.variables?.decision === "reject"}
              disabled={needsComment("reject") && !comment.trim()} onClick={() => submit("reject")}>
              <XCircle className="h-4 w-4" aria-hidden /> Reject
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
}
