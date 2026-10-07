import { AlertOctagon, AlertTriangle, CheckCircle2, CircleDot, Clock, ShieldAlert, XCircle } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/** Grades use the status palette, always with the letter + an icon (never colour alone). */
const GRADE: Record<string, { color: string; icon: ReactNode; label: string }> = {
  A: { color: "var(--status-good)", icon: <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />, label: "very low risk" },
  B: { color: "var(--status-good)", icon: <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />, label: "low risk" },
  C: { color: "var(--status-warning)", icon: <AlertTriangle className="h-3.5 w-3.5" aria-hidden />, label: "medium risk" },
  D: { color: "var(--status-serious)", icon: <ShieldAlert className="h-3.5 w-3.5" aria-hidden />, label: "high risk" },
  E: { color: "var(--status-critical)", icon: <AlertOctagon className="h-3.5 w-3.5" aria-hidden />, label: "very high risk" },
  LOW: { color: "var(--status-good)", icon: <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />, label: "low" },
  MEDIUM: { color: "var(--status-warning)", icon: <AlertTriangle className="h-3.5 w-3.5" aria-hidden />, label: "medium" },
  HIGH: { color: "var(--status-critical)", icon: <AlertOctagon className="h-3.5 w-3.5" aria-hidden />, label: "high" },
};

function Pill({ color, icon, children, title }: { color: string; icon: ReactNode; children: ReactNode; title?: string }) {
  return (
    <span
      title={title}
      className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium text-ink"
      style={{ borderColor: color }}
    >
      <span style={{ color }} className="inline-flex">
        {icon}
      </span>
      {children}
    </span>
  );
}

export function GradeBadge({ grade, size = "sm" }: { grade?: string | null; size?: "sm" | "lg" }) {
  if (!grade) return <span className="text-xs text-ink-muted">—</span>;
  const g = GRADE[grade] ?? { color: "var(--text-muted)", icon: <CircleDot className="h-3.5 w-3.5" />, label: "" };
  if (size === "lg") {
    return (
      <div
        className="inline-flex flex-col items-center rounded-lg border-2 px-5 py-2"
        style={{ borderColor: g.color }}
        aria-label={`Grade ${grade}, ${g.label}`}
      >
        <span className="text-3xl font-bold text-ink">{grade}</span>
        <span className="flex items-center gap-1 text-xs text-ink-2">
          <span style={{ color: g.color }}>{g.icon}</span>
          {g.label}
        </span>
      </div>
    );
  }
  return (
    <Pill color={g.color} icon={g.icon} title={g.label}>
      {grade}
    </Pill>
  );
}

const STATUS: Record<string, { color: string; icon: ReactNode; label: string }> = {
  pending: { color: "var(--text-muted)", icon: <Clock className="h-3.5 w-3.5" aria-hidden />, label: "Pending" },
  scored: { color: "var(--series-1)", icon: <CircleDot className="h-3.5 w-3.5" aria-hidden />, label: "Awaiting decision" },
  approved: { color: "var(--status-good)", icon: <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />, label: "Approved" },
  rejected: { color: "var(--status-critical)", icon: <XCircle className="h-3.5 w-3.5" aria-hidden />, label: "Rejected" },
  approve: { color: "var(--status-good)", icon: <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />, label: "Approve" },
  review: { color: "var(--status-warning)", icon: <AlertTriangle className="h-3.5 w-3.5" aria-hidden />, label: "Review" },
  reject: { color: "var(--status-critical)", icon: <XCircle className="h-3.5 w-3.5" aria-hidden />, label: "Reject" },
  block: { color: "var(--status-critical)", icon: <AlertOctagon className="h-3.5 w-3.5" aria-hidden />, label: "Block" },
  blocked: { color: "var(--status-critical)", icon: <AlertOctagon className="h-3.5 w-3.5" aria-hidden />, label: "Blocked" },
  confirmed_fraud: { color: "var(--status-critical)", icon: <ShieldAlert className="h-3.5 w-3.5" aria-hidden />, label: "Confirmed fraud" },
  cleared: { color: "var(--status-good)", icon: <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />, label: "Cleared" },
  ok: { color: "var(--status-good)", icon: <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />, label: "Stable" },
  drift: { color: "var(--status-critical)", icon: <AlertOctagon className="h-3.5 w-3.5" aria-hidden />, label: "Drift" },
  skipped: { color: "var(--text-muted)", icon: <Clock className="h-3.5 w-3.5" aria-hidden />, label: "Skipped" },
  failed: { color: "var(--status-serious)", icon: <XCircle className="h-3.5 w-3.5" aria-hidden />, label: "Failed" },
};

export function StatusBadge({ status, className }: { status: string; className?: string }) {
  const s = STATUS[status] ?? { color: "var(--text-muted)", icon: <CircleDot className="h-3.5 w-3.5" />, label: status };
  return (
    <span className={cn("inline-flex", className)}>
      <Pill color={s.color} icon={s.icon}>
        {s.label}
      </Pill>
    </span>
  );
}
