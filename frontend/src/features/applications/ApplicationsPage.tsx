import { Plus } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { GradeBadge, StatusBadge } from "@/components/badges";
import { Button, Card, ErrorBox, Input, PageHeader, Select, Spinner, StatTile } from "@/components/ui";
import { useAuth } from "@/lib/auth";
import { formatDate, formatMoney, formatPct } from "@/lib/utils";

import { useApplications, useStats } from "./hooks";

export function ApplicationsPage() {
  const [status, setStatus] = useState("");
  const [grade, setGrade] = useState("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const { can } = useAuth();
  const navigate = useNavigate();
  const stats = useStats();
  const list = useApplications({ status, grade, search, page });

  return (
    <div>
      <PageHeader
        title="Loan applications"
        subtitle="Score applications, review the reasons, decide."
        actions={can("decide") && (
          <Button onClick={() => navigate("/applications/new")}>
            <Plus className="h-4 w-4" aria-hidden /> New application
          </Button>
        )}
      />

      <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Applications" value={stats.data?.total ?? "—"} sub={`${stats.data?.by_status.pending ?? 0} pending`} />
        <StatTile label="Awaiting decision" value={stats.data?.by_status.scored ?? 0} />
        <StatTile label="Approval rate" value={formatPct(stats.data?.approval_rate)} sub="of decided applications" />
        <StatTile label="Override rate" value={formatPct(stats.data?.override_rate)} sub="decisions against the model" />
      </div>

      <Card className="p-0">
        <div className="flex flex-wrap gap-2 border-b border-[var(--border)] p-3">
          <Input placeholder="Search reference or name…" value={search} className="max-w-xs"
            onChange={(e) => { setSearch(e.target.value); setPage(1); }} aria-label="Search" />
          <Select value={status} className="w-44" aria-label="Status"
            onChange={(e) => { setStatus(e.target.value); setPage(1); }}>
            <option value="">All statuses</option>
            <option value="pending">Pending</option>
            <option value="scored">Awaiting decision</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
          </Select>
          <Select value={grade} className="w-32" aria-label="Grade" onChange={(e) => { setGrade(e.target.value); setPage(1); }}>
            <option value="">All grades</option>
            {["A", "B", "C", "D", "E"].map((g) => <option key={g} value={g}>Grade {g}</option>)}
          </Select>
        </div>
        {list.isLoading && <Spinner />}
        {list.error && <div className="p-3"><ErrorBox error={list.error} /></div>}
        {list.data && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-ink-2">
                <tr className="border-b border-[var(--border)]">
                  <th className="px-4 py-2 font-medium">Reference</th>
                  <th className="px-4 py-2 font-medium">Applicant</th>
                  <th className="px-4 py-2 text-right font-medium">Loan</th>
                  <th className="px-4 py-2 text-right font-medium">Income</th>
                  <th className="px-4 py-2 font-medium">Grade</th>
                  <th className="px-4 py-2 text-right font-medium">PD</th>
                  <th className="px-4 py-2 font-medium">Status</th>
                  <th className="px-4 py-2 font-medium">Created</th>
                </tr>
              </thead>
              <tbody className="tabular">
                {list.data.results.map((a) => (
                  <tr key={a.id} className="border-b border-[var(--border)] hover:bg-[var(--gridline)]/40">
                    <td className="px-4 py-2"><Link className="font-medium text-brand hover:underline" to={`/applications/${a.id}`}>{a.reference}</Link></td>
                    <td className="px-4 py-2 text-ink">{a.applicant_name}</td>
                    <td className="px-4 py-2 text-right">{formatMoney(a.amt_credit)}</td>
                    <td className="px-4 py-2 text-right text-ink-2">{formatMoney(a.amt_income_total)}</td>
                    <td className="px-4 py-2"><GradeBadge grade={a.latest_grade} /></td>
                    <td className="px-4 py-2 text-right">{formatPct(a.latest_probability)}</td>
                    <td className="px-4 py-2"><StatusBadge status={a.status} />{a.is_override && <span className="ml-1 text-xs text-ink-muted">override</span>}</td>
                    <td className="px-4 py-2 text-xs text-ink-2">{formatDate(a.created_at)}</td>
                  </tr>
                ))}
                {list.data.results.length === 0 && (
                  <tr><td colSpan={8} className="px-4 py-8 text-center text-ink-muted">No applications match these filters.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        )}
        {list.data && (
          <div className="flex items-center justify-between p-3 text-xs text-ink-2">
            <span>{list.data.count} applications</span>
            <div className="flex gap-2">
              <Button size="sm" variant="secondary" disabled={!list.data.previous} onClick={() => setPage((p) => p - 1)}>Previous</Button>
              <Button size="sm" variant="secondary" disabled={!list.data.next} onClick={() => setPage((p) => p + 1)}>Next</Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
