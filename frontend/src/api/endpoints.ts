import { z } from "zod";

import { request } from "./client";
import {
  applicationDetailSchema,
  applicationListItemSchema,
  auditEntrySchema,
  driftReportSchema,
  loginResponseSchema,
  modelsResponseSchema,
  paginated,
  predictionSchema,
  statsSchema,
  transactionSchema,
  txStatsSchema,
  userSchema,
} from "./schemas";

export type ApplicationFilters = { status?: string; grade?: string; search?: string; page?: number };

const qs = (params: Record<string, string | number | undefined>) => {
  const p = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => v !== undefined && v !== "" && p.set(k, String(v)));
  const s = p.toString();
  return s ? `?${s}` : "";
};

export const api = {
  login: (username: string, password: string) =>
    request("/api/auth/token/", loginResponseSchema, { method: "POST", json: { username, password } }),
  me: () => request("/api/auth/me/", userSchema),

  applications: (f: ApplicationFilters) =>
    request(`/api/applications/${qs({ ...f, ordering: "-created_at" })}`, paginated(applicationListItemSchema)),
  application: (id: string) => request(`/api/applications/${id}/`, applicationDetailSchema),
  createApplication: (body: { applicant: Record<string, string>; features: Record<string, unknown> }) =>
    request("/api/applications/", applicationDetailSchema, { method: "POST", json: body }),
  scoreApplication: (id: string) =>
    request(`/api/applications/${id}/score/`, predictionSchema, { method: "POST" }),
  decide: (id: string, decision: "approve" | "reject", comment: string) =>
    request(`/api/applications/${id}/decision/`, applicationDetailSchema, {
      method: "POST",
      json: { decision, comment },
    }),
  audit: (id: string) => request(`/api/applications/${id}/audit/`, z.array(auditEntrySchema)),
  stats: () => request("/api/applications/stats/", statsSchema),

  transactions: (params: { flagged?: boolean; since_id?: number; page_size?: number }) =>
    request(
      `/api/transactions/${qs({
        flagged: params.flagged === undefined ? undefined : String(params.flagged),
        since_id: params.since_id,
        page_size: params.page_size ?? 50,
        ordering: "-created_at",
      })}`,
      paginated(transactionSchema),
    ),
  txStats: () => request("/api/transactions/stats/", txStatsSchema),
  reviewTransaction: (id: number, label: "fraud" | "legit") =>
    request(`/api/transactions/${id}/review/`, transactionSchema, { method: "POST", json: { label } }),

  models: () => request("/api/models/", modelsResponseSchema),
  reloadModels: () => request("/api/models/reload/", z.record(z.string()), { method: "POST", json: {} }),

  drift: (task: string) =>
    request(`/api/monitoring/drift/${qs({ task, page_size: 60 })}`, paginated(driftReportSchema)),
  runDrift: (task: string, hours?: number) =>
    request("/api/monitoring/drift/run/", driftReportSchema, { method: "POST", json: { task, hours } }),
};
