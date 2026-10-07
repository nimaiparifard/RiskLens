/**
 * Zod schemas = runtime validation of every API response (the backend can never
 * silently drift from what the UI expects). Mirrors the DRF serializers /
 * OpenAPI schema (`npm run types` regenerates src/api/schema.d.ts).
 */
import { z } from "zod";

export const roleSchema = z.enum(["viewer", "officer", "admin"]);

export const userSchema = z.object({
  id: z.number(),
  username: z.string(),
  email: z.string().optional().default(""),
  first_name: z.string().optional().default(""),
  last_name: z.string().optional().default(""),
  role: roleSchema,
});

export const loginResponseSchema = z.object({
  access: z.string(),
  refresh: z.string(),
  user: userSchema,
});

export const reasonSchema = z.object({
  feature: z.string(),
  value: z.unknown().optional(),
  contribution: z.number(),
  direction: z.enum(["increases_risk", "decreases_risk"]),
  text: z.string(),
});

const num = z.coerce.number();

export const predictionSchema = z.object({
  id: z.string(),
  application: z.string(),
  created_at: z.string(),
  model_name: z.string(),
  model_version: z.string(),
  probability: z.number(),
  score_points: z.number().nullable(),
  grade: z.string(),
  decision: z.string(),
  threshold: z.number(),
  shap_reasons: z.array(reasonSchema),
  latency_ms: z.number(),
  shadow: z
    .object({ model_version: z.string(), probability: z.number(), decision: z.string() })
    .nullable()
    .optional(),
});

export const applicationStatusSchema = z.enum(["pending", "scored", "approved", "rejected"]);

export const applicationListItemSchema = z.object({
  id: z.string(),
  reference: z.string(),
  applicant_name: z.string(),
  contract_type: z.string(),
  amt_credit: num,
  amt_income_total: num,
  status: applicationStatusSchema,
  latest_grade: z.string(),
  latest_probability: z.number().nullable(),
  recommendation: z.string(),
  is_override: z.boolean(),
  created_at: z.string(),
});

export const applicationDetailSchema = z.object({
  id: z.string(),
  reference: z.string(),
  applicant: z.object({
    id: z.string(),
    external_id: z.string(),
    full_name: z.string(),
    email: z.string(),
    phone: z.string(),
  }),
  contract_type: z.string(),
  amt_credit: num,
  amt_annuity: num.nullable(),
  amt_income_total: num,
  features: z.record(z.unknown()),
  history: z.record(z.unknown()),
  status: applicationStatusSchema,
  latest_grade: z.string(),
  latest_probability: z.number().nullable(),
  recommendation: z.string(),
  decided_by: z.string().nullable(),
  decided_at: z.string().nullable(),
  decision_comment: z.string(),
  is_override: z.boolean(),
  created_at: z.string(),
  updated_at: z.string(),
  latest_prediction: predictionSchema.nullable(),
});

export const auditEntrySchema = z.object({
  id: z.number(),
  created_at: z.string(),
  actor: z.string().nullable(),
  action: z.string(),
  target_type: z.string(),
  target_id: z.string(),
  payload: z.record(z.unknown()),
  request_id: z.string(),
});

export const statsSchema = z.object({
  total: z.number(),
  by_status: z.record(z.number()),
  by_grade: z.record(z.number()),
  approval_rate: z.number().nullable(),
  override_rate: z.number().nullable(),
  avg_probability: z.number().nullable(),
});

export const transactionSchema = z.object({
  id: z.number(),
  transaction_id: z.number(),
  card1: z.number(),
  amount: num,
  product_cd: z.string(),
  p_emaildomain: z.string(),
  device_type: z.string(),
  transaction_dt: z.number(),
  probability: z.number().nullable(),
  grade: z.string(),
  decision: z.string(),
  reasons: z.array(reasonSchema),
  model_version: z.string(),
  latency_ms: z.number().nullable(),
  status: z.string(),
  is_flagged: z.boolean(),
  reviewed_by: z.string().nullable(),
  reviewed_at: z.string().nullable(),
  created_at: z.string(),
});

export const txStatsSchema = z.object({
  total: z.number(),
  flagged: z.number(),
  blocked: z.number(),
  confirmed: z.number(),
  avg_latency_ms: z.number().nullable(),
  latency_window_minutes: z.number().optional(),
});

export const driftReportSchema = z.object({
  id: z.number(),
  created_at: z.string(),
  task: z.string(),
  model_version: z.string(),
  window_start: z.string(),
  window_end: z.string(),
  n_samples: z.number(),
  feature_psi: z.record(z.number()),
  prediction_psi: z.number().nullable(),
  max_psi: z.number().nullable(),
  drifted_features: z.array(z.string()),
  status: z.string(),
  retrain_triggered: z.boolean(),
  report_url: z.string().nullable(),
  error: z.string(),
});

export const modelVersionSchema = z.object({
  version: z.string(),
  aliases: z.array(z.string()),
  created_at: z.union([z.number(), z.string()]).nullable().optional(),
  run_id: z.string().nullable().optional(),
  status: z.string().nullable().optional(),
  metrics: z.record(z.number()),
  is_champion: z.boolean(),
});

export const modelsResponseSchema = z.object({
  registry: z.record(
    z.object({
      name: z.string(),
      aliases: z.record(z.string()).default({}),
      versions: z.array(modelVersionSchema).default([]),
      error: z.string().optional(),
    }),
  ),
  serving: z.record(z.unknown()),
});

export const paginated = <T extends z.ZodTypeAny>(item: T) =>
  z.object({ count: z.number(), next: z.string().nullable(), previous: z.string().nullable(), results: z.array(item) });

export type User = z.infer<typeof userSchema>;
export type Role = z.infer<typeof roleSchema>;
export type Reason = z.infer<typeof reasonSchema>;
export type Prediction = z.infer<typeof predictionSchema>;
export type ApplicationListItem = z.infer<typeof applicationListItemSchema>;
export type ApplicationDetail = z.infer<typeof applicationDetailSchema>;
export type AuditEntry = z.infer<typeof auditEntrySchema>;
export type Stats = z.infer<typeof statsSchema>;
export type Transaction = z.infer<typeof transactionSchema>;
export type TxStats = z.infer<typeof txStatsSchema>;
export type DriftReport = z.infer<typeof driftReportSchema>;
export type ModelsResponse = z.infer<typeof modelsResponseSchema>;
export type ModelVersion = z.infer<typeof modelVersionSchema>;
