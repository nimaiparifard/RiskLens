import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const money = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
export const formatMoney = (v: number | null | undefined) => (v == null ? "—" : money.format(v));
export const formatPct = (v: number | null | undefined, digits = 1) =>
  v == null ? "—" : `${(v * 100).toFixed(digits)}%`;
export const formatDate = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—";

export function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "missing";
  if (typeof v === "number") return Math.abs(v) >= 1000 ? money.format(v) : String(Math.round(v * 1000) / 1000);
  return String(v);
}

/** Human-readable labels for Home Credit field names. */
export const FIELD_LABELS: Record<string, string> = {
  AMT_INCOME_TOTAL: "Annual income",
  AMT_CREDIT: "Loan amount",
  AMT_ANNUITY: "Annuity (yearly payment)",
  AMT_GOODS_PRICE: "Goods price",
  DAYS_BIRTH: "Age",
  DAYS_EMPLOYED: "Employment length",
  NAME_CONTRACT_TYPE: "Contract type",
  NAME_INCOME_TYPE: "Income type",
  NAME_EDUCATION_TYPE: "Education",
  NAME_FAMILY_STATUS: "Family status",
  OCCUPATION_TYPE: "Occupation",
  ORGANIZATION_TYPE: "Employer type",
  CODE_GENDER: "Gender (audit only, not a model input)",
  EXT_SOURCE_1: "External score 1",
  EXT_SOURCE_2: "External score 2",
  EXT_SOURCE_3: "External score 3",
  CNT_CHILDREN: "Children",
  FLAG_OWN_CAR: "Owns car",
  FLAG_OWN_REALTY: "Owns property",
};

export function displayField(key: string, value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (key === "DAYS_BIRTH" && typeof value === "number") return `${Math.floor(-value / 365.25)} years`;
  if (key === "DAYS_EMPLOYED" && typeof value === "number")
    return value === 365243 ? "not employed / pensioner" : `${(-value / 365.25).toFixed(1)} years`;
  return formatValue(value);
}
