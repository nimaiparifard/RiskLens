import { useMutation } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";

import { api } from "@/api/endpoints";
import { Button, Card, ErrorBox, Input, Label, PageHeader, Select } from "@/components/ui";

type Form = Record<string, string>;

const INITIAL: Form = {
  full_name: "", email: "", NAME_CONTRACT_TYPE: "Cash loans", CODE_GENDER: "F", age: "35", years_employed: "5",
  AMT_INCOME_TOTAL: "180000", AMT_CREDIT: "500000", AMT_ANNUITY: "25000", AMT_GOODS_PRICE: "450000",
  NAME_INCOME_TYPE: "Working", NAME_EDUCATION_TYPE: "Secondary / secondary special",
  NAME_FAMILY_STATUS: "Married", CNT_CHILDREN: "0", FLAG_OWN_CAR: "N", FLAG_OWN_REALTY: "Y",
  OCCUPATION_TYPE: "Laborers", ORGANIZATION_TYPE: "Business Entity Type 3",
  EXT_SOURCE_1: "", EXT_SOURCE_2: "0.55", EXT_SOURCE_3: "0.5",
};

/** Convert friendly form fields into Home Credit feature names. */
export function toFeatures(f: Form): Record<string, unknown> {
  const n = (v: string) => (v === "" ? null : Number(v));
  return {
    NAME_CONTRACT_TYPE: f.NAME_CONTRACT_TYPE,
    CODE_GENDER: f.CODE_GENDER,
    FLAG_OWN_CAR: f.FLAG_OWN_CAR,
    FLAG_OWN_REALTY: f.FLAG_OWN_REALTY,
    CNT_CHILDREN: n(f.CNT_CHILDREN),
    AMT_INCOME_TOTAL: n(f.AMT_INCOME_TOTAL),
    AMT_CREDIT: n(f.AMT_CREDIT),
    AMT_ANNUITY: n(f.AMT_ANNUITY),
    AMT_GOODS_PRICE: n(f.AMT_GOODS_PRICE),
    DAYS_BIRTH: -Math.round(Number(f.age) * 365.25),
    DAYS_EMPLOYED: f.years_employed === "" ? 365243 : -Math.round(Number(f.years_employed) * 365.25),
    NAME_INCOME_TYPE: f.NAME_INCOME_TYPE,
    NAME_EDUCATION_TYPE: f.NAME_EDUCATION_TYPE,
    NAME_FAMILY_STATUS: f.NAME_FAMILY_STATUS,
    OCCUPATION_TYPE: f.OCCUPATION_TYPE || null,
    ORGANIZATION_TYPE: f.ORGANIZATION_TYPE || null,
    EXT_SOURCE_1: n(f.EXT_SOURCE_1),
    EXT_SOURCE_2: n(f.EXT_SOURCE_2),
    EXT_SOURCE_3: n(f.EXT_SOURCE_3),
  };
}

const OPTIONS: Record<string, string[]> = {
  NAME_CONTRACT_TYPE: ["Cash loans", "Revolving loans"],
  CODE_GENDER: ["F", "M"],
  NAME_INCOME_TYPE: ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student"],
  NAME_EDUCATION_TYPE: ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
  NAME_FAMILY_STATUS: ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
  FLAG_OWN_CAR: ["Y", "N"],
  FLAG_OWN_REALTY: ["Y", "N"],
  OCCUPATION_TYPE: ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff", "Accountants", "Medicine staff", ""],
  ORGANIZATION_TYPE: ["Business Entity Type 3", "Self-employed", "Government", "School", "Medicine", "Trade: type 7", "Construction", ""],
};

const SECTIONS: Array<[string, Array<[string, string, string?]>]> = [
  ["Applicant", [["full_name", "Full name"], ["email", "E-mail"], ["age", "Age (years)", "number"], ["CODE_GENDER", "Gender (fairness audit only)"],
    ["NAME_FAMILY_STATUS", "Family status"], ["CNT_CHILDREN", "Children", "number"]]],
  ["Employment & income", [["NAME_INCOME_TYPE", "Income type"], ["years_employed", "Years employed (blank = none)", "number"],
    ["OCCUPATION_TYPE", "Occupation"], ["ORGANIZATION_TYPE", "Employer type"], ["NAME_EDUCATION_TYPE", "Education"],
    ["AMT_INCOME_TOTAL", "Annual income", "number"]]],
  ["Loan", [["NAME_CONTRACT_TYPE", "Contract type"], ["AMT_CREDIT", "Loan amount", "number"], ["AMT_ANNUITY", "Annuity", "number"],
    ["AMT_GOODS_PRICE", "Goods price", "number"], ["FLAG_OWN_CAR", "Owns car"], ["FLAG_OWN_REALTY", "Owns property"]]],
  ["External credit bureau scores (0–1)", [["EXT_SOURCE_1", "Bureau 1", "number"], ["EXT_SOURCE_2", "Bureau 2", "number"], ["EXT_SOURCE_3", "Bureau 3", "number"]]],
];

export function NewApplicationPage() {
  const [form, setForm] = useState<Form>(INITIAL);
  const navigate = useNavigate();
  const create = useMutation({
    mutationFn: () => api.createApplication({
      applicant: { full_name: form.full_name, email: form.email },
      features: toFeatures(form),
    }),
    onSuccess: (app) => navigate(`/applications/${app.id}`),
  });

  const set = (k: string) => (e: { target: { value: string } }) => setForm((f) => ({ ...f, [k]: e.target.value }));
  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    create.mutate();
  };

  return (
    <form onSubmit={onSubmit}>
      <PageHeader title="New loan application" subtitle="Bureau history (if any) is attached automatically when the applicant is known."
        actions={<Button type="submit" loading={create.isPending}>Create application</Button>} />
      {create.error && <div className="mb-4"><ErrorBox error={create.error} /></div>}
      <div className="grid gap-4 lg:grid-cols-2">
        {SECTIONS.map(([title, fields]) => (
          <Card key={title}>
            <h2 className="mb-3 text-sm font-semibold text-ink">{title}</h2>
            <div className="grid gap-3 sm:grid-cols-2">
              {fields.map(([key, label, type]) => (
                <div key={key}>
                  <Label htmlFor={key}>{label}</Label>
                  {OPTIONS[key] ? (
                    <Select id={key} value={form[key]} onChange={set(key)}>
                      {OPTIONS[key].map((o) => <option key={o} value={o}>{o || "— not provided —"}</option>)}
                    </Select>
                  ) : (
                    <Input id={key} type={type ?? "text"} step="any" value={form[key]} onChange={set(key)}
                      required={["full_name", "age", "AMT_INCOME_TOTAL", "AMT_CREDIT"].includes(key)}
                      min={type === "number" && key.startsWith("EXT") ? 0 : undefined}
                      max={key.startsWith("EXT") ? 1 : undefined} />
                  )}
                </div>
              ))}
            </div>
          </Card>
        ))}
      </div>
    </form>
  );
}
