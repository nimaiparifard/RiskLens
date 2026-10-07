import { render, screen } from "@testing-library/react";

import { GradeBadge, StatusBadge } from "@/components/badges";
import { buildWaterfall, ScoreGauge, ShapWaterfall } from "@/components/charts";
import { toFeatures } from "@/features/applications/NewApplicationPage";
import { psiSeries } from "@/features/monitoring/MonitoringPage";
import { displayField } from "@/lib/utils";

const reasons = [
  { feature: "EXT_SOURCE_2", value: 0.1, contribution: 0.8, direction: "increases_risk" as const, text: "Low external score" },
  { feature: "AMT_CREDIT", value: 500000, contribution: -0.3, direction: "decreases_risk" as const, text: "Small loan" },
];

describe("badges", () => {
  it("grade badge shows the letter and a text label (never colour alone)", () => {
    render(<GradeBadge grade="E" size="lg" />);
    expect(screen.getByLabelText("Grade E, very high risk")).toBeInTheDocument();
    expect(screen.getByText("very high risk")).toBeInTheDocument();
  });
  it("status badge renders human labels", () => {
    render(<StatusBadge status="scored" />);
    expect(screen.getByText("Awaiting decision")).toBeInTheDocument();
  });
  it("missing grade renders a dash", () => {
    render(<GradeBadge grade="" />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});

describe("charts", () => {
  it("gauge shows PD and cut-off", () => {
    render(<ScoreGauge probability={0.123} threshold={0.2} points={612} />);
    expect(screen.getByText("12.3%")).toBeInTheDocument();
    expect(screen.getByText(/cut-off 20.0%/)).toBeInTheDocument();
    expect(screen.getByText("612")).toBeInTheDocument();
  });
  it("waterfall steps are cumulative and end in a net effect", () => {
    const steps = buildWaterfall(reasons);
    expect(steps[0].range).toEqual([0, 0.8]);
    expect(steps[1].range[0]).toBeCloseTo(0.5);
    expect(steps[1].range[1]).toBeCloseTo(0.8);
    expect(steps[2].name).toBe("Net effect");
    expect(steps[2].contribution).toBeCloseTo(0.5);
  });
  it("waterfall renders a legend", () => {
    render(<ShapWaterfall reasons={reasons} />);
    expect(screen.getByText("increases risk")).toBeInTheDocument();
  });
  it("empty waterfall message", () => {
    render(<ShapWaterfall reasons={[]} />);
    expect(screen.getByText(/No explanation/)).toBeInTheDocument();
  });
});

describe("domain helpers", () => {
  it("converts the form into Home Credit features", () => {
    const f = toFeatures({ age: "40", years_employed: "", AMT_CREDIT: "1000", EXT_SOURCE_1: "", EXT_SOURCE_2: "0.5" } as Record<string, string>);
    expect(f.DAYS_BIRTH).toBe(-14610);
    expect(f.DAYS_EMPLOYED).toBe(365243); // Home Credit's "not employed" sentinel
    expect(f.AMT_CREDIT).toBe(1000);
    expect(f.EXT_SOURCE_1).toBeNull();
  });
  it("formats Home Credit day fields", () => {
    expect(displayField("DAYS_BIRTH", -14610)).toBe("40 years");
    expect(displayField("DAYS_EMPLOYED", 365243)).toBe("not employed / pensioner");
  });
  it("builds PSI series from reports (oldest first, top-3 of latest)", () => {
    const mk = (id: number, end: string, psi: Record<string, number>, status = "ok") => ({
      id, created_at: end, task: "credit", model_version: "1", window_start: end, window_end: end, n_samples: 100,
      feature_psi: psi, prediction_psi: 0.01, max_psi: Math.max(...Object.values(psi)), drifted_features: [],
      status, retrain_triggered: false, report_url: null, error: "",
    });
    const { points, top } = psiSeries([
      mk(3, "2026-10-03T00:00:00Z", { a: 0.3, b: 0.1, c: 0.05, d: 0.01 }, "drift"),
      mk(2, "2026-10-02T00:00:00Z", { a: 0.1, b: 0.1, c: 0.02, d: 0.0 }),
      mk(1, "2026-10-01T00:00:00Z", {}, "skipped"),
    ]);
    expect(top).toEqual(["a", "b", "c"]);
    expect(points).toHaveLength(2);
    expect(points[1].a).toBe(0.3);
  });
});
