import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Reason } from "@/api/schemas";
import { formatPct, formatValue } from "@/lib/utils";

const AXIS = { stroke: "var(--baseline)", tick: { fill: "var(--text-muted)", fontSize: 11 } };
const TOOLTIP_STYLE = {
  background: "var(--surface-1)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  color: "var(--text-primary)",
  fontSize: 12,
};

/* ------------------------------------------------------------------ gauge */
/** Semicircular PD gauge: the arc shows PD on a 0-50% scale, the tick marks the cut-off. */
export function ScoreGauge({ probability, threshold, points }: { probability: number; threshold: number; points?: number | null }) {
  const max = 0.5;
  const angle = (v: number) => Math.PI * (1 - Math.min(v, max) / max);
  const pt = (v: number, r: number) => [60 + r * Math.cos(angle(v)), 60 - r * Math.sin(angle(v))];
  const arc = (from: number, to: number) => {
    const [x1, y1] = pt(from, 48);
    const [x2, y2] = pt(to, 48);
    return `M ${x1} ${y1} A 48 48 0 0 1 ${x2} ${y2}`;
  };
  const over = probability >= threshold;
  const [tx1, ty1] = pt(threshold, 40);
  const [tx2, ty2] = pt(threshold, 58);
  return (
    <figure className="flex flex-col items-center" aria-label={`Probability of default ${formatPct(probability)}`}>
      <svg viewBox="0 0 120 70" className="w-48" role="img">
        <path d={arc(0, max)} stroke="var(--gridline)" strokeWidth={10} fill="none" strokeLinecap="round" />
        <path
          d={arc(0, Math.max(probability, 0.002))}
          stroke={over ? "var(--status-critical)" : "var(--series-1)"}
          strokeWidth={10}
          fill="none"
          strokeLinecap="round"
        />
        <line x1={tx1} y1={ty1} x2={tx2} y2={ty2} stroke="var(--text-primary)" strokeWidth={1.5} />
        <text x={60} y={56} textAnchor="middle" className="fill-[var(--text-primary)]" fontSize={16} fontWeight={600}>
          {formatPct(probability)}
        </text>
        <text x={60} y={67} textAnchor="middle" className="fill-[var(--text-muted)]" fontSize={7}>
          probability of default
        </text>
      </svg>
      <figcaption className="mt-1 text-xs text-ink-2">
        cut-off {formatPct(threshold)}
        {points != null && <> · score <strong className="text-ink">{points}</strong></>}
      </figcaption>
    </figure>
  );
}

/* --------------------------------------------------------------- waterfall */
type Step = { name: string; range: [number, number]; contribution: number; text: string; value: string; total?: boolean };

export function buildWaterfall(reasons: Reason[]): Step[] {
  const steps: Step[] = [];
  let cum = 0;
  for (const r of reasons) {
    const start = cum;
    cum += r.contribution;
    steps.push({
      name: r.feature,
      range: [Math.min(start, cum), Math.max(start, cum)],
      contribution: r.contribution,
      text: r.text,
      value: formatValue(r.value),
    });
  }
  steps.push({ name: "Net effect", range: [Math.min(0, cum), Math.max(0, cum)], contribution: cum, text: "Sum of the factors above", value: "", total: true });
  return steps;
}

/**
 * SHAP waterfall of the top reasons (log-odds). Diverging encoding:
 * red = pushes towards default, blue = pushes towards repayment.
 */
export function ShapWaterfall({ reasons }: { reasons: Reason[] }) {
  if (!reasons.length) return <p className="text-sm text-ink-muted">No explanation available.</p>;
  const data = buildWaterfall(reasons);
  return (
    <div>
      <div className="mb-2 flex gap-4 text-xs text-ink-2" aria-hidden>
        <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm bg-[var(--div-pos)]" />increases risk</span>
        <span className="flex items-center gap-1"><i className="inline-block h-2.5 w-2.5 rounded-sm bg-[var(--div-neg)]" />decreases risk</span>
      </div>
      <ResponsiveContainer width="100%" height={46 * data.length + 30}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 48, top: 4, bottom: 4 }} barCategoryGap={6}>
          <CartesianGrid horizontal={false} stroke="var(--gridline)" />
          <XAxis type="number" {...AXIS} tickFormatter={(v: number) => v.toFixed(2)} />
          <YAxis type="category" dataKey="name" width={170} {...AXIS} tick={{ ...AXIS.tick, fill: "var(--text-secondary)" }} />
          <ReferenceLine x={0} stroke="var(--baseline)" />
          <Tooltip
            cursor={{ fill: "var(--gridline)", opacity: 0.4 }}
            contentStyle={TOOLTIP_STYLE}
            formatter={(_v, _n, item) => {
              const s = item.payload as Step;
              return [`${s.contribution > 0 ? "+" : ""}${s.contribution.toFixed(3)} log-odds`, s.text];
            }}
            labelFormatter={(label, payload) => {
              const s = payload?.[0]?.payload as Step | undefined;
              return s?.value ? `${label} = ${s.value}` : String(label);
            }}
          />
          <Bar dataKey="range" radius={4} isAnimationActive={false}>
            {data.map((s) => (
              <Cell
                key={s.name}
                fill={s.contribution > 0 ? "var(--div-pos)" : "var(--div-neg)"}
                fillOpacity={s.total ? 0.55 : 1}
                stroke="var(--surface-1)"
                strokeWidth={2}
              />
            ))}
            {/* explicit renderer: the built-in position math is wrong for floating (range) bars */}
            <LabelList
              dataKey="contribution"
              content={({ x, y, width, height, value }) => {
                const v = Number(value);
                return (
                  <text
                    x={Number(x) + Number(width) + 6}
                    y={Number(y) + Number(height) / 2}
                    dominantBaseline="central"
                    fontSize={11}
                    fill="var(--text-secondary)"
                  >
                    {`${v > 0 ? "+" : ""}${v.toFixed(2)}`}
                  </text>
                );
              }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ----------------------------------------------------------------- PSI */
const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];

export function PsiOverTime({ points, features, threshold = 0.2 }: {
  points: Array<Record<string, number | string>>;
  features: string[];
  threshold?: number;
}) {
  const shown = features.slice(0, 3); // categorical cap: 3 slots validate all-pairs
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={points} margin={{ left: 0, right: 90, top: 10, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke="var(--gridline)" />
        <XAxis dataKey="date" {...AXIS} />
        <YAxis {...AXIS} width={40} tickFormatter={(v: number) => v.toFixed(2)} />
        <ReferenceLine
          y={threshold}
          stroke="var(--status-critical)"
          strokeDasharray="4 4"
          label={{ value: `significant (${threshold})`, fill: "var(--text-secondary)", fontSize: 11, position: "insideTopLeft" }}
        />
        <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => v.toFixed(3)} />
        <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-secondary)" }} />
        {shown.map((f, i) => (
          <Line
            key={f}
            dataKey={f}
            name={f}
            stroke={SERIES[i]}
            strokeWidth={2}
            dot={{ r: 4, strokeWidth: 2, stroke: "var(--surface-1)", fill: SERIES[i] }}
            activeDot={{ r: 5 }}
            isAnimationActive={false}
          >
            <LabelList
              dataKey={f}
              content={({ x, y, index }) =>
                index === points.length - 1 ? (
                  <text x={Number(x) + 8} y={Number(y) + 4} fontSize={11} fill="var(--text-secondary)">
                    {f.length > 14 ? `${f.slice(0, 13)}…` : f}
                  </text>
                ) : null
              }
            />
          </Line>
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PsiBars({ psi, threshold = 0.2 }: { psi: Record<string, number>; threshold?: number }) {
  const data = Object.entries(psi)
    .map(([feature, value]) => ({ feature, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 12);
  return (
    <ResponsiveContainer width="100%" height={28 * data.length + 30}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 56, top: 4, bottom: 4 }} barCategoryGap={4}>
        <CartesianGrid horizontal={false} stroke="var(--gridline)" />
        <XAxis type="number" {...AXIS} tickFormatter={(v: number) => v.toFixed(2)} />
        <YAxis type="category" dataKey="feature" width={190} {...AXIS} tick={{ ...AXIS.tick, fill: "var(--text-secondary)" }} />
        <ReferenceLine x={threshold} stroke="var(--status-critical)" strokeDasharray="4 4" />
        <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [v.toFixed(3), "PSI"]} cursor={{ fill: "var(--gridline)", opacity: 0.4 }} />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.feature} fill={d.value > threshold ? "var(--status-critical)" : "var(--series-1)"} />
          ))}
          <LabelList
            dataKey="value"
            position="right"
            formatter={(v: number) => (v > threshold ? `${v.toFixed(2)} ⚠` : v.toFixed(2))}
            style={{ fill: "var(--text-secondary)", fontSize: 11 }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
