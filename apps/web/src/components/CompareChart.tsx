import { useState } from "react";

/** Grouped bar chart (model x metric) with a hover/focus tooltip - real
 * client-side render over the same data the table below shows, replacing
 * the old static matplotlib PNG so values are reachable on mouse-move (a
 * flat image can't do that). Colors are validated CVD-safe (see the
 * `dataviz` skill's palette validator) and assigned by model identity, not
 * by position, so a given model always gets the same color regardless of
 * what order the API returns it in. */

type CompareData = Record<string, Record<string, number>>;

// Categorical slots 1-3 (dark-mode steps), validated all-pairs CVD-safe
// together - see references/palette.md in the dataviz skill.
const COLOR_BY_KEYWORD: [pattern: RegExp, color: string][] = [
  [/random forest/i, "#3987e5"], // slot 1 - blue
  [/u-?net/i, "#d95926"], // slot 2 - orange
  [/quantum/i, "#199e70"], // slot 3 - aqua
];
const FALLBACK_COLORS = ["#c98500", "#d55181", "#9085e9", "#e66767"]; // slots 4-7, for any unexpected extra series

function colorForModel(name: string, fallbackIndex: number): string {
  for (const [pattern, color] of COLOR_BY_KEYWORD) {
    if (pattern.test(name)) return color;
  }
  return FALLBACK_COLORS[fallbackIndex % FALLBACK_COLORS.length];
}

function niceTicks(max: number, count = 5): number[] {
  const step = max / count;
  return Array.from({ length: count + 1 }, (_, i) => +(step * i).toFixed(4));
}

export default function CompareChart({ data }: { data: CompareData }) {
  const [hovered, setHovered] = useState<string | null>(null);

  const models = Object.keys(data);
  const metrics = Array.from(new Set(models.flatMap((m) => Object.keys(data[m]))));
  if (models.length === 0 || metrics.length === 0) return null;

  const allValues = models.flatMap((m) => metrics.map((met) => data[m][met] ?? 0));
  const yMax = Math.max(1, ...allValues);
  const ticks = niceTicks(yMax);

  const W = 760;
  const H = 340;
  const marginLeft = 44;
  const marginBottom = 32;
  const marginTop = 12;
  const plotW = W - marginLeft - 12;
  const plotH = H - marginTop - marginBottom;

  const groupW = plotW / metrics.length;
  const barGap = 3; // surface gap between adjacent bars in a group
  const groupPad = 10; // air between groups
  const barW = Math.min(24, (groupW - groupPad * 2 - barGap * (models.length - 1)) / models.length);

  const yFor = (v: number) => marginTop + plotH * (1 - v / yMax);
  const colors = models.map((m, i) => colorForModel(m, i));

  const hoveredMetric = hovered;
  const hoveredGroupIndex = hoveredMetric ? metrics.indexOf(hoveredMetric) : -1;

  return (
    <div className="compare-chart">
      <div className="compare-chart-legend">
        {models.map((m, i) => (
          <span className="legend-item" key={m}>
            <span className="legend-swatch" style={{ background: colors[i] }} />
            {m}
          </span>
        ))}
      </div>

      <div className="compare-chart-svg-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Classical vs quantum model comparison by metric">
          {/* gridlines */}
          {ticks.map((t) => (
            <line
              key={t}
              x1={marginLeft}
              x2={W - 12}
              y1={yFor(t)}
              y2={yFor(t)}
              className="chart-gridline"
            />
          ))}
          {/* y-axis ticks */}
          {ticks.map((t) => (
            <text key={t} x={marginLeft - 8} y={yFor(t)} textAnchor="end" dominantBaseline="middle" className="chart-axis-text">
              {t.toFixed(t < 1 && t !== 0 ? 1 : 0)}
            </text>
          ))}

          {metrics.map((metric, gi) => {
            const groupX = marginLeft + gi * groupW;
            const isGroupHovered = hoveredMetric === metric;
            return (
              <g key={metric}>
                {/* full-height hit area for this group - bigger than the bars themselves */}
                <rect
                  x={groupX}
                  y={marginTop}
                  width={groupW}
                  height={plotH}
                  fill="transparent"
                  tabIndex={0}
                  role="button"
                  aria-label={`${metric}: ${models.map((m) => `${m} ${(data[m][metric] ?? 0).toFixed(4)}`).join(", ")}`}
                  onPointerEnter={() => setHovered(metric)}
                  onPointerLeave={() => setHovered(null)}
                  onFocus={() => setHovered(metric)}
                  onBlur={() => setHovered(null)}
                  className="chart-hit-area"
                />
                {models.map((model, mi) => {
                  const value = data[model][metric] ?? 0;
                  const x = groupX + groupPad + mi * (barW + barGap);
                  const y = yFor(value);
                  const h = marginTop + plotH - y;
                  return (
                    <rect
                      key={model}
                      x={x}
                      y={y}
                      width={barW}
                      height={Math.max(h, 0)}
                      rx={4}
                      fill={colors[mi]}
                      opacity={isGroupHovered || hoveredMetric === null ? 1 : 0.45}
                      className="chart-bar"
                    />
                  );
                })}
                <text x={groupX + groupW / 2} y={H - marginBottom + 16} textAnchor="middle" className="chart-axis-text">
                  {metric}
                </text>
              </g>
            );
          })}

          <line x1={marginLeft} x2={marginLeft} y1={marginTop} y2={marginTop + plotH} className="chart-axis-line" />
          <line x1={marginLeft} x2={W - 12} y1={marginTop + plotH} y2={marginTop + plotH} className="chart-axis-line" />
        </svg>

        {hoveredMetric && hoveredGroupIndex >= 0 && (
          <div
            className="chart-tooltip"
            style={{
              left: `${((marginLeft + hoveredGroupIndex * groupW + groupW / 2) / W) * 100}%`,
            }}
          >
            <div className="chart-tooltip-header">{hoveredMetric}</div>
            {models.map((model, i) => (
              <div className="chart-tooltip-row" key={model}>
                <span className="chart-tooltip-key" style={{ background: colors[i] }} />
                <span className="chart-tooltip-label">{model}</span>
                <span className="chart-tooltip-value">{(data[model][hoveredMetric] ?? 0).toFixed(4)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
