import PageNav from '../components/PageNav'
import SectionHeader from '../components/SectionHeader'
import DiagramCard from '../components/DiagramCard'
import Callout from '../components/Callout'
import { diagrams, rq2RealPriceMAE, rq2RetrainTimeRankings, rq2SyntheticTable } from '../data/results'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LabelList, Legend
} from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-dash-card border border-dash-border rounded-lg p-3 text-xs">
      <p className="font-semibold text-dash-text mb-2">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color ?? p.fill }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}
        </p>
      ))}
    </div>
  )
}

const improvementData = rq2RealPriceMAE.map((d) => ({
  model: d.model,
  improvement: d.improvement,
  color: d.improvement > 0 ? '#5a9e7c' : '#525252',
}))

export default function RQ2() {
  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <SectionHeader
        eyebrow="Research Question 2"
        title="TDTR vs Non-Adaptive Baselines"
        color="#9d7bb5"
        desc="How effectively does Trial-Based Drift-Triggered Retraining (TDTR) improve forecasting performance compared
        to non-adaptive baselines across synthetic drift scenarios and
        real financial data?"
      />

      <Callout color="#9d7bb5">
        <strong className="text-dash-text">Key Finding:</strong> TDTR reduces Price MAE by{' '}
        <strong className="text-dash-accent">41%</strong> for LSTM and{' '}
        <strong className="text-dash-accent">31%</strong> for PSO-LSTM on real S&P 500 data.
        ELM-based models show no significant benefit due to strong static generalisation from analytical weight solutions.
        KSWIN outperforms ADWIN and Page-Hinkley as the primary drift detector.
      </Callout>

      {/* Price MAE comparison + improvement */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-dash-card border border-dash-border rounded-xl p-5">
          <p className="text-[12px] font-bold uppercase tracking-widest text-dash-muted mb-1">Real Data - S&P 500 (GSPC)</p>
          <p className="text-[15px] font-semibold text-dash-text mb-4">Price MAE: Baseline vs TDTR</p>
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={rq2RealPriceMAE} barCategoryGap="30%">
              <CartesianGrid strokeDasharray="3 3" stroke="#272727" />
              <XAxis dataKey="model" tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} domain={[0, 40]} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: '#8a8a8a' }} />
              <Bar dataKey="baseline" name="Baseline" fill="#b05c5c" radius={[3, 3, 0, 0]} />
              <Bar dataKey="adaptive" name="TDTR"     fill="#c9a84c" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-dash-card border border-dash-border rounded-xl p-5">
          <p className="text-[12px] font-bold uppercase tracking-widest text-dash-muted mb-1">Improvement (%)</p>
          <p className="text-[15px] font-semibold text-dash-text mb-4">Price MAE Reduction via TDTR</p>
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={improvementData} barCategoryGap="40%">
              <CartesianGrid strokeDasharray="3 3" stroke="#272727" />
              <XAxis dataKey="model" tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} domain={[0, 50]} unit="%" />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="improvement" name="Improvement %" radius={[4, 4, 0, 0]}>
                <LabelList dataKey="improvement" position="top" style={{ fill: '#8a8a8a', fontSize: 11 }} formatter={(v) => `${v}%`} />
                {improvementData.map((d) => <Cell key={d.model} fill={d.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* CD diagrams */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <DiagramCard
          src={diagrams.rq2CdReturnMAE}
          title="Critical Difference - Return MAE (All Models)"
          caption="PSO-ELM static (rank 2.23) remains best overall. Connected groups are not significantly different. TDTR adaptive variants rank lower on aggregate synthetic results where ELM already generalises well."
          lightBg
        />
        <DiagramCard
          src={diagrams.rq2CdRetrainTime}
          title="Critical Difference - Total Retrain Time"
          caption="ELM adaptive is fastest (rank 1.00) due to its analytical closed-form solution. PSO-LSTM is slowest (rank 3.99) due to PSO overhead at every retraining event."
          lightBg
        />
      </div>

      {/* Retrain time bar chart */}
      <div className="bg-dash-card border border-dash-border rounded-xl p-5">
        <p className="text-[12px] font-bold uppercase tracking-widest text-dash-muted mb-1">Computational Cost</p>
        <p className="text-[15px] font-semibold text-dash-text mb-4">Total Retrain Time - Average Rank (lower = faster)</p>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={rq2RetrainTimeRankings} layout="vertical" barCategoryGap="25%">
            <CartesianGrid strokeDasharray="3 3" stroke="#272727" horizontal={false} />
            <XAxis type="number" domain={[0, 5]} tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="model" tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} width={130} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="rank" name="Rank" radius={[0, 4, 4, 0]}>
              <LabelList dataKey="rank" position="right" style={{ fill: '#8a8a8a', fontSize: 11 }} formatter={(v) => v.toFixed(2)} />
              {rq2RetrainTimeRankings.map((d) => <Cell key={d.model} fill={d.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Summary table */}
      <div className="bg-dash-card border border-dash-border rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-dash-border">
          <p className="text-[15px] font-bold text-dash-text">RQ2 Results - S&P 500 (GSPC)</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-dash-border">
                {['Model', 'Baseline Price MAE', 'TDTR Price MAE', 'Reduction', 'Return MAE (TDTR)'].map((h) => (
                  <th key={h} className="text-left text-[11px] font-bold uppercase tracking-widest text-dash-dim px-5 py-3">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { model: 'LSTM',     base: '34.70', tdtr: '20.35', red: '41%', ret: '0.00749', gain: true  },
                { model: 'PSO-LSTM', base: '28.86', tdtr: '19.87', red: '31%', ret: '0.00724', gain: true  },
                { model: 'ELM',      base: '19.83', tdtr: '20.25', red: '0%',  ret: '0.00739', gain: false },
                { model: 'PSO-ELM',  base: '19.98', tdtr: '20.48', red: '0%',  ret: '0.00747', gain: false },
              ].map((r) => (
                <tr key={r.model} className="border-b border-dash-border/50 hover:bg-dash-surface transition-colors">
                  <td className="px-5 py-3 font-bold font-mono text-dash-text">{r.model}</td>
                  <td className="px-5 py-3 text-dash-muted font-mono">{r.base}</td>
                  <td className="px-5 py-3 text-dash-muted font-mono">{r.tdtr}</td>
                  <td className={`px-5 py-3 font-bold font-mono ${r.gain ? 'text-dash-green' : 'text-dash-dim'}`}>{r.red}</td>
                  <td className="px-5 py-3 text-dash-muted font-mono">{r.ret}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {/* Synthetic results table — matches report format */}
      <div className="bg-dash-card border border-dash-border rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-dash-border">
          <p className="text-[15px] font-bold text-dash-text">
            Synthetic Benchmark - Return MAE by Drift Type (mean and std, 30 runs)
          </p>
          <p className="text-[12px] text-dash-dim mt-0.5">
            Bold = lowest mean per drift type within each model group. Values shown as mean (std).
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[13px] border-collapse">
            <thead>
              {/* Group header row */}
              <tr>
                <th className="px-4 py-2.5 text-left text-[12px] text-dash-dim font-semibold w-36 bg-[#171717]" rowSpan={2}>
                  Drift Type
                </th>
                <th className="px-4 py-2.5 text-center text-[12px] font-bold text-dash-muted bg-[#1e1e1e] border-b border-dash-border" colSpan={5}>
                  Static Models
                </th>
                <th className="px-4 py-2.5 text-center text-[12px] font-bold text-dash-accent bg-[#1a1a1a] border-b border-dash-border" colSpan={4}>
                  Drift-Adaptive Models
                </th>
              </tr>
              <tr className="border-b-2 border-dash-border">
                {rq2SyntheticTable.staticModels.map((m) => (
                  <th key={m} className="px-4 py-2 text-center text-[12px] font-bold text-dash-muted bg-[#1e1e1e]">{m}</th>
                ))}
                {rq2SyntheticTable.adaptiveModels.map((m) => (
                  <th key={m} className="px-4 py-2 text-center text-[12px] font-bold text-dash-accent bg-[#1a1a1a]">{m}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rq2SyntheticTable.rows.map((row, rowIdx) => {
                const bestStatic   = Math.min(...row.static.map((c) => c.mean))
                const bestAdaptive = Math.min(...row.adaptive.map((c) => c.mean))
                const stripe = rowIdx % 2 === 0 ? 'bg-dash-card' : 'bg-[#141414]'
                return (
                  <tr key={row.drift} className={`border-b border-dash-border/60 hover:bg-dash-border/30 transition-colors ${stripe}`}>
                    <td className="px-4 py-3 font-semibold text-dash-text text-[13px] bg-[#1e1e1e]">{row.drift}</td>
                    {row.static.map((cell, i) => {
                      const isBest = cell.mean === bestStatic
                      return (
                        <td key={i} className="px-4 py-3 text-center">
                          <span className={`block text-[13px] ${isBest ? 'font-bold text-dash-text' : 'text-dash-muted'}`}>
                            {cell.mean.toFixed(3)}
                          </span>
                          <span className="block text-[11px] text-dash-dim">({cell.std.toFixed(3)})</span>
                        </td>
                      )
                    })}
                    {row.adaptive.map((cell, i) => {
                      const isBest = cell.mean === bestAdaptive
                      return (
                        <td key={i} className="px-4 py-3 text-center">
                          <span className={`block text-[13px] ${isBest ? 'font-bold text-dash-accent' : 'text-dash-muted'}`}>
                            {cell.mean.toFixed(3)}
                          </span>
                          <span className="block text-[11px] text-dash-dim">({cell.std.toFixed(3)})</span>
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <p className="px-5 py-3 text-[14px] text-dash-dim italic border-t border-dash-border">
          Note: Mean and standard deviation over 30 runs. Bold indicates the lowest mean MAE per drift type within each model category (Static and Adaptive).
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {[
          { title: 'KSWIN selected as best detector',              color: '#e8e8e8', body: 'KSWIN achieved the highest recall (0.89, detecting 11 out of 12 true drifts), though at the cost of more false positives (55), which TDTR\'s cooldown and trial mechanism helps absorb.' },
          { title: 'TDTR strongly helps LSTM on real data',        color: '#e8e8e8', body: 'LSTM improves by 41% (34.70 → 20.35) and PSO-LSTM by 31% (28.86 → 19.87) in Price MAE, confirming that recurrent models accumulate errors over time and benefit greatly from retraining.' },
          { title: 'ELM-based models do not benefit from TDTR',   color: '#e8e8e8', body: 'Static ELM already sits at the performance ceiling (~19.83). Retraining on a short window actually worsens performance slightly by replacing a stable global fit with a noise-sensitive local one.' },
          { title: 'Adaptation is mixed on synthetic data',        color: '#e8e8e8', body: 'Static PSO-ELM remains the best performer overall (average rank 2.50), while ELM-A ranks worst (6.36), reinforcing that retraining hurts feed-forward models on recurring benchmarks.' },
          { title: 'The ceiling effect is confirmed',              color: '#e8e8e8', body: 'All adaptive models converge to roughly the same Price MAE (~20) on real data, regardless of architecture.' },
          { title: 'PSO-based models are far more expensive',      color: '#e8e8e8', body: 'PSO-LSTM-A takes 99.07s total vs 1.11s for LSTM-A to retrain, with no corresponding accuracy gain.' },
        ].map((c) => (
          <div key={c.title} className="bg-dash-card border border-dash-border rounded-xl p-4" style={{ borderTopColor: c.color, borderTopWidth: 2 }}>
            <p className="text-[13px] font-bold mb-1.5" style={{ color: c.color }}>{c.title}</p>
            <p className="text-[13px] text-dash-muted leading-relaxed">{c.body}</p>
          </div>
        ))}
      </div>

      <PageNav />
    </div>
  )
}
