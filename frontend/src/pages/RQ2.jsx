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
        desc="How effectively does TDTR improve forecasting performance compared to non-adaptive baselines across synthetic drift scenarios and real S&P 500 financial data?"
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
                {['Model', 'Baseline Price MAE', 'TDTR Price MAE', 'Reduction', 'Return MAE (TDTR)', 'ELM Ceiling?'].map((h) => (
                  <th key={h} className="text-left text-[11px] font-bold uppercase tracking-widest text-dash-dim px-5 py-3">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { model: 'LSTM',     base: '~33.9', tdtr: '~19.8', red: '41%', ret: '0.0072', ceiling: false },
                { model: 'PSO-LSTM', base: '~28.6', tdtr: '~19.7', red: '31%', ret: '0.0072', ceiling: false },
                { model: 'PSO-ELM', base: '~20.2', tdtr: '~20.2', red: '~0%', ret: '0.0074', ceiling: true  },
                { model: 'ELM',     base: '~20.1', tdtr: '~20.1', red: '~0%', ret: '0.0073', ceiling: true  },
              ].map((r) => (
                <tr key={r.model} className="border-b border-dash-border/50 hover:bg-dash-surface transition-colors">
                  <td className="px-5 py-3 font-bold font-mono text-dash-text">{r.model}</td>
                  <td className="px-5 py-3 text-dash-red">{r.base}</td>
                  <td className="px-5 py-3 text-dash-green">{r.tdtr}</td>
                  <td className={`px-5 py-3 font-bold ${r.ceiling ? 'text-dash-dim' : 'text-dash-green'}`}>{r.red}</td>
                  <td className="px-5 py-3 text-dash-muted font-mono">{r.ret}</td>
                  <td className="px-5 py-3">
                    <span className={`text-[11px] font-bold px-2 py-0.5 rounded ${r.ceiling ? 'bg-dash-yellow/10 text-dash-yellow' : 'bg-dash-green/10 text-dash-green'}`}>
                      {r.ceiling ? 'Yes' : 'No'}
                    </span>
                  </td>
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

      <PageNav />
    </div>
  )
}
