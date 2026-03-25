import PageNav from '../components/PageNav'
import SectionHeader from '../components/SectionHeader'
import Callout from '../components/Callout'
import { rq4RealResults } from '../data/results'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend
} from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-dash-card border border-dash-border rounded-lg p-3 text-xs">
      <p className="font-semibold text-dash-text mb-2">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }}>{p.name}: {p.value?.toFixed?.(4) ?? p.value}</p>
      ))}
    </div>
  )
}

const priceData = [
  { model: 'RF Baseline', priceMAE: 20.8, fill: '#c47a3a' },
  { model: 'RF TDTR',     priceMAE: 20.4, fill: '#c9a84c' },
  { model: 'SVR Baseline',priceMAE: 23.7, fill: '#c47a3a' },
  { model: 'SVR TDTR',    priceMAE: 23.7, fill: '#b05c5c' },
]

const returnData = [
  { model: 'RF Baseline', returnMAE: 0.00753, fill: '#c47a3a' },
  { model: 'RF TDTR',     returnMAE: 0.00745, fill: '#c9a84c' },
  { model: 'SVR Baseline',returnMAE: 0.00913, fill: '#c47a3a' },
  { model: 'SVR TDTR',    returnMAE: 0.00911, fill: '#b05c5c' },
]

export default function RQ4() {
  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <SectionHeader
        eyebrow="Research Question 4"
        title="Conventional ML - Random Forest & SVR"
        color="#c47a3a"
        desc="Can conventional machine learning methods (Random Forest and SVR) benefit from drift adaptation within TDTR, and do they exhibit the same ceiling effect as analytical neural models?"
      />

      <Callout color="#c47a3a">
        <strong className="text-dash-text">Key Finding:</strong> Random Forest shows marginal and
        statistically non-significant improvements under TDTR. SVR shows no benefit whatsoever —
        identical results with and without adaptation. Both exhibit ceiling effects similar to ELM,
        suggesting they have already captured the exploitable signal in the training window.
      </Callout>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-dash-card border border-dash-border rounded-xl p-5">
          <p className="text-[12px] font-bold uppercase tracking-widest text-dash-muted mb-1">Real Data - S&P 500</p>
          <p className="text-[15px] font-semibold text-dash-text mb-4">Price MAE: Baseline vs TDTR</p>
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={priceData} barCategoryGap="30%">
              <CartesianGrid strokeDasharray="3 3" stroke="#272727" />
              <XAxis dataKey="model" tick={{ fill: '#8a8a8a', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} domain={[18, 26]} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="priceMAE" name="Price MAE" radius={[4, 4, 0, 0]}>
                {priceData.map((d) => <Cell key={d.model} fill={d.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-dash-card border border-dash-border rounded-xl p-5">
          <p className="text-[12px] font-bold uppercase tracking-widest text-dash-muted mb-1">Real Data - S&P 500</p>
          <p className="text-[15px] font-semibold text-dash-text mb-4">Return MAE: Baseline vs TDTR</p>
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={returnData} barCategoryGap="30%">
              <CartesianGrid strokeDasharray="3 3" stroke="#272727" />
              <XAxis dataKey="model" tick={{ fill: '#8a8a8a', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false}
                domain={[0.006, 0.010]} tickFormatter={(v) => v.toFixed(3)} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="returnMAE" name="Return MAE" radius={[4, 4, 0, 0]}>
                {returnData.map((d) => <Cell key={d.model} fill={d.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model explanations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[
          {
            model: 'Random Forest',
            color: '#e8e8e8',
            baseline: '20.8', tdtr: '20.4', gain: 'Marginal (~2%)',
            explanation: 'RF benefits from fresh bootstrap samples on new data windows when retrained. However, the ensemble averaging smooths out distributional changes, limiting the gain. Improvements are not statistically significant across 30 runs.',
            verdict: 'Marginal benefit',
          },
          {
            model: 'SVR',
            color: '#e8e8e8',
            baseline: '23.7', tdtr: '23.7', gain: 'None (~0%)',
            explanation: 'SVR\'s deterministic kernel mapping produces consistent predictions regardless of retraining. The support vectors from initial training already define a stable decision boundary that drift retraining does not meaningfully update.',
            verdict: 'No benefit',
          },
        ].map((m) => (
          <div key={m.model} className="bg-dash-card border border-dash-border rounded-xl p-5" style={{ borderTopColor: m.color, borderTopWidth: 2 }}>
            <div className="flex items-center justify-between mb-3">
              <p className="text-[14px] font-bold text-dash-text">{m.model}</p>
              <span className="text-[11px] font-bold px-2 py-0.5 rounded" style={{ background: `${m.color}18`, color: m.color }}>
                {m.verdict}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-3 mb-4">
              {[
                { label: 'Baseline MAE', value: m.baseline },
                { label: 'TDTR MAE',     value: m.tdtr     },
                { label: 'Gain',         value: m.gain      },
              ].map((s) => (
                <div key={s.label} className="bg-dash-surface rounded-lg p-2.5 text-center">
                  <p className="text-[11px] text-dash-dim mb-0.5">{s.label}</p>
                  <p className="text-[13px] font-bold text-dash-text">{s.value}</p>
                </div>
              ))}
            </div>
            <p className="text-[13px] text-dash-muted leading-relaxed">{m.explanation}</p>
          </div>
        ))}
      </div>

      {/* Ceiling effect insight */}
      <div className="bg-dash-card border border-dash-border rounded-xl p-5">
        <p className="text-[15px] font-bold text-dash-text mb-2">The Ceiling Effect Explained</p>
        <p className="text-[13px] text-dash-muted leading-relaxed mb-4">
          Both ELM and conventional ML models (RF, SVR) exhibit a <em>ceiling effect</em> under TDTR - their performance
          does not meaningfully improve with drift-triggered retraining. The contrast with LSTM (41% gain) reveals the
          underlying mechanism:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {[
            { label: 'ELM / RF / SVR', body: 'Analytical or kernel-based solutions that already exploit available structure in the training window. Retraining on a slightly different window yields essentially the same solution.', color: '#e8e8e8' },
            { label: 'LSTM (no ceiling)', body: 'Iterative gradient-based training gets "stuck" on stale gradients from the old concept. Fresh retraining on a recent window provides a genuine escape from the outdated solution.', color: '#e8e8e8' },
            { label: 'Practical Implication', body: 'TDTR should be prioritised for gradient-based models. For analytical/kernel models, invest computational budget elsewhere - adaptation does not help.', color: '#e8e8e8' },
          ].map((c) => (
            <div key={c.label} className="bg-dash-surface rounded-lg p-3.5 border border-dash-border">
              <p className="text-[13px] font-bold mb-1" style={{ color: c.color }}>{c.label}</p>
              <p className="text-[13px] text-dash-muted leading-relaxed">{c.body}</p>
            </div>
          ))}
        </div>
      </div>
      <PageNav />
    </div>
  )
}
