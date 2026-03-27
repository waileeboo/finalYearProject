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
  { model: 'RF Baseline', priceMAE: 20.46, fill: '#c47a3a' },
  { model: 'RF TDTR',     priceMAE: 20.65, fill: '#c9a84c' },
  { model: 'SVR Baseline',priceMAE: 27.50, fill: '#c47a3a' },
  { model: 'SVR TDTR',    priceMAE: 23.71, fill: '#b05c5c' },
]

const returnData = [
  { model: 'RF Baseline', returnMAE: 0.00749, fill: '#c47a3a' },
  { model: 'RF TDTR',     returnMAE: 0.00756, fill: '#c9a84c' },
  { model: 'SVR Baseline',returnMAE: 0.01005, fill: '#c47a3a' },
  { model: 'SVR TDTR',    returnMAE: 0.00913, fill: '#b05c5c' },
]

export default function RQ4() {
  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <SectionHeader
        eyebrow="Research Question 4"
        title="Conventional ML - Random Forest & SVR"
        color="#c47a3a"
        desc="Can conventional machine learning methods such as
        Random Forest and Support Vector Regression (SVR) benefit from drift adaptation within the TDTR framework, and
        do they exhibit the same ceiling effect observed in analytical
        neural models?"
      />

      <Callout color="#c47a3a">
        <strong className="text-dash-text">Key Finding: </strong> 
        SVR benefits clearly from TDTR, reducing Price MAE by 14% (27.50 → 23.71), mirroring the LSTM pattern in RQ2. Random Forest shows no benefit — it already sits near the performance ceiling (20.46), consistent with ELM in RQ2, confirming the ceiling is a dataset-level barrier rather than architecture-specific.
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
              <YAxis tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} domain={[18, 30]} />
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
            baseline: '20.46', tdtr: '20.65', gain: 'None (~0%)',
            explanation: 'RF benefits from fresh bootstrap samples on new data windows when retrained. However, the ensemble averaging smooths out distributional changes, limiting the gain. Improvements are not statistically significant across 30 runs.',
            verdict: 'No benefit',
          },
          {
            model: 'SVR',
            color: '#e8e8e8',
            baseline: '27.50', tdtr: '23.71', gain: '14% reduction',
            explanation: 'SVR starts well above the performance ceiling and TDTR pulls it down from 27.50 to 23.71, mirroring the LSTM pattern in RQ2. Models with more room to improve benefit most from drift-triggered retraining.',
            verdict: 'Clear benefit',
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
          On real data, ELM and RF exhibit a <em>ceiling effect</em> under TDTR — both already sit near Price MAE ~20 and gain nothing from retraining. SVR, starting well above the ceiling (27.50), benefits clearly (→ 23.71). The contrast with LSTM (41% gain) reveals the underlying mechanism:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {[
            { label: 'Ceiling effect generalises beyond neural models', color: '#e8e8e8', body: 'RF already sits near the performance ceiling (Price MAE 20.46) and gains nothing from TDTR, consistent with the ELM finding in RQ2, confirming the ceiling is a dataset-level barrier rather than an architecture-specific one.' },
            { label: 'SVR benefits clearly from TDTR', color: '#e8e8e8', body: 'Static SVR is well above the ceiling (Price MAE 27.50) and adaptation drives it down to 23.71, mirroring the LSTM pattern in RQ2 where weaker baselines have more room to improve.' },
            { label: 'Adaptation is drift-type dependent on synthetic data', color: '#e8e8e8', body: 'Retraining hurts slightly under gradual drift, where both RF and SVR worsen after adaptation, consistent with the ELM finding in RQ2. However, under abrupt drift both models improve substantially, with RF dropping from 0.073 to 0.044 and SVR from 0.112 to 0.070, confirming that the benefit of retraining depends on how sudden and severe the structural shift is.' },
          ].map((c) => (
            <div key={c.label} className="bg-dash-card border border-dash-border rounded-xl p-4" style={{ borderTopColor: c.color, borderTopWidth: 2 }}>
              <p className="text-[13px] font-bold mb-1.5" style={{ color: c.color }}>{c.label}</p>
              <p className="text-[13px] text-dash-muted leading-relaxed">{c.body}</p>
            </div>
          ))}
        </div>
      </div>
      <PageNav />
    </div>
  )
}
