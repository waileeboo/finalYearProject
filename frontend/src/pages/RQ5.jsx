import PageNav from '../components/PageNav'
import SectionHeader from '../components/SectionHeader'
import DiagramCard from '../components/DiagramCard'
import Callout from '../components/Callout'
import { diagrams, rq5UnseenGain } from '../data/results'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LabelList
} from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-dash-card border border-dash-border rounded-lg p-3 text-xs">
      <p className="font-semibold text-dash-text mb-2">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }}>{p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</p>
      ))}
    </div>
  )
}

const gainColors = ['#5a9e7c', '#c9a84c', '#c47a3a', '#9d7bb5']

export default function RQ5() {
  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <SectionHeader
        eyebrow="Research Question 5"
        title="Limited Training - Unseen Concepts"
        color="#e8e8e8"
        desc="Does TDTR improve forecasting performance when models are trained on only the first two concepts (10% of the series) and evaluated on genuinely unseen concept configurations?"
      />

      <Callout color="#e8e8e8">
        <strong className="text-dash-text">Key Finding:</strong> TDTR produces gains of up to{' '}
        <strong className="text-dash-red">70%</strong> (PSO-LSTM) over non-adaptive baselines on genuinely
        unseen concept configurations. Gains are strongest in the unseen concept window (concepts 5–7) and
        moderate in the recurring concept window (concepts 8–10). LSTM achieves 57% gain.
      </Callout>

      {/* Experiment setup */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {[
          { label: 'Training Coverage', value: '10%', desc: 'First 2 concepts only (1,000 steps of 10,000)', color: '#e8e8e8' },
          { label: 'Unseen Zone',       value: 'C5–7', desc: 'Concepts 5–7: genuinely unseen configurations', color: '#e8e8e8' },
          { label: 'Recurring Zone',    value: 'C8–10', desc: 'Concepts 8–10: recurring (seen during training)', color: '#e8e8e8' },
        ].map((s) => (
          <div key={s.label} className="bg-dash-card border border-dash-border rounded-xl p-4">
            <p className="text-[11px] font-bold uppercase tracking-widest text-dash-dim mb-1">{s.label}</p>
            <p className="text-2xl font-extrabold mb-1" style={{ color: s.color }}>{s.value}</p>
            <p className="text-[13px] text-dash-muted">{s.desc}</p>
          </div>
        ))}
      </div>

      {/* Rolling MAE diagrams */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <DiagramCard
          src={diagrams.rq5ELMRolling}
          title="ELM Baseline - Rolling MAE on Test Set"
          caption="Linear Abrupt Drift, Series 1. Rolling MAE spikes in the unseen zone (pink) and drops in the recurring zone. TDTR adaptive version significantly reduces error in the unseen window."
        />
        <DiagramCard
          src={diagrams.rq5LSTMRolling}
          title="LSTM Baseline - Rolling MAE on Test Set"
          caption="Linear Abrupt Drift, Series 1. LSTM shows a similar pattern but recovers faster in recurring concepts. The drift point (red dashed) marks where the test set begins."
        />
      </div>

      {/* Gain chart */}
      <div className="bg-dash-card border border-dash-border rounded-xl p-5">
        <p className="text-[12px] font-bold uppercase tracking-widest text-dash-muted mb-1">Unseen Concept Window</p>
        <p className="text-[15px] font-semibold text-dash-text mb-1">Rolling MAE Reduction: TDTR vs Static Baseline (%)</p>
        <p className="text-[12px] text-dash-dim mb-4">Averaged across linear abrupt drift series. Higher = more improvement from TDTR.</p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={rq5UnseenGain} barCategoryGap="40%">
            <CartesianGrid strokeDasharray="3 3" stroke="#272727" />
            <XAxis dataKey="model" tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} domain={[0, 80]} unit="%" />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="gain" name="MAE Reduction %" radius={[4, 4, 0, 0]}>
              <LabelList dataKey="gain" position="top" style={{ fill: '#8a8a8a', fontSize: 11 }} formatter={(v) => `${v}%`} />
              {rq5UnseenGain.map((d, i) => <Cell key={d.model} fill={gainColors[i]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Baseline vs Adaptive table */}
      <div className="bg-dash-card border border-dash-border rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-dash-border">
          <p className="text-[15px] font-bold text-dash-text">RQ5 Results - Unseen Concept Window (Approx. Rolling MAE)</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-dash-border">
                {['Model', 'Baseline Rolling MAE', 'TDTR Rolling MAE', 'Gain (%)', 'Interpretation'].map((h) => (
                  <th key={h} className="text-left text-[11px] font-bold uppercase tracking-widest text-dash-dim px-5 py-3">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { model: 'PSO-LSTM', base: '0.098', tdtr: '0.029', gain: '70%', note: 'Largest gain - PSO weight reinit helps most under novelty' },
                { model: 'LSTM',     base: '0.112', tdtr: '0.048', gain: '57%', note: 'Large gain - retraining resets stale gradients effectively' },
                { model: 'ELM',      base: '0.087', tdtr: '0.071', gain: '18%', note: 'Modest gain - analytical model already generalises well' },
                { model: 'PSO-ELM',  base: '0.079', tdtr: '0.068', gain: '14%', note: 'Smallest gain - strong static generalisation limits benefit' },
              ].map((r, i) => (
                <tr key={r.model} className="border-b border-dash-border/50 hover:bg-dash-surface transition-colors">
                  <td className="px-5 py-3 font-bold font-mono text-dash-text">{r.model}</td>
                  <td className="px-5 py-3 text-dash-red font-mono">{r.base}</td>
                  <td className="px-5 py-3 text-dash-green font-mono">{r.tdtr}</td>
                  <td className="px-5 py-3 font-bold text-dash-green">{r.gain}</td>
                  <td className="px-5 py-3 text-dash-muted">{r.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Insight */}
      <div className="bg-dash-card border border-dash-border rounded-xl p-5" style={{ borderLeftColor: '#e8e8e8', borderLeftWidth: 3 }}>
        <p className="text-[15px] font-bold text-dash-text mb-2">Why Gains Are Larger Under Limited Training</p>
        <p className="text-[13px] text-dash-muted leading-relaxed">
          When a model is trained on only 10% of the concept space, its initial weights are heavily biased toward
          the first two concepts. When it encounters genuinely unseen regimes (concepts 5-7), its error spikes sharply.
          TDTR's drift detector catches this spike, triggers retraining on a recent window from the new concept, and
          the challenger model learns the new regime. The gain is far larger than in the full-training scenario because
          the static baseline has much more to lose - it cannot generalise to truly novel distributions without adaptation.
        </p>
      </div>
      <PageNav />
    </div>
  )
}
