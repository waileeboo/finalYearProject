import PageNav from '../components/PageNav'
import SectionHeader from '../components/SectionHeader'
import DiagramCard from '../components/DiagramCard'
import Callout from '../components/Callout'
import { diagrams } from '../data/results'
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


export default function RQ5() {
  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <SectionHeader
        eyebrow="Research Question 5"
        title="Limited Training - Unseen Concepts"
        color="#e8e8e8"
        desc="Does the TDTR framework improve forecasting
        performance over non-adaptive baselines when models are
        trained on limited concept coverage and evaluated on unseen
        concept configurations?"
      />

      <Callout color="#e8e8e8">
        <strong className="text-dash-text">Key Finding:</strong> Averaged across all drift types, ELM-A achieves the largest gain (
        <strong className="text-dash-accent">+69%</strong>) and LSTM-A follows with{' '}
        <strong className="text-dash-accent">+55%</strong>.
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
        <p className="text-[12px] font-bold uppercase tracking-widest text-dash-muted mb-1">Average Across All Drift Types</p>
        <p className="text-[15px] font-semibold text-dash-text mb-1">Return MAE Reduction: TDTR vs Static Baseline (%)</p>
        <p className="text-[12px] text-dash-dim mb-4">Averaged across all 4 drift types (30 runs each). Higher = more improvement from TDTR.</p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={[
            { model: 'ELM',      gain: 69, fill: '#5a9e7c' },
            { model: 'LSTM',     gain: 55, fill: '#c9a84c' },
            { model: 'PSO-ELM',  gain: 7,  fill: '#9d7bb5' },
            { model: 'PSO-LSTM', gain: -4, fill: '#b05c5c' },
          ]} barCategoryGap="40%">
            <CartesianGrid strokeDasharray="3 3" stroke="#272727" />
            <XAxis dataKey="model" tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} domain={[-10, 80]} unit="%" />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="gain" name="MAE Reduction %" radius={[4, 4, 0, 0]}>
              <LabelList dataKey="gain" position="top" style={{ fill: '#8a8a8a', fontSize: 11 }} formatter={(v) => `${v}%`} />
              {[
                { model: 'ELM', gain: 69, fill: '#5a9e7c' },
                { model: 'LSTM', gain: 55, fill: '#c9a84c' },
                { model: 'PSO-ELM', gain: 7, fill: '#9d7bb5' },
                { model: 'PSO-LSTM', gain: -4, fill: '#b05c5c' },
              ].map((d) => <Cell key={d.model} fill={d.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Baseline vs Adaptive table */}
      <div className="bg-dash-card border border-dash-border rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-dash-border">
          <p className="text-[15px] font-bold text-dash-text">RQ5 Results — Average Return MAE Across All Drift Types</p>
          <p className="text-[12px] text-dash-dim mt-0.5">Mean over 4 drift types × 30 runs each.</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-dash-border">
                {['Model', 'Static Avg MAE', 'Adaptive Avg MAE', 'Improvement', 'Interpretation'].map((h) => (
                  <th key={h} className="text-left text-[11px] font-bold uppercase tracking-widest text-dash-dim px-5 py-3">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { model: 'ELM',      base: '0.198', tdtr: '0.062', gain: '+69%', good: true,  note: 'Largest gain!! random fixed inputs collapse under unseen concepts; retraining rescues it' },
                { model: 'LSTM',     base: '0.098', tdtr: '0.044', gain: '+55%', good: true,  note: 'LSTM weights trained on old concepts become outdated; retraining on recent data fixes this' },
                { model: 'PSO-ELM',  base: '0.044', tdtr: '0.041', gain: '+7%',  good: true,  note: 'Already starts strong. PSO provides implicit robustness, little room to improve' },
                { model: 'PSO-LSTM', base: '0.051', tdtr: '0.054', gain: '-4%',  good: false, note: 'Partial output-layer retraining adds noise rather than signal' },
              ].map((r) => (
                <tr key={r.model} className="border-b border-dash-border/50 hover:bg-dash-surface transition-colors">
                  <td className="px-5 py-3 font-bold font-mono text-dash-text">{r.model}</td>
                  <td className="px-5 py-3 text-dash-muted font-mono">{r.base}</td>
                  <td className="px-5 py-3 text-dash-muted font-mono">{r.tdtr}</td>
                  <td className={`px-5 py-3 font-bold font-mono ${r.good ? 'text-dash-green' : 'text-dash-dim'}`}>{r.gain}</td>
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
          the first concept. When it encounters genuinely unseen regimes (concepts 5-7), its error spikes sharply.
          TDTR's drift detector catches this spike, triggers retraining on a recent window from the new concept, and
          the challenger model learns the new regime. The gain is far larger than in the full-training scenario because
          the static baseline has much more to lose - it cannot generalise to truly novel distributions without adaptation.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {[
          { title: 'Static baselines collapse under limited training',         color: '#e8e8e8', body: 'ELM static avg MAE rises to 0.198 and LSTM to 0.098, roughly triple the RQ2 values, confirming both architectures struggle badly when they have never seen the incoming concept during training.' },
          { title: 'TDTR produces its largest gains across the whole study',   color: '#e8e8e8', body: 'Averaged across all 4 drift types, ELM-A improves by 69% (0.198 → 0.062) and LSTM-A by 55% (0.098 → 0.044), the largest improvements observed across the entire study.' },
          { title: 'Full retraining beats partial retraining for LSTM',        color: '#e8e8e8', body: 'LSTM-A outperforms PSO-LSTM-A under linear abrupt (0.046 vs 0.086) and linear gradual drift (0.041 vs 0.050), proving that updating only the output layer is insufficient when the model encounters a genuinely new concept.' },
          { title: 'PSO-ELM-A is the best adaptive ELM variant',               color: '#e8e8e8', body: 'PSO re-optimises input weights at every retrain, actively reshaping feature projections to fit the new distribution and consistently beating plain ELM-A across all drift types.' },
          { title: 'PSO-based static models remain surprisingly competitive',  color: '#e8e8e8', body: 'PSO-ELM (avg 0.044) and PSO-LSTM (avg 0.051) degrade far less than ELM (0.198) and LSTM (0.098) under limited training, suggesting swarm-based optimisation provides an implicit robustness that reduces the need for adaptation even under unseen concepts.' },
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
