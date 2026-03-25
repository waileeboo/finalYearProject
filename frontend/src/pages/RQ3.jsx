import PageNav from '../components/PageNav'
import SectionHeader from '../components/SectionHeader'
import Callout from '../components/Callout'
import StatCard from '../components/StatCard'
import { rq3EfficiencyData } from '../data/results'
import { Zap, Target, Clock, TrendingDown } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-dash-card border border-dash-border rounded-lg p-3 text-xs">
      <p className="font-semibold text-dash-text mb-2">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }}>{p.name}: {p.value}</p>
      ))}
    </div>
  )
}

export default function RQ3() {
  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <SectionHeader
        eyebrow="Research Question 3"
        title="Continuous Retraining Ablation"
        color="#5a9e7c"
        desc="How does continuous online retraining without a drift detector compare to TDTR under identical retraining windows and computational constraints?"
      />

      <Callout color="#5a9e7c">
        <strong className="text-dash-text">Key Finding:</strong> Fixed-interval retraining matches TDTR in
        forecasting accuracy but at <strong className="text-dash-green">roughly double the computational cost</strong>.
        TDTR achieves comparable precision by retraining only when drift is detected, making it significantly more
        efficient without sacrificing accuracy. TDTR is strictly preferable under resource constraints.
      </Callout>

      {/* KPI row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard label="TDTR Retrains (avg)"     value="8"   sub="Only on detected drift events"    icon={Zap}       color="#5a9e7c" />
        <StatCard label="Fixed Retrains (avg)"    value="18"  sub="Every N steps regardless"         icon={Clock}     color="#e8e8e8" />
        <StatCard label="Accuracy Difference"     value="≈0%"  sub="Statistically equivalent MAE"    icon={Target}    color="#c9a84c" />
        <StatCard label="Cost Reduction"          value="~2×"  sub="TDTR vs fixed-interval"          icon={TrendingDown} color="#c47a3a" />
      </div>

      {/* Efficiency chart */}
      <div className="bg-dash-card border border-dash-border rounded-xl p-5">
        <p className="text-[12px] font-bold uppercase tracking-widest text-dash-muted mb-1">Ablation Comparison</p>
        <p className="text-[15px] font-semibold text-dash-text mb-4">TDTR vs Fixed-Interval Retraining - Key Metrics</p>
        <ResponsiveContainer width="100%" height={230}>
          <BarChart data={rq3EfficiencyData} barCategoryGap="30%">
            <CartesianGrid strokeDasharray="3 3" stroke="#272727" />
            <XAxis dataKey="metric" tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 11, color: '#8a8a8a' }} />
            <Bar dataKey="tdtr"  name="TDTR"           fill="#c9a84c" radius={[3, 3, 0, 0]} />
            <Bar dataKey="fixed" name="Fixed-Interval" fill="#b05c5c" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <p className="text-[12px] text-dash-dim mt-2">* Return MAE multiplied by 100 for scale. Retrains and Total Time (s) shown as absolute values.</p>
      </div>

      {/* Prose insight cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {[
          {
            icon: '≈',
            title: 'Accuracy Parity',
            body: 'Both strategies produce statistically equivalent Return MAE and Price MAE on real and synthetic data. No accuracy trade-off exists in favour of fixed-interval retraining.',
          },
          {
            icon: '⚡',
            title: '2× Efficiency',
            body: 'TDTR requires roughly half the retraining operations. It only retriggers when KSWIN detects a genuine distributional shift, avoiding the fixed-interval overhead in stable periods.',
          },
          {
            icon: '→',
            title: 'Practical Verdict',
            body: 'In resource-constrained deployments (edge devices, high-frequency trading), TDTR is strictly preferable. Fixed-interval retraining provides no benefit that justifies its extra cost.',
          },
        ].map((c) => (
          <div key={c.title} className="bg-dash-card border border-dash-border rounded-xl p-4" style={{ borderTopColor: '#e8e8e8', borderTopWidth: 2 }}>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-lg font-black text-dash-text">{c.icon}</span>
              <p className="text-[15px] font-bold text-dash-text">{c.title}</p>
            </div>
            <p className="text-[13px] text-dash-muted leading-relaxed">{c.body}</p>
          </div>
        ))}
      </div>

      {/* Detail table */}
      <div className="bg-dash-card border border-dash-border rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-dash-border">
          <p className="text-[15px] font-bold text-dash-text">Comparative Summary - TDTR vs Fixed-Interval (LSTM KSWIN on GSPC)</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-dash-border">
                {['Metric', 'TDTR', 'Fixed-Interval', 'Verdict'].map((h) => (
                  <th key={h} className="text-left text-[11px] font-bold uppercase tracking-widest text-dash-dim px-5 py-3">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { metric: 'Price MAE',          tdtr: '~19.8',  fixed: '~19.6',  verdict: 'Equivalent',           good: false },
                { metric: 'Return MAE',         tdtr: '~0.0072', fixed: '~0.0071', verdict: 'Equivalent',          good: false },
                { metric: 'Avg. Retrains',      tdtr: '8',      fixed: '18',     verdict: 'TDTR 55% fewer',        good: true  },
                { metric: 'Total Time (s)',      tdtr: '~62',    fixed: '~130',   verdict: 'TDTR ~2× faster',       good: true  },
                { metric: 'Drift Detector',     tdtr: 'KSWIN',  fixed: 'None',   verdict: 'TDTR uses detection',   good: false },
              ].map((r) => (
                <tr key={r.metric} className="border-b border-dash-border/50 hover:bg-dash-surface transition-colors">
                  <td className="px-5 py-3 font-semibold text-dash-text">{r.metric}</td>
                  <td className="px-5 py-3 text-dash-accent font-mono">{r.tdtr}</td>
                  <td className="px-5 py-3 text-dash-muted font-mono">{r.fixed}</td>
                  <td className={`px-5 py-3 font-semibold ${r.good ? 'text-dash-green' : 'text-dash-muted'}`}>{r.verdict}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <PageNav />
    </div>
  )
}
