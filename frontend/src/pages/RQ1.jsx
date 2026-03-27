import PageNav from '../components/PageNav'
import SectionHeader from '../components/SectionHeader'
import DiagramCard from '../components/DiagramCard'
import Callout from '../components/Callout'
import { diagrams, rq1Rankings, rq1ReturnMAE } from '../data/results'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LabelList
} from 'recharts'

const tableRows = [
  { model: 'ELM',      priceMAE: '19.83',  priceColor: '#4ade80', returnMAE: '0.00726', returnColor: '#4ade80', verdict: 'Analytical solution, strong baseline' },
  { model: 'PSO-ELM',  priceMAE: '19.98',  priceColor: '#a3e635', returnMAE: '0.00732', returnColor: '#a3e635', verdict: 'Best average rank across all benchmarks' },
  { model: 'PSO-LSTM', priceMAE: '28.86',  priceColor: '#facc15', returnMAE: '0.00785', returnColor: '#fb923c', verdict: 'Not significantly better than LSTM' },
  { model: 'LSTM',     priceMAE: '34.70',  priceColor: '#fb923c', returnMAE: '0.00945', returnColor: '#f87171', verdict: 'Highest return error among neural models' },
  { model: 'ARIMA',    priceMAE: '522.73', priceColor: '#f87171', returnMAE: '0.00783', returnColor: '#facc15', verdict: 'Static linear model, poor price accuracy' },
]

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-dash-card border border-dash-border rounded-lg p-3 text-xs">
      <p className="font-semibold text-dash-text mb-1">{label}</p>
      <p style={{ color: payload[0].fill }}>Rank: {payload[0].value}</p>
    </div>
  )
}

export default function RQ1() {
  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <SectionHeader
        eyebrow="Research Question 1"
        title="PSO Optimisation vs Static Baselines"
        color="#c9a84c"
        desc="To what extent does PSO-based optimisation of output
        layer weights improve the robustness of neural forecasting
        models (ELM and LSTM) under different types of concept
        drift?"
      />

      <Callout color="#c9a84c">
        <strong className="text-dash-text">Key Finding:</strong> PSO-ELM achieves the best average rank (1.79) on
        Return MAE across all synthetic and real benchmarks. ARIMA (rank 4.94) is significantly worse than all
        neural models. PSO-LSTM (2.73) and ELM (2.96) are statistically indistinguishable.
      </Callout>

      {/* Rank chart + CD diagram */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-dash-card border border-dash-border rounded-xl p-5">
          <p className="text-[12px] font-bold uppercase tracking-widest text-dash-muted mb-1">Average Rank (lower = better)</p>
          <p className="text-[15px] font-semibold text-dash-text mb-4">Return MAE - Friedman Average Ranks</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={rq1Rankings} layout="vertical" barCategoryGap="25%">
              <CartesianGrid strokeDasharray="3 3" stroke="#272727" horizontal={false} />
              <XAxis type="number" domain={[0, 6]} tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="model" tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={false} tickLine={false} width={70} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="rank" radius={[0, 4, 4, 0]}>
                <LabelList dataKey="rank" position="right" style={{ fill: '#8a8a8a', fontSize: 11 }} formatter={(v) => v.toFixed(2)} />
                {rq1Rankings.map((entry) => (
                  <Cell key={entry.model} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <DiagramCard
          src={diagrams.rq1CdReturnMAE}
          title="Critical Difference Diagram - Return MAE"
          caption="Ranks computed over synthetic datasets only (GSPC excluded). Connected models are not statistically significantly different (Nemenyi post-hoc test, p > 0.05)."
          lightBg
        />
      </div>

      {/* ARIMA diagram */}
      <DiagramCard
        src={diagrams.arimaBaseline}
        title="ARIMA Baseline - S&P 500 Forecast vs Actual"
        caption="ARIMA produces a smooth trend extrapolation (red dashed) that diverges from actual prices (blue). Its static linear structure makes it fundamentally ill-suited to non-stationary financial data with concept drift."
        lightBg
      />

      {/* Results table */}
      <div className="bg-dash-card border border-dash-border rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-dash-border">
          <p className="text-[15px] font-bold text-dash-text">Model Performance Summary - RQ1</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-dash-border">
                {['Model', 'Price MAE (GSPC)', 'Return MAE', 'Verdict'].map((h) => (
                  <th key={h} className="text-left text-[11px] font-bold uppercase tracking-widest text-dash-dim px-5 py-3">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.map((r) => (
                <tr key={r.model} className="border-b border-dash-border/50 hover:bg-dash-surface transition-colors">
                  <td className="px-5 py-3 font-bold font-mono text-dash-text">{r.model}</td>
                  <td className="px-5 py-3 font-mono font-bold" style={{ color: r.priceColor }}>{r.priceMAE}</td>
                  <td className="px-5 py-3 font-mono text-dash-muted">{r.returnMAE}</td>
                  <td className="px-5 py-3 text-dash-muted">{r.verdict}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Insight cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {[
          { title: 'PSO is not universally beneficial',          color: '#e8e8e8', body: 'PSO does not consistently improve performance across all architectures and drift types. Standard models remain competitive and sometimes superior under simpler linear drift.' },
          { title: 'PSO-ELM is the standout performer',         color: '#e8e8e8', body: 'PSO-ELM achieves the best average rank (1.79) across all synthetic datasets and is most beneficial under non-linear drift where the underlying structure is more complex.' },
          { title: 'PSO helps LSTM on real data but not synthetic', color: '#e8e8e8', body: 'On S&P 500, PSO-LSTM reduces Price MAE from 34.70 to 28.86, but on synthetic data the improvement over standard LSTM is not statistically significant.' },
          { title: 'PSO does not help ELM on real data',        color: '#e8e8e8', body: 'The static ELM already generalises so well that PSO adds no meaningful benefit.' },
          { title: 'The ceiling effect appears early',          color: '#e8e8e8', body: 'ELM and PSO-ELM converge to very similar performance on real data (~19.83 vs 19.98), suggesting an absolute error floor exists regardless of optimisation strategy.' },
          { title: 'Return MAE is misleading on real data',     color: '#e8e8e8', body: 'Models like ARIMA appear competitive under Return MAE simply by predicting near-zero returns, making Price MAE the more informative metric.' },
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
