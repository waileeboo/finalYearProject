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
  { model: 'PSO-ELM',  rank: 1.79, priceMAE: '~20.1', returnMAE: '~0.0051', verdict: 'Best static model', good: true },
  { model: 'LSTM',     rank: 2.58, priceMAE: '~33.9', returnMAE: '~0.0060', verdict: 'Strong temporal learner' },
  { model: 'PSO-LSTM', rank: 2.73, priceMAE: '~28.6', returnMAE: '~0.0061', verdict: 'Not significantly better than LSTM' },
  { model: 'ELM',      rank: 2.96, priceMAE: '~20.1', returnMAE: '~0.0063', verdict: 'Analytical solution, strong baseline' },
  { model: 'ARIMA',    rank: 4.94, priceMAE: '~706',  returnMAE: '~0.0078', verdict: 'Significantly worse than all neural models', bad: true },
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
        desc="To what extent does PSO-based optimisation of output layer weights improve the robustness of neural forecasting models (ELM and LSTM) under different types of concept drift - without any drift adaptation?"
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
          <p className="text-[15px] font-semibold text-dash-text mb-4">Return MAE - Friedman / Wilcoxon Rankings</p>
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
          caption="Connected models are not statistically significantly different (Wilcoxon + Bonferroni). PSO-ELM dominates; ARIMA is significantly worse than all neural models."
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
                {['Model', 'Avg. Rank', 'Price MAE (GSPC)', 'Return MAE', 'Verdict'].map((h) => (
                  <th key={h} className="text-left text-[11px] font-bold uppercase tracking-widest text-dash-dim px-5 py-3">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.map((r) => (
                <tr key={r.model} className="border-b border-dash-border/50 hover:bg-dash-surface transition-colors">
                  <td className="px-5 py-3 font-bold font-mono text-dash-text">{r.model}</td>
                  <td className={`px-5 py-3 font-bold ${r.good ? 'text-dash-green' : r.bad ? 'text-dash-red' : 'text-dash-muted'}`}>{r.rank}</td>
                  <td className="px-5 py-3 text-dash-muted">{r.priceMAE}</td>
                  <td className="px-5 py-3 text-dash-muted font-mono">{r.returnMAE}</td>
                  <td className={`px-5 py-3 ${r.good ? 'text-dash-green' : r.bad ? 'text-dash-red' : 'text-dash-muted'}`}>{r.verdict}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Insight cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {[
          { title: 'PSO Benefit for ELM',  color: '#e8e8e8', body: 'PSO-ELM (rank 1.79) significantly outperforms standard ELM (rank 2.96). PSO effectively optimises the output layer weights of the analytical model.' },
          { title: 'PSO Benefit for LSTM', color: '#e8e8e8', body: 'PSO-LSTM (2.73) does not significantly outperform LSTM (2.58). Gradient-based backpropagation already provides effective weight optimisation for LSTM.' },
          { title: 'ARIMA Ceiling',        color: '#e8e8e8', body: 'ARIMA (rank 4.94) is statistically significantly worse than all neural models. Its static linear assumption cannot capture non-stationary distributional shifts.' },
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
