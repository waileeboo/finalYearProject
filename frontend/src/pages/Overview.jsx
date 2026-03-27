import { useState, useEffect } from 'react'
import PageNav from '../components/PageNav'
import { Link } from 'react-router-dom'
import { TrendingDown, Zap, BarChart2, ArrowRight, Activity, X } from 'lucide-react'
import StatCard from '../components/StatCard'
import DiagramCard from '../components/DiagramCard'
import { diagrams, rq2RealPriceMAE } from '../data/results'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'

const kpis = [
  { label: 'LSTM Price MAE Reduction', value: '41%',  sub: 'TDTR vs static baseline on S&P 500', icon: TrendingDown, color: '#e8e8e8' },
  { label: 'PSO-LSTM MAE Reduction',   value: '31%',  sub: 'Real financial data (GSPC)',          icon: TrendingDown, color: '#e8e8e8' },
  { label: 'Smart vs Blind approach', value: '2×',   sub: 'Non-drift models retrain blindly on a schedule. TDTR only updates when drift is detected, halving computational waste.',      icon: Zap,          color: '#e8e8e8' },
  { label: 'Max Gain on Unseen',       value: '70%',  sub: 'PSO-LSTM under limited training',     icon: BarChart2,    color: '#e8e8e8' },
  { label: 'Models Evaluated',         value: '6',    sub: 'LSTM, ELM, PSO variants, RF, SVR',    icon: Activity,     color: '#e8e8e8' },
  { label: 'Drift Scenarios',          value: '120',  sub: '4 types × 30 runs each',              icon: BarChart2,    color: '#e8e8e8' },
]

const rqCards = [
  { rq: 'RQ1', title: 'Static Baselines',    color: '#b05c5c', to: '/rq1', desc: 'PSO-ELM achieves the best static rank (1.79). ARIMA is significantly worse than all neural models.' },
  { rq: 'RQ2', title: 'TDTR Adaptive',       color: '#b05c5c', to: '/rq2', desc: 'TDTR delivers 41% Price MAE reduction for LSTM on real S&P 500 data. ELM shows ceiling effect.' },
  { rq: 'RQ3', title: 'Ablation Study',      color: '#b05c5c', to: '/rq3', desc: 'Fixed-interval retraining matches TDTR accuracy at ~2× the computational cost.' },
  { rq: 'RQ4', title: 'Conventional ML',     color: '#b05c5c', to: '/rq4', desc: 'RF shows marginal gains. SVR shows no benefit - ceiling effect similar to ELM models.' },
  { rq: 'RQ5', title: 'Limited Training',    color: '#b05c5c', to: '/rq5', desc: 'TDTR gains up to 70% on unseen concepts when trained on only the first 20% of the data.' },
]

function PresentationModal({ onClose }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-6"
      onClick={onClose}
    >
      <div
        className="relative max-w-2xl w-full bg-dash-card border border-dash-border rounded-xl shadow-2xl overflow-y-auto max-h-[85vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-dash-border">
          <span className="text-[11px] font-black uppercase tracking-widest text-dash-accent">Presentation Script</span>
          <button onClick={onClose} className="text-dash-dim hover:text-dash-text transition-colors">
            <X size={18} />
          </button>
        </div>
        <div className="px-6 py-5 space-y-4 text-[15px] text-dash-muted leading-relaxed">
          <p>
            Hello. My project investigates drift-adaptive forecasting.
          </p>
          <p>
            In traditional offline training, we assume that the joint probability distribution of the input features X and target variable y at time t is the same at all future times. However, in financial markets, this assumption breaks down - market regimes shift, volatility changes, and patterns that held yesterday no longer hold today.
          </p>
          <p>
            This is called <strong className="text-dash-text">concept drift</strong>. When drift occurs, a
            model trained on historical data becomes stale. It continues to extrapolate from a distribution
            that no longer matches reality, which causes its prediction error to grow over time.
          </p>
          <p>
            Existing approaches try to handle this using ensemble methods, basically running multiple models simultaneously. While effective, they carry massive memory and compute overhead because all models must be kept active simultaneously. 
          </p>
          <p>
            To solve this, I developed <strong className="text-dash-text">TDTR</strong> - Trial-Based Drift-Triggered Retraining. Rather than a heavy ensemble, TDTR maintains just a single active model. It uses a statistical detector to monitor the error stream. When drift is flagged, the system trains a 'challenger' model on recent data and puts it through a short trial period. If the challenger outperforms the active model, it gets promoted. If not, it stays in the pool to compete in future trials. 
          </p>
          <p>
            This gives us the best of both worlds: the system adapts to market shifts, but with bounded memory and minimal overhead. I evaluated TDTR across six model types - LSTM, PSO-LSTM, ELM, PSO-ELM, Random Forest, and SVR, testing them on four synthetic drift benchmarks and real S&amp;P 500 data.
          </p>
          <p>
            The key finding? On real financial data, TDTR delivered a massive 41% reduction in error for LSTM models.
          </p>
        </div>
      </div>
    </div>
  )
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-dash-card border border-dash-border rounded-lg p-3 text-xs">
      <p className="font-semibold text-dash-text mb-2">{label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {p.value.toFixed(1)}
        </p>
      ))}
    </div>
  )
}

export default function Overview() {
  const [scriptOpen, setScriptOpen] = useState(false)
  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {scriptOpen && <PresentationModal onClose={() => setScriptOpen(false)} />}
      {/* Hero */}
      <div
        className="bg-dash-card border border-dash-border rounded-xl p-6 cursor-pointer hover:border-dash-border2 transition-colors"
        style={{ background: 'linear-gradient(135deg, #111111 0%, #1a1d28 100%)' }}
        onClick={() => setScriptOpen(true)}
        title="Click to view presentation script"
      >
        <div className="flex flex-col gap-3 max-w-3xl">
          <span className="inline-flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-widest text-dash-red border border-dash-red/20 bg-dash-red/5 px-3 py-1 rounded-full w-fit">
            Final Year Project · University of Birmingham
          </span>
          <h1 className="text-2xl font-extrabold text-dash-text tracking-tight leading-snug">
            Drift-Resilient Time Series Forecasting
          </h1>
          <p className="text-[15px] text-dash-muted leading-relaxed">
            Financial markets change constantly, but traditional forecasting models don't.{" "}
            This project introduces <strong className="text-dash-text">Trial-Based Drift-Triggered Retraining (TDTR)</strong> - a lightweight
            framework that monitors prediction error using statistical drift detectors and retrains a challenger model
            only when drift is detected, adopting it only if it outperforms the current active model. Evaluated across
            six model types on four synthetic drift benchmarks and real S&amp;P 500 data.
          </p>
          <div className="flex flex-wrap gap-x-4 gap-y-1.5 pt-1 text-[13px] text-dash-dim">
            <span className="text-[12px] text-dash-accent/70 italic">Click to view presentation script</span>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1.5 pt-1 text-[13px] text-dash-dim">
            <span><strong className="text-dash-muted">Author:</strong> Wai Lee Boo (2625170)</span>
            <span>·</span>
            <span><strong className="text-dash-muted">Supervisor:</strong> Professor Leandro L. Minku</span>
            <span>·</span>
            <span><strong className="text-dash-muted">Inspector:</strong> Professor Jens Christian Claussen</span>
            <span>·</span>
            <span><strong className="text-dash-muted">Institution:</strong> University of Birmingham</span>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div>
        <p className="text-[12px] font-bold uppercase tracking-widest text-dash-dim mb-3">Key Metrics</p>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {kpis.map((k) => (
            <StatCard key={k.label} {...k} />
          ))}
        </div>
      </div>

      {/* Chart + Diagram row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Price MAE chart */}
        <div className="bg-dash-card border border-dash-border rounded-xl p-5">
          <p className="text-[12px] font-bold uppercase tracking-widest text-dash-muted mb-1">RQ2 Snapshot</p>
          <p className="text-[15px] font-semibold text-dash-text mb-4">Price MAE: Baseline vs TDTR (S&amp;P 500)</p>
          <ResponsiveContainer width="100%" height={200}>
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

        {/* Synthetic drift */}
        <DiagramCard
          src={diagrams.syntheticDrift}
          title="Synthetic Drift Benchmarks"
          caption="Four drift scenarios × 30 runs: linear/nonlinear × abrupt/gradual. Red dashed lines mark concept boundaries."
        />
      </div>

      {/* RQ cards */}
      <div>
        <p className="text-[12px] font-bold uppercase tracking-widest text-dash-dim mb-3">Research Questions</p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {rqCards.map((c) => (
            <Link
              key={c.rq}
              to={c.to}
              className="bg-dash-card border border-dash-border rounded-xl p-4 hover:border-dash-border2 transition-all group relative overflow-hidden flex flex-col gap-3"
            >
              <div
                className="absolute top-0 left-0 right-0 h-[2px]"
                style={{ background: c.color }}
              />
              <div className="flex items-center justify-between">
                <span
                  className="text-[11px] font-black uppercase tracking-widest px-2 py-0.5 rounded"
                  style={{ background: `${c.color}18`, color: c.color }}
                >
                  {c.rq}
                </span>
                <ArrowRight size={13} className="text-dash-dim group-hover:text-dash-accent transition-colors" />
              </div>
              <div>
                <p className="text-[15px] font-bold text-dash-text mb-1">{c.title}</p>
                <p className="text-[13px] text-dash-muted leading-relaxed">{c.desc}</p>
              </div>
            </Link>
          ))}
        </div>
      </div>
      <PageNav />
    </div>
  )
}
