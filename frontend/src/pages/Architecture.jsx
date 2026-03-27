import PageNav from '../components/PageNav'
import DiagramCard from '../components/DiagramCard'
import SectionHeader from '../components/SectionHeader'
import { diagrams } from '../data/results'

const steps = [
  { label: 'A', title: 'Offline Pre-Training',    color: '#dc2626',
    desc: 'One of ELM, PSO-ELM, LSTM, or PSO-LSTM is trained on historical data and set as the initial active model.' },
  { label: 'B', title: 'Online Adaptive Loop',    color: '#dc2626',
    desc: 'A drift detector continuously monitors the prediction error stream. On detection, the active model is cloned and scheduled for retraining.' },
  { label: 'C', title: 'Trial Phase (20 steps)',  color: '#dc2626',
    desc: 'The retrained challenger runs silently alongside the active model for exactly 20 steps. Rolling MAE is tracked for both.' },
  { label: 'D', title: 'Promotion Decision',      color: '#dc2626',
    desc: 'The model with lower trial MAE is promoted to active. The displaced model is retained in the challenger pool with its error history cleared.' },
  { label: 'E', title: 'Pool Management',         color: '#dc2626',
    desc: 'Challenger pool is capped at 2 models. When full, the worst-performing challenger is evicted to make room for newly retrained candidates.' },
  { label: 'F', title: 'Cooldown Period',         color: '#dc2626',
    desc: 'A suppression window follows each trial to prevent cascading false alarms from repeated detection of the same drift event.' },
]

const detectors = [
  { name: 'KSWIN',         desc: 'Kolmogorov-Smirnov Windowing test. Non-parametric two-sample test comparing recent vs reference windows. Selected as primary detector for Phase 2.', selected: true },
  { name: 'ADWIN',         desc: 'Adaptive Windowing. Maintains a sliding window and detects drift by testing sub-windows for distributional equality.', selected: false },
  { name: 'Page-Hinkley',  desc: 'Sequential change-point detection based on cumulative sums of error deviations from a running mean.', selected: false },
]

export default function Architecture() {
  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <SectionHeader
        eyebrow="Framework Design"
        title="TDTR System Architecture"
        color="#9d7bb5"
        desc="The TDTR framework consists of three main phases: offline pre-training, an online adaptive loop, and a challenger pool with trial-based evaluation. Unlike heavy ensemble methods, only a single active model makes live predictions at any given time."
      />

      {/* System diagram */}
      <DiagramCard
        src={diagrams.systemOverview}
        title="Full System Flowchart"
        caption="Block A: Offline pre-training. Block B: Online adaptive loop with drift detection. Block C: Challenger pool with trial-based promotion logic."
        lightBg
      />

      {/* Steps */}
      <div>
        <p className="text-[12px] font-bold uppercase tracking-widest text-dash-dim mb-3">Framework Steps</p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {steps.map((s) => (
            <div key={s.label} className="bg-dash-card border border-dash-border rounded-xl p-4 flex gap-3">
              <div
                className="w-7 h-7 rounded-lg flex items-center justify-center text-xs font-black text-white shrink-0 mt-0.5"
                style={{ background: s.color }}
              >
                {s.label}
              </div>
              <div>
                <p className="text-[15px] font-bold text-dash-text mb-1">{s.title}</p>
                <p className="text-[13px] text-dash-muted leading-relaxed">{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Drift detectors */}
      <div>
        <p className="text-[12px] font-bold uppercase tracking-widest text-dash-dim mb-3">Drift Detectors Compared</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {detectors.map((d) => (
            <div
              key={d.name}
              className={`bg-dash-card rounded-xl p-4 border ${d.selected ? 'border-red-600' : 'border-dash-border'}`}
            >
              <div className="flex items-center justify-between mb-2">
                <p className="text-[15px] font-bold text-dash-text font-mono">{d.name}</p>
                {d.selected && (
                  <span className="text-[11px] font-bold bg-white/10 text-dash-text px-2 py-0.5 rounded">Selected</span>
                )}
              </div>
              <p className="text-[13px] text-dash-muted leading-relaxed">{d.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Pool activity + key properties */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <DiagramCard
          src={diagrams.poolActivity}
          title="Challenger Pool - Model Version Lifecycle (ELM KSWIN)"
          caption="Each row is a model version. Bars show the steps during which that version was active in the pool."
        />
        <div className="space-y-3">
          <p className="text-[12px] font-bold uppercase tracking-widest text-dash-dim">Key Design Properties</p>
          {[
            { label: 'Single Active Model',   desc: 'Only one model makes predictions at any time. No ensemble overhead.' },
            { label: 'Bounded Memory',        desc: 'Challenger pool capped at 2 models. When a new challenger arrives and the pool is full, the weakest current challenger is replaced, but any displaced active model is always retained for potential re-use if its concept recurs.' },
            { label: 'Trial Duration',        desc: '20-step fixed trial window balances responsiveness and evaluation stability.' },
            { label: 'Cooldown Suppression',  desc: 'Post-trial cooldown prevents false-alarm cascades from the same drift event.' },
            { label: 'Pool Retention',        desc: 'Displaced models are retained with cleared history, enabling recovery if prior regime recurs.' },
          ].map((p) => (
            <div key={p.label} className="bg-dash-card border border-dash-border rounded-xl p-3.5">
              <p className="text-[13px] font-bold text-dash-text mb-0.5">{p.label}</p>
              <p className="text-[13px] text-dash-muted">{p.desc}</p>
            </div>
          ))}
        </div>
      </div>
      <PageNav />
    </div>
  )
}
