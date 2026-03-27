import PageNav from '../components/PageNav'
import DiagramCard from '../components/DiagramCard'
import SectionHeader from '../components/SectionHeader'

const generalFindings = [
  {
    color: '#e8e8e8',
    title: 'TDTR: When to Use It and When Not To',
    result: 'Use for: gradient-based models or models facing unseen or abrupt concept shifts',
    body: 'TDTR is most valuable when the data stream contains genuinely unseen concept configurations — where a static model has never seen the incoming regime during training. Under these conditions, gradient-based models (LSTM, PSO-LSTM) benefit the most, as stale weights accumulate bias toward the old concept and retraining actively resets them. TDTR is also effective under abrupt drift, where the shift is sudden enough for the detector to fire cleanly. It is less effective when drift is gradual, as the static model partially adapts through its training window overlap.',
    tags: ['Use with LSTM', 'Strongest on unseen concepts', 'Effective under abrupt drift', 'Limited gain under gradual drift', 'No benefit for ELM/SVR/RF on recurring data'],
  },
]

const qa = [
  {
    q: 'Why does LSTM improve on recurring concepts under TDTR, while ELM does not?',
    a: 'LSTM maintains a recursive hidden state updated at every time step. Because its gating mechanisms were optimised for the original training distribution, they fail to discard obsolete patterns as the regime changes, causing the hidden state to carry stale representations even before retraining occurs. TDTR detects this shift and retrains a challenger on a fresh window from the new regime, effectively resetting the stale hidden state and realigning the weights. ELM, by contrast, solves its output weights analytically in a single closed-form step, producing an equally generalised solution regardless of which concept is active, as there is no accumulated state bias to reset. This is why TDTR yields a 41% Price_MAE reduction for LSTM on the S&P 500 but near-zero improvement for ELM.',
    image: '/diagrams/lstm_memory.png',
    imageCaption: 'LSTM memory diagram, illustrating how hidden state and cell state carry concept-specific gradient bias across time steps.',
  },
]

export default function Findings() {
  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      <SectionHeader
        eyebrow="Summary"
        title="Key Findings"
        color="#5a9e7c"
        desc="Consolidated answers to all five research questions, with practical guidance on when and where TDTR provides value."
      />

      {/* Summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { value: '41%',   label: 'Peak Price MAE Reduction' },
          { value: '70%',   label: 'Peak Gain on Unseen Data' },
          { value: '2×',    label: 'Efficiency vs Fixed Retrain' },
          { value: 'KSWIN', label: 'Best Drift Detector' },
        ].map((s) => (
          <div key={s.label} className="bg-dash-card border border-dash-border rounded-xl p-4 text-center">
            <p className="text-2xl font-extrabold mb-1 text-dash-text">{s.value}</p>
            <p className="text-[13px] text-dash-muted">{s.label}</p>
          </div>
        ))}
      </div>

      {/* General finding card */}
      {generalFindings.map((f) => (
        <div
          key={f.title}
          className="bg-dash-card border border-dash-border rounded-xl overflow-hidden"
          style={{ borderTopColor: f.color, borderTopWidth: 2 }}
        >
          <div className="p-5">
            <div className="flex items-center gap-2.5 mb-3">
              <span
                className="text-[11px] font-black uppercase tracking-widest px-2 py-0.5 rounded"
                style={{ background: `${f.color}18`, color: f.color }}
              >
                General
              </span>
              <h3 className="text-[15px] font-bold text-dash-text leading-snug">{f.title}</h3>
            </div>
            <p
              className="text-[13px] font-mono font-semibold mb-3 px-2.5 py-1.5 rounded"
              style={{ background: `${f.color}0d`, color: f.color }}
            >
              {f.result}
            </p>
            <p className="text-[14px] text-dash-muted leading-relaxed mb-3">{f.body}</p>
            <div className="flex flex-wrap gap-1.5">
              {f.tags.map((t) => (
                <span
                  key={t}
                  className="text-[12px] font-semibold px-2 py-0.5 rounded border"
                  style={{ color: f.color, borderColor: `${f.color}30`, background: `${f.color}0a` }}
                >
                  {t}
                </span>
              ))}
            </div>
          </div>
        </div>
      ))}

      {/* Q&A */}
      <div className="space-y-4">
        <h2 className="text-[17px] font-bold text-dash-text px-1">Discussion</h2>
        {qa.map(({ q, a, image, imageCaption }, i) => (
          <div key={i} className="bg-dash-card border border-dash-border rounded-xl overflow-hidden">
            <div className="flex items-start gap-3 px-5 py-5 border-b border-dash-border bg-dash-card">
              <span className="mt-0.5 w-6 h-6 rounded-full bg-white/10 text-dash-text text-[12px] font-black flex items-center justify-center shrink-0">
                Q
              </span>
              <p className="text-[17px] font-semibold text-dash-text leading-snug">{q}</p>
            </div>
            <div className="flex items-start gap-3 px-5 py-5">
              <span className="mt-0.5 w-6 h-6 rounded-full bg-white/10 text-dash-text text-[12px] font-black flex items-center justify-center shrink-0">
                A
              </span>
              <div className="flex-1 space-y-4">
                <p className="text-[16px] text-dash-muted leading-relaxed">{a}</p>
                {image && (
                  <div className="max-w-lg mx-auto">
                    <DiagramCard src={image} caption={imageCaption} lightBg />
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Technologies & Methods */}
      <div className="bg-dash-card border border-dash-border rounded-xl p-5">
        <p className="text-[13px] font-bold uppercase tracking-widest text-dash-dim mb-4">Technologies & Methods</p>
        <div className="flex flex-wrap gap-2">
          {[
            'PyTorch (LSTM)', 'NumPy (ELM)', 'Particle Swarm Optimisation',
            'KSWIN', 'ADWIN', 'Page-Hinkley', 'River (Drift Detection)',
            'scikit-learn (RF / SVR)', 'yfinance (S&P 500)', 'Optuna (Hyperparameter Tuning)',
            'Critical Difference Diagrams',
          ].map((t) => (
            <span key={t} className="text-[13px] font-medium px-3 py-1 rounded bg-dash-surface border border-dash-border text-dash-muted">
              {t}
            </span>
          ))}
        </div>
      </div>

      <PageNav />
    </div>
  )
}
