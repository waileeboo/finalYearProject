import { useLocation } from 'react-router-dom'
import { GraduationCap } from 'lucide-react'

const titles = {
  '/':             { label: 'Overview',              sub: 'Project summary and key metrics' },
  '/architecture': { label: 'System Architecture',   sub: 'TDTR framework design and flow' },
  '/rq1':          { label: 'RQ1 - Static Baselines',sub: 'PSO optimisation vs baselines without drift adaptation' },
  '/rq2':          { label: 'RQ2 - TDTR Adaptive',   sub: 'TDTR vs non-adaptive baselines on synthetic and real data' },
  '/rq3':          { label: 'RQ3 - Ablation Study',  sub: 'Continuous retraining vs drift-triggered retraining' },
  '/rq4':          { label: 'RQ4 - Conventional ML', sub: 'Random Forest and SVR under TDTR adaptation' },
  '/rq5':          { label: 'RQ5 - Limited Training', sub: 'TDTR with 20% training coverage on unseen concepts' },
  '/findings':     { label: 'Key Findings',           sub: 'Consolidated answers to all research questions' },
}

export default function Header() {
  const { pathname } = useLocation()
  const page = titles[pathname] ?? { label: 'Dashboard', sub: '' }

  return (
    <header className="h-14 shrink-0 bg-dash-surface border-b border-dash-border flex items-center px-6 relative">
      {/* Centred title */}
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <h1 className="text-sm font-bold text-dash-text leading-tight">{page.label}</h1>
        {page.sub && <p className="text-[12px] text-dash-dim leading-tight">{page.sub}</p>}
      </div>
      {/* Right side */}
      <div className="ml-auto flex items-center gap-2 text-dash-dim">
        <GraduationCap size={14} />
        <span className="text-[12px]">University of Birmingham · 2025</span>
      </div>
    </header>
  )
}
