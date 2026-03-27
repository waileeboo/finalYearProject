import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, GitBranch, Lightbulb, FileText, TrendingUp,
} from 'lucide-react'

const nav = [
  { to: '/',             label: 'Overview',            icon: <LayoutDashboard size={15} /> },
  { to: '/architecture', label: 'Architecture',         icon: <GitBranch size={15} /> },
  { type: 'divider' },
  { to: '/rq1', label: 'RQ1 - Static Baselines',   icon: <RQBadge n={1} /> },
  { to: '/rq2', label: 'RQ2 - TDTR Adaptive',      icon: <RQBadge n={2} /> },
  { to: '/rq3', label: 'RQ3 - Ablation Study',     icon: <RQBadge n={3} /> },
  { to: '/rq4', label: 'RQ4 - Conventional ML',    icon: <RQBadge n={4} /> },
  { to: '/rq5', label: 'RQ5 - Limited Training',   icon: <RQBadge n={5} /> },
  { type: 'divider' },
  { to: '/findings', label: 'Key Findings', icon: <Lightbulb size={15} /> },
]

function RQBadge({ n }) {
  return (
    <span className="w-[22px] h-[22px] rounded-md flex items-center justify-center text-[12px] font-black leading-none bg-dash-red/15 text-dash-red">
      {n}
    </span>
  )
}

export default function Sidebar() {
  return (
    <aside className="w-14 shrink-0 bg-dash-surface border-r border-dash-border flex flex-col h-full">
      {/* Logo */}
      <div className="flex items-center justify-center h-14 border-b border-dash-border">
        <div className="w-7 h-7 rounded-lg bg-dash-red flex items-center justify-center">
          <TrendingUp size={14} className="text-white" />
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto py-3 flex flex-col items-center gap-0.5">
        {nav.map((item, i) => {
          if (item.type === 'divider') {
            return <div key={i} className="w-6 border-t border-dash-border my-2" />
          }
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              title={item.label}
              className={({ isActive }) =>
                `w-10 h-10 rounded-xl flex items-center justify-center transition-all
                ${isActive
                  ? 'bg-white/10 text-dash-text'
                  : 'text-dash-muted hover:text-dash-text hover:bg-dash-card'
                }`
              }
            >
              {item.icon}
            </NavLink>
          )
        })}
      </nav>

      {/* PDF link */}
      <div className="flex items-center justify-center py-3 border-t border-dash-border">
        <a
          href="/docs/Final_Year_Project_wlb370.pdf"
          target="_blank"
          rel="noreferrer"
          title="View Full Report (PDF)"
          className="w-10 h-10 rounded-xl flex items-center justify-center text-dash-muted hover:text-dash-accent hover:bg-dash-card transition-all"
        >
          <FileText size={15} />
        </a>
      </div>
    </aside>
  )
}
