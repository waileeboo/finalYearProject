import { Link, useLocation } from 'react-router-dom'
import { ChevronLeft, ChevronRight } from 'lucide-react'

const pages = [
  { to: '/',             label: 'Overview'             },
  { to: '/architecture', label: 'Architecture'         },
  { to: '/rq1',          label: 'RQ1 - Static Baselines'  },
  { to: '/rq2',          label: 'RQ2 - TDTR Adaptive'     },
  { to: '/rq3',          label: 'RQ3 - Ablation Study'    },
  { to: '/rq4',          label: 'RQ4 - Conventional ML'   },
  { to: '/rq5',          label: 'RQ5 - Limited Training'  },
  { to: '/findings',     label: 'Key Findings'            },
]

export default function PageNav() {
  const { pathname } = useLocation()
  const idx = pages.findIndex((p) => p.to === pathname)
  const prev = idx > 0 ? pages[idx - 1] : null
  const next = idx < pages.length - 1 ? pages[idx + 1] : null

  return (
    <div className="flex items-center justify-between mt-10 pt-6 border-t border-dash-border">
      {prev ? (
        <Link
          to={prev.to}
          className="flex items-center gap-2 text-[15px] font-medium text-dash-muted hover:text-dash-accent transition-colors group"
        >
          <div className="w-8 h-8 rounded-lg bg-dash-card border border-dash-border flex items-center justify-center group-hover:border-dash-accent/40 transition-colors">
            <ChevronLeft size={15} />
          </div>
          <div>
            <p className="text-[11px] text-dash-dim uppercase tracking-widest leading-tight">Previous</p>
            <p className="leading-tight">{prev.label}</p>
          </div>
        </Link>
      ) : (
        <div />
      )}

      {next ? (
        <Link
          to={next.to}
          className="flex items-center gap-2 text-[15px] font-medium text-dash-muted hover:text-dash-accent transition-colors group text-right"
        >
          <div className="text-right">
            <p className="text-[11px] text-dash-dim uppercase tracking-widest leading-tight">Next</p>
            <p className="leading-tight">{next.label}</p>
          </div>
          <div className="w-8 h-8 rounded-lg bg-dash-card border border-dash-border flex items-center justify-center group-hover:border-dash-accent/40 transition-colors">
            <ChevronRight size={15} />
          </div>
        </Link>
      ) : (
        <div />
      )}
    </div>
  )
}
