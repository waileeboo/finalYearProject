export default function StatCard({ label, value, sub, icon: Icon, color = '#e8e8e8', trend }) {
  return (
    <div className="bg-dash-card border border-dash-border rounded-xl p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <p className="text-[12px] font-semibold uppercase tracking-widest text-dash-muted">{label}</p>
        {Icon && (
          <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: `${color}18` }}>
            <Icon size={14} style={{ color }} />
          </div>
        )}
      </div>
      <div>
        <p className="text-2xl font-extrabold leading-none" style={{ color }}>{value}</p>
        {sub && <p className="text-[12px] text-dash-dim mt-1 leading-tight">{sub}</p>}
      </div>
      {trend && (
        <p className="text-[12px] text-dash-muted border-t border-dash-border pt-2">{trend}</p>
      )}
    </div>
  )
}
