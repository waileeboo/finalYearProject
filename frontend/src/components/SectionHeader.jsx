export default function SectionHeader({ eyebrow, title, desc, color = '#c9a84c' }) {
  return (
    <div className="mb-6">
      {eyebrow && (
        <p className="text-[12px] font-bold uppercase tracking-widest mb-1 text-dash-text">
          {eyebrow}
        </p>
      )}
      <h2 className="text-xl font-extrabold text-dash-text mb-2 tracking-tight">{title}</h2>
      {desc && <p className="text-[15px] text-dash-muted leading-relaxed max-w-2xl">{desc}</p>}
    </div>
  )
}
