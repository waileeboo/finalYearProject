export default function Callout({ children, color = '#c9a84c' }) {
  return (
    <div
      className="rounded-r-xl p-4 mb-6 text-[15px] text-dash-muted leading-relaxed bg-dash-card"
      style={{
        borderLeft: '3px solid #e8e8e8',
        borderTop: '1px solid #272727',
        borderRight: '1px solid #272727',
        borderBottom: '1px solid #272727',
      }}
    >
      {children}
    </div>
  )
}
