import { useState, useEffect } from 'react'
import { X, ZoomIn } from 'lucide-react'

function Lightbox({ src, title, onClose }) {
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
        className="relative max-w-5xl w-full max-h-full flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute -top-10 right-0 text-white/70 hover:text-white transition-colors"
        >
          <X size={22} />
        </button>
        <img
          src={src}
          alt={title}
          className="w-full h-auto max-h-[85vh] object-contain rounded-xl"
        />
        {title && (
          <p className="text-center text-white/60 text-[13px] mt-3">{title}</p>
        )}
      </div>
    </div>
  )
}

export default function DiagramCard({ src, title, caption, lightBg = false, className = '' }) {
  const [open, setOpen] = useState(false)

  return (
    <>
      <div className={`bg-dash-card border border-dash-border rounded-xl overflow-hidden flex flex-col ${className}`}>
        <div
          className={`relative group cursor-zoom-in ${lightBg ? 'bg-white p-3' : 'bg-dash-bg'}`}
          onClick={() => setOpen(true)}
        >
          <img src={src} alt={title} className="w-full h-auto block" />
          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-all flex items-center justify-center">
            <ZoomIn size={28} className="text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-lg" />
          </div>
        </div>
        {(title || caption) && (
          <div className="p-4 border-t border-dash-border">
            {title   && <p className="text-[15px] font-semibold text-dash-text mb-1">{title}</p>}
            {caption && <p className="text-[13px] text-dash-muted leading-relaxed">{caption}</p>}
          </div>
        )}
      </div>

      {open && <Lightbox src={src} title={title} onClose={() => setOpen(false)} />}
    </>
  )
}
