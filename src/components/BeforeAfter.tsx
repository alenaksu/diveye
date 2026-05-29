import { useCallback, useEffect, useRef, useState } from 'react'

interface Props {
  before: ImageBitmap
  after: ImageBitmap
  width: number
  height: number
}

export function BeforeAfter({ before, after, width, height }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [sliderX, setSliderX] = useState(0.5) // 0-1 fraction
  const dragging = useRef(false)

  // Measure container so we can fit both axes
  const [containerSize, setContainerSize] = useState({ w: 0, h: 0 })
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => {
      setContainerSize({ w: entry.contentRect.width, h: entry.contentRect.height })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Fit within container while preserving aspect ratio; never upscale beyond original
  const cw = containerSize.w || width
  const ch = containerSize.h || height
  const scale = Math.min(cw / width, ch / height, 1)
  const displayW = Math.max(1, Math.round(width  * scale))
  const displayH = Math.max(1, Math.round(height * scale))

  // Physical pixel multiplier for HiDPI / Retina screens
  const dpr = typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!

    // Work in logical (CSS) pixels; dpr scaling is applied via ctx.scale
    const w = displayW
    const h = displayH
    const splitX = Math.round(sliderX * w)

    // Reset transform before clearing so we always wipe the full physical buffer
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Scale all drawing to physical pixels for crisp HiDPI output
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(before, 0, 0, w, h)
    ctx.save()
    ctx.beginPath()
    ctx.rect(splitX, 0, w - splitX, h)
    ctx.clip()
    ctx.drawImage(after, 0, 0, w, h)
    ctx.restore()

    // Divider line
    ctx.beginPath()
    ctx.moveTo(splitX, 0)
    ctx.lineTo(splitX, h)
    ctx.strokeStyle = 'rgba(255,255,255,0.85)'
    ctx.lineWidth = 2
    ctx.stroke()

    // Handle circle
    const cy = h / 2
    ctx.beginPath()
    ctx.arc(splitX, cy, 18, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(255,255,255,0.95)'
    ctx.fill()
    ctx.strokeStyle = 'rgba(0,0,0,0.2)'
    ctx.lineWidth = 1
    ctx.stroke()

    // Arrows
    ctx.fillStyle = '#0b4a6a'
    const drawArrow = (x: number, dir: -1 | 1) => {
      ctx.beginPath()
      ctx.moveTo(x + dir * 5, cy)
      ctx.lineTo(x + dir * 9, cy - 4)
      ctx.lineTo(x + dir * 9, cy + 4)
      ctx.closePath()
      ctx.fill()
    }
    drawArrow(splitX - 4, -1)
    drawArrow(splitX + 4, 1)

    // Labels
    ctx.font = 'bold 11px system-ui'
    ctx.fillStyle = 'rgba(255,255,255,0.7)'
    ctx.textBaseline = 'top'
    if (splitX > 50) {
      ctx.textAlign = 'left'
      ctx.fillText('BEFORE', 10, 10)
    }
    if (splitX < w - 50) {
      ctx.textAlign = 'right'
      ctx.fillText('AFTER', w - 10, 10)
    }
  }, [before, after, sliderX, displayW, displayH, dpr])

  useEffect(() => {
    draw()
  }, [draw])

  const getX = (e: React.MouseEvent | React.TouchEvent | MouseEvent | TouchEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const clientX = 'touches' in e ? e.touches[0].clientX : (e as MouseEvent).clientX
    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
  }

  const onPointerDown = (e: React.MouseEvent | React.TouchEvent) => {
    dragging.current = true
    setSliderX(getX(e))
  }

  useEffect(() => {
    const onMove = (e: MouseEvent | TouchEvent) => {
      if (!dragging.current) return
      setSliderX(getX(e))
    }
    const onUp = () => { dragging.current = false }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('touchmove', onMove, { passive: true })
    window.addEventListener('mouseup', onUp)
    window.addEventListener('touchend', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('touchmove', onMove)
      window.removeEventListener('mouseup', onUp)
      window.removeEventListener('touchend', onUp)
    }
  }, [])

  return (
    <div ref={containerRef} className="w-full h-full flex items-center justify-center">
      <canvas
        ref={canvasRef}
        width={displayW * dpr}
        height={displayH * dpr}
        style={{ width: displayW, height: displayH }}
        className="rounded-2xl shadow-2xl cursor-col-resize touch-none select-none max-w-full max-h-full"
        onMouseDown={onPointerDown}
        onTouchStart={onPointerDown}
      />
    </div>
  )
}
