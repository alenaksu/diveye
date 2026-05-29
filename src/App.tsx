import { useCallback, useEffect, useRef, useState } from 'react'
import { useInference } from './hooks/useInference'
import { ImageUpload } from './components/ImageUpload'
import { BeforeAfter } from './components/BeforeAfter'
import { EP_LABELS, type ExecutionProvider } from './lib/backend-detect'
import type { DehazeMethod, DcpOptions, ClaheOptions, DehazeOptions } from './lib/dehaze'

interface Result {
  beforeBitmap: ImageBitmap
  afterBitmap: ImageBitmap
  width: number
  height: number
  filename: string
}

export default function App() {
  const [preferredEP, setPreferredEP] = useState<ExecutionProvider>('wasm')
  const { status, ep, enhance, dehaze } = useInference(preferredEP)

  // If the worker fell back to a different backend, sync the selector
  useEffect(() => {
    if (ep && ep !== preferredEP) setPreferredEP(ep)
  }, [ep])
  const [result, setResult]             = useState<Result | null>(null)
  const [enhanceError, setEnhanceError] = useState<string | null>(null)
  const [showDropzone, setShowDropzone] = useState(true)

  // Raw LU2Net output — kept so dehaze re-runs don't need a full model pass
  const rawOutputRef = useRef<ImageData | null>(null)

  // Dehaze settings
  const [dehazeEnabled, setDehazeEnabled] = useState(false)
  const [dehazeMethod, setDehazeMethod]   = useState<DehazeMethod>('dcp')
  const [dcpOpts, setDcpOpts]     = useState<DcpOptions>({ omega: 0.75, patchSize: 15 })
  const [claheOpts, setClaheOpts] = useState<ClaheOptions>({ clipLimit: 2.0, tiles: 8 })

  // JPEG quality (60–100)
  const [jpegQuality, setJpegQuality] = useState(92)

  // Tracks dehaze re-apply (separate from worker processing phase)
  const [isDehazing, setIsDehazing] = useState(false)

  // Mobile bottom sheet
  const [sheetOpen, setSheetOpen] = useState(false)

  // -------------------------------------------------------------------------
  // Helper: take a raw ImageData, optionally dehaze it, then push to result
  // -------------------------------------------------------------------------
  const applyAndDisplay = useCallback(
    async (
      raw: ImageData,
      beforeBitmap: ImageBitmap,
      filename: string,
      doDehazeEnabled: boolean,
      doDehazeOptions: DehazeOptions,
    ) => {
      const cloned = new ImageData(
        new Uint8ClampedArray(raw.data),
        raw.width,
        raw.height,
      )

      const display = doDehazeEnabled
        ? await dehaze(cloned, doDehazeOptions)
        : cloned

      const afterCanvas = document.createElement('canvas')
      afterCanvas.width  = display.width
      afterCanvas.height = display.height
      const afterCtx = afterCanvas.getContext('2d')!
      afterCtx.putImageData(display, 0, 0)
      const afterBitmap = await createImageBitmap(afterCanvas)
      afterCanvas.width = 0
      afterCanvas.height = 0

      setResult((prev) => {
        prev?.afterBitmap.close()
        return { beforeBitmap, afterBitmap, width: beforeBitmap.width, height: beforeBitmap.height, filename }
      })
    },
    [dehaze],
  )

  // -------------------------------------------------------------------------
  // File drop → run full pipeline
  // -------------------------------------------------------------------------
  const handleFile = useCallback(
    async (file: File) => {
      if (status.phase !== 'ready') return

      setEnhanceError(null)
      setShowDropzone(false)

      const bitmap = await createImageBitmap(file)
      const canvas = document.createElement('canvas')
      canvas.width  = bitmap.width
      canvas.height = bitmap.height
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(bitmap, 0, 0)
      const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height)
      canvas.width = 0
      canvas.height = 0

      try {
        const outputData = await enhance(imageData)

        rawOutputRef.current = new ImageData(
          new Uint8ClampedArray(outputData.data),
          outputData.width,
          outputData.height,
        )

        const filename = file.name.replace(/\.[^.]+$/, '') + '_enhanced.jpg'
        setResult((prev) => { prev?.beforeBitmap.close(); return prev })
        const dehazeOptions: DehazeOptions = dehazeMethod === 'dcp'
          ? { method: 'dcp', dcp: dcpOpts }
          : { method: 'clahe', clahe: claheOpts }
        await applyAndDisplay(outputData, bitmap, filename, dehazeEnabled, dehazeOptions)
      } catch (err) {
        console.error('Enhance failed:', err)
        setEnhanceError(err instanceof Error ? err.message : String(err))
        bitmap.close()
      }
    },
    [status.phase, enhance, applyAndDisplay, dehazeEnabled, dehazeMethod, dcpOpts, claheOpts],
  )

  // -------------------------------------------------------------------------
  // Upload new — clear everything and show dropzone
  // -------------------------------------------------------------------------
  const handleUploadNew = useCallback(() => {
    result?.beforeBitmap.close()
    result?.afterBitmap.close()
    rawOutputRef.current = null
    setResult(null)
    setEnhanceError(null)
    setShowDropzone(true)
  }, [result])

  // -------------------------------------------------------------------------
  // Re-apply whenever dehaze settings change (if we have raw output)
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (!rawOutputRef.current || !result) return
    if (status.phase !== 'ready') return

    const raw = rawOutputRef.current
    const { beforeBitmap, filename } = result

    applyAndDisplay(raw, beforeBitmap, filename, dehazeEnabled, dehazeMethod === 'dcp'
      ? { method: 'dcp', dcp: dcpOpts }
      : { method: 'clahe', clahe: claheOpts },
    ).catch(
      (err) => console.error('Dehaze re-apply failed:', err),
    ).finally(() => setIsDehazing(false))

    setIsDehazing(true)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dehazeEnabled, dehazeMethod, dcpOpts, claheOpts])

  // -------------------------------------------------------------------------
  // Download as JPEG
  // -------------------------------------------------------------------------
  const handleDownload = useCallback(() => {
    if (!result) return
    const canvas = document.createElement('canvas')
    canvas.width  = result.afterBitmap.width
    canvas.height = result.afterBitmap.height
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(result.afterBitmap, 0, 0)
    canvas.toBlob(
      (blob) => {
        if (!blob) return
        const url = URL.createObjectURL(blob)
        const a   = document.createElement('a')
        a.href     = url
        a.download = result.filename
        a.click()
        URL.revokeObjectURL(url)
      },
      'image/jpeg',
      jpegQuality / 100,
    )
  }, [result, jpegQuality])

  const isReady      = status.phase === 'ready'
  const isProcessing = status.phase === 'processing'
  const isLoading    = status.phase === 'loading'
  const isError      = status.phase === 'error'
  const hasResult    = !!result
  const isBusy       = isProcessing || isDehazing

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-gradient-to-b from-ocean-950 to-slate-950 text-white">

      {/* Header */}
      <header className="shrink-0 border-b border-white/5 bg-black/20 backdrop-blur-sm">
        <div className="mx-auto flex max-w-screen-xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🤿</span>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">AquaTune</h1>
              <p className="text-xs text-white/40">Underwater image enhancement</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {/* Backend selector — full pills on desktop, active-only chip on mobile */}
            <div className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 p-0.5">
              {(['wasm', 'webgl', 'webgpu'] as ExecutionProvider[]).map((backend) => {
                const active = preferredEP === backend
                const isLoaded = ep === backend
                const activeColors: Record<ExecutionProvider, string> = {
                  webgpu: 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40',
                  webgl:  'bg-blue-500/20 text-blue-300 border border-blue-500/40',
                  wasm:   'bg-amber-500/20 text-amber-300 border border-amber-500/40',
                }
                // On mobile hide inactive pills
                const mobileVisibility = active ? '' : 'hidden sm:flex'
                return (
                  <button
                    key={backend}
                    onClick={() => {
                      if (preferredEP !== backend) {
                        setPreferredEP(backend)
                        handleUploadNew()
                      }
                    }}
                    title={backend === 'webgl' ? 'WebGL — may fail (Erf op unsupported)' : backend === 'webgpu' ? 'WebGPU — may produce incorrect results' : 'WASM — recommended'}
                    className={[
                      'relative flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-all',
                      mobileVisibility,
                      active
                        ? activeColors[backend]
                        : 'border border-transparent text-white/40 hover:text-white/70',
                    ].join(' ')}
                  >
                    {isLoaded && (
                      <span className="size-1.5 rounded-full bg-current opacity-70" />
                    )}
                    {EP_LABELS[backend]}
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      </header>

      {/* Model loading / error banner */}
      {(isLoading || isError) && (
        <div className="shrink-0 border-b border-white/5 bg-black/10 px-6 py-3">
          <div className="mx-auto max-w-screen-xl">
            {isLoading && (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-white/50">Loading model…</span>
                  <span className="text-white/30">
                    {status.total > 0 ? `${Math.round((status.loaded / status.total) * 100)}%` : 'connecting…'}
                  </span>
                </div>
                <div className="h-0.5 w-full overflow-hidden rounded-full bg-white/10">
                  <div
                    className="h-full rounded-full bg-ocean-400 transition-all duration-300"
                    style={{
                      width: status.total > 0 ? `${(status.loaded / status.total) * 100}%` : '100%',
                      animation: status.total === 0 ? 'pulse 1.5s ease-in-out infinite' : 'none',
                    }}
                  />
                </div>
              </div>
            )}
            {isError && <p className="text-xs text-red-400">Error: {status.message}</p>}
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* DROPZONE VIEW                                                        */}
      {/* ------------------------------------------------------------------ */}
      {showDropzone && (
        <div className="flex flex-1 items-center justify-center px-6 py-16">
          <div className="w-full max-w-xl">
            <ImageUpload onFile={handleFile} disabled={!isReady} />
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* RESULT / PROCESSING VIEW                                            */}
      {/* ------------------------------------------------------------------ */}
      {!showDropzone && (
        <div className="flex flex-1 overflow-hidden">

          {/* ── Desktop sidebar (hidden on mobile) ─────────────────────── */}
          <aside className="hidden sm:flex w-56 shrink-0 flex-col gap-6 border-r border-white/5 bg-black/20 px-4 py-6">

            {/* Upload new */}
            <button
              onClick={handleUploadNew}
              disabled={isBusy}
              className="flex items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white/80 transition hover:bg-white/10 hover:border-white/20 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <svg className="size-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
              Upload new image
            </button>

            {/* Filename */}
            {result && (
              <p className="truncate text-xs text-white/30" title={result.filename}>
                {result.filename}
              </p>
            )}

            <div className="h-px bg-white/5" />

            {/* Dehaze */}
            <div className="space-y-3">
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <button
                  role="switch"
                  aria-checked={dehazeEnabled}
                  disabled={!hasResult || isBusy}
                  onClick={() => setDehazeEnabled((v) => !v)}
                  className={[
                    'relative inline-flex h-5 w-9 shrink-0 rounded-full border border-white/20 transition-colors focus:outline-none',
                    dehazeEnabled ? 'bg-ocean-500' : 'bg-white/10',
                    !hasResult || isBusy ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer',
                  ].join(' ')}
                >
                  <span className={[
                    'pointer-events-none inline-block h-4 w-4 translate-y-[1px] rounded-full bg-white shadow transition-transform',
                    dehazeEnabled ? 'translate-x-[17px]' : 'translate-x-[1px]',
                  ].join(' ')} />
                </button>
                <span className="text-sm text-white/70">Dehaze</span>
              </label>

              <div className={`space-y-4 pl-1 transition-opacity ${dehazeEnabled && hasResult && !isBusy ? 'opacity-100' : 'opacity-30 pointer-events-none'}`}>
                {/* Method selector */}
                <div className="space-y-2">
                  {(['dcp', 'clahe'] as DehazeMethod[]).map((m) => (
                    <label key={m} className="flex items-center gap-2 cursor-pointer select-none">
                      <input
                        type="radio"
                        name="dehaze-method"
                        value={m}
                        checked={dehazeMethod === m}
                        onChange={() => setDehazeMethod(m)}
                        className="accent-ocean-400"
                      />
                      <span className="text-xs text-white/60">{m.toUpperCase()}</span>
                    </label>
                  ))}
                </div>

                {/* DCP options */}
                {dehazeMethod === 'dcp' && (
                  <div className="space-y-3">
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-white/40">Strength</span>
                        <span className="text-xs text-white/60 tabular-nums">{dcpOpts.omega.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min={0.3}
                        max={1.0}
                        step={0.05}
                        value={dcpOpts.omega}
                        onChange={(e) => setDcpOpts((o) => ({ ...o, omega: Number(e.target.value) }))}
                        className="w-full accent-ocean-400"
                      />
                    </div>
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-white/40">Patch size</span>
                        <span className="text-xs text-white/60 tabular-nums">{dcpOpts.patchSize} px</span>
                      </div>
                      <input
                        type="range"
                        min={5}
                        max={31}
                        step={2}
                        value={dcpOpts.patchSize}
                        onChange={(e) => setDcpOpts((o) => ({ ...o, patchSize: Number(e.target.value) }))}
                        className="w-full accent-ocean-400"
                      />
                    </div>
                  </div>
                )}

                {/* CLAHE options */}
                {dehazeMethod === 'clahe' && (
                  <div className="space-y-3">
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-white/40">Clip limit</span>
                        <span className="text-xs text-white/60 tabular-nums">{claheOpts.clipLimit.toFixed(1)}</span>
                      </div>
                      <input
                        type="range"
                        min={1.0}
                        max={5.0}
                        step={0.1}
                        value={claheOpts.clipLimit}
                        onChange={(e) => setClaheOpts((o) => ({ ...o, clipLimit: Number(e.target.value) }))}
                        className="w-full accent-ocean-400"
                      />
                    </div>
                    <div className="space-y-1.5">
                      <span className="text-xs text-white/40">Tile grid</span>
                      <div className="flex items-center gap-1 rounded-full border border-white/10 bg-white/5 p-0.5 mt-1">
                        {([4, 8, 16] as ClaheOptions['tiles'][]).map((t) => (
                          <button
                            key={t}
                            onClick={() => setClaheOpts((o) => ({ ...o, tiles: t }))}
                            className={[
                              'flex-1 rounded-full py-0.5 text-xs font-medium transition-all',
                              claheOpts.tiles === t
                                ? 'bg-white/15 text-white'
                                : 'text-white/40 hover:text-white/70',
                            ].join(' ')}
                          >
                            {t}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="h-px bg-white/5" />

            {/* JPEG quality */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-white/40">JPEG quality</span>
                <span className="text-xs text-white/60 tabular-nums">{jpegQuality}</span>
              </div>
              <input
                type="range"
                min={60}
                max={100}
                step={1}
                value={jpegQuality}
                onChange={(e) => setJpegQuality(Number(e.target.value))}
                className="w-full accent-ocean-400"
              />
            </div>

            {/* Download */}
            <button
              onClick={handleDownload}
              disabled={!hasResult || isBusy}
              className="flex items-center justify-center gap-2 rounded-lg bg-ocean-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-ocean-500 active:bg-ocean-700 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <svg className="size-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download JPEG
            </button>

            {/* Dimensions */}
            {result && (
              <p className="text-xs text-white/20 text-center tabular-nums">
                {result.width} × {result.height}px
              </p>
            )}

            <div className="mt-auto text-center text-xs text-white/10">
              Drag the slider to compare
            </div>
          </aside>

          {/* ── Main column (image + mobile chrome) ─────────────────────── */}
          <div className="relative flex flex-1 flex-col overflow-hidden">

            {/* Image area */}
            <div className="relative flex flex-1 items-center justify-center overflow-hidden p-4 sm:p-6">

              {/* Full-screen spinner — initial enhance only (no image yet) */}
              {isProcessing && !hasResult && (
                <div className="flex flex-col items-center gap-4 text-white/50">
                  <svg className="size-10 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  <span className="text-sm">Enhancing…</span>
                </div>
              )}

              {/* Error */}
              {enhanceError && !isProcessing && (
                <div className="max-w-md rounded-xl border border-red-500/30 bg-red-500/10 px-5 py-4">
                  <p className="text-sm text-red-400">Enhancement failed: {enhanceError}</p>
                </div>
              )}

              {/* Result — always shown when available; overlay appears on top while busy */}
              {hasResult && (
                <div className="relative w-full h-full">
                  <BeforeAfter
                    before={result.beforeBitmap}
                    after={result.afterBitmap}
                    width={result.width}
                    height={result.height}
                  />

                  {/* Busy overlay */}
                  {isBusy && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 rounded-xl bg-black/50 backdrop-blur-sm">
                      <svg className="size-8 animate-spin text-white/70" viewBox="0 0 24 24" fill="none">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                      </svg>
                      <span className="text-sm text-white/60">
                        {isDehazing ? 'Applying dehaze…' : 'Enhancing…'}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ── Mobile bottom bar (hidden on desktop) ───────────────── */}
            <div className="flex sm:hidden shrink-0 items-center justify-around border-t border-white/10 bg-black/40 backdrop-blur-md px-2 py-2 safe-area-bottom">

              {/* Upload new */}
              <button
                onClick={handleUploadNew}
                disabled={isBusy}
                title="Upload new image"
                className="flex flex-col items-center gap-1 rounded-xl px-4 py-2 text-white/60 transition hover:text-white/90 active:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
                <span className="text-[10px]">Upload</span>
              </button>

              {/* Dehaze quick-toggle */}
              <button
                onClick={() => hasResult && !isBusy && setDehazeEnabled((v) => !v)}
                disabled={!hasResult || isBusy}
                title="Toggle dehaze"
                className={[
                  'flex flex-col items-center gap-1 rounded-xl px-4 py-2 transition disabled:opacity-40 disabled:cursor-not-allowed',
                  dehazeEnabled ? 'text-ocean-400' : 'text-white/60 hover:text-white/90',
                ].join(' ')}
              >
                <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v1m0 16v1M4.22 4.22l.7.7m12.16 12.16.7.7M1 12h1m19 0h1M4.22 19.78l.7-.7M18.36 5.64l.7-.7" />
                  <circle cx="12" cy="12" r="4" strokeWidth={2} />
                </svg>
                <span className="text-[10px]">Dehaze</span>
              </button>

              {/* Tune (opens sheet) */}
              <button
                onClick={() => setSheetOpen((v) => !v)}
                title="Settings"
                className={[
                  'flex flex-col items-center gap-1 rounded-xl px-4 py-2 transition',
                  sheetOpen ? 'text-white bg-white/10' : 'text-white/60 hover:text-white/90 active:bg-white/10',
                ].join(' ')}
              >
                <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />
                </svg>
                <span className="text-[10px]">Settings</span>
              </button>

              {/* Download */}
              <button
                onClick={handleDownload}
                disabled={!hasResult || isBusy}
                title="Download JPEG"
                className="flex flex-col items-center gap-1 rounded-xl px-4 py-2 text-ocean-400 transition hover:text-ocean-300 active:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                <span className="text-[10px]">Download</span>
              </button>
            </div>
          </div>

          {/* ── Mobile bottom sheet ──────────────────────────────────────── */}
          {/* Backdrop */}
          <div
            className={[
              'fixed inset-0 z-30 bg-black/60 transition-opacity duration-300 sm:hidden',
              sheetOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none',
            ].join(' ')}
            onClick={() => setSheetOpen(false)}
          />
          {/* Sheet */}
          <div
            className={[
              'fixed inset-x-0 bottom-0 z-40 flex flex-col rounded-t-2xl border-t border-white/10 bg-slate-900 transition-transform duration-300 ease-out sm:hidden',
              sheetOpen ? 'translate-y-0' : 'translate-y-full',
            ].join(' ')}
            style={{ maxHeight: '85dvh' }}
          >
            {/* Drag handle */}
            <div className="flex justify-center pt-3 pb-1 shrink-0">
              <div className="h-1 w-10 rounded-full bg-white/20" />
            </div>
            {/* Scrollable content */}
            <div className="overflow-y-auto px-5 pb-8 pt-2 space-y-6">

              {/* Filename */}
              {result && (
                <p className="truncate text-xs text-white/30 text-center" title={result.filename}>
                  {result.filename}
                </p>
              )}

              {/* Dehaze */}
              <div className="space-y-3">
                <label className="flex items-center gap-3 cursor-pointer select-none">
                  <button
                    role="switch"
                    aria-checked={dehazeEnabled}
                    disabled={!hasResult || isBusy}
                    onClick={() => setDehazeEnabled((v) => !v)}
                    className={[
                      'relative inline-flex h-6 w-11 shrink-0 rounded-full border border-white/20 transition-colors focus:outline-none',
                      dehazeEnabled ? 'bg-ocean-500' : 'bg-white/10',
                      !hasResult || isBusy ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer',
                    ].join(' ')}
                  >
                    <span className={[
                      'pointer-events-none inline-block h-5 w-5 translate-y-[1px] rounded-full bg-white shadow transition-transform',
                      dehazeEnabled ? 'translate-x-[21px]' : 'translate-x-[1px]',
                    ].join(' ')} />
                  </button>
                  <span className="text-base text-white/80">Dehaze</span>
                </label>

                <div className={`space-y-5 transition-opacity ${dehazeEnabled && hasResult && !isBusy ? 'opacity-100' : 'opacity-30 pointer-events-none'}`}>
                  {/* Method selector — pills */}
                  <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 p-0.5">
                    {(['dcp', 'clahe'] as DehazeMethod[]).map((m) => (
                      <button
                        key={m}
                        onClick={() => setDehazeMethod(m)}
                        className={[
                          'flex-1 rounded-full py-1.5 text-sm font-medium transition-all',
                          dehazeMethod === m ? 'bg-white/15 text-white' : 'text-white/40 hover:text-white/70',
                        ].join(' ')}
                      >
                        {m.toUpperCase()}
                      </button>
                    ))}
                  </div>

                  {/* DCP options */}
                  {dehazeMethod === 'dcp' && (
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-white/50">Strength</span>
                          <span className="text-sm text-white/70 tabular-nums">{dcpOpts.omega.toFixed(2)}</span>
                        </div>
                        <input
                          type="range"
                          min={0.3} max={1.0} step={0.05}
                          value={dcpOpts.omega}
                          onChange={(e) => setDcpOpts((o) => ({ ...o, omega: Number(e.target.value) }))}
                          className="w-full accent-ocean-400 h-1.5"
                        />
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-white/50">Patch size</span>
                          <span className="text-sm text-white/70 tabular-nums">{dcpOpts.patchSize} px</span>
                        </div>
                        <input
                          type="range"
                          min={5} max={31} step={2}
                          value={dcpOpts.patchSize}
                          onChange={(e) => setDcpOpts((o) => ({ ...o, patchSize: Number(e.target.value) }))}
                          className="w-full accent-ocean-400 h-1.5"
                        />
                      </div>
                    </div>
                  )}

                  {/* CLAHE options */}
                  {dehazeMethod === 'clahe' && (
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-white/50">Clip limit</span>
                          <span className="text-sm text-white/70 tabular-nums">{claheOpts.clipLimit.toFixed(1)}</span>
                        </div>
                        <input
                          type="range"
                          min={1.0} max={5.0} step={0.1}
                          value={claheOpts.clipLimit}
                          onChange={(e) => setClaheOpts((o) => ({ ...o, clipLimit: Number(e.target.value) }))}
                          className="w-full accent-ocean-400 h-1.5"
                        />
                      </div>
                      <div className="space-y-2">
                        <span className="text-sm text-white/50">Tile grid</span>
                        <div className="flex items-center gap-1 rounded-full border border-white/10 bg-white/5 p-0.5 mt-1">
                          {([4, 8, 16] as ClaheOptions['tiles'][]).map((t) => (
                            <button
                              key={t}
                              onClick={() => setClaheOpts((o) => ({ ...o, tiles: t }))}
                              className={[
                                'flex-1 rounded-full py-1 text-sm font-medium transition-all',
                                claheOpts.tiles === t ? 'bg-white/15 text-white' : 'text-white/40 hover:text-white/70',
                              ].join(' ')}
                            >
                              {t}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="h-px bg-white/5" />

              {/* JPEG quality */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-white/50">JPEG quality</span>
                  <span className="text-sm text-white/70 tabular-nums">{jpegQuality}</span>
                </div>
                <input
                  type="range"
                  min={60} max={100} step={1}
                  value={jpegQuality}
                  onChange={(e) => setJpegQuality(Number(e.target.value))}
                  className="w-full accent-ocean-400 h-1.5"
                />
              </div>

              {/* Dimensions */}
              {result && (
                <p className="text-xs text-white/20 text-center tabular-nums">
                  {result.width} × {result.height}px
                </p>
              )}
            </div>
          </div>

        </div>
      )}

    </div>
  )
}
