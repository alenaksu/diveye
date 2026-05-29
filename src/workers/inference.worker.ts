/**
 * Inference Web Worker — runs LU2Net via ONNX Runtime Web.
 * Exposed via comlink so the main thread can call enhance() as an async function.
 *
 * The input image is downscaled so its longest side is at most MAX_INFERENCE_DIM
 * before being passed to the model.  This caps memory usage while keeping enough
 * spatial resolution for accurate global colour correction.
 */

import * as Comlink from 'comlink'
import * as ort from 'onnxruntime-web/all'
import { fetchModel } from '../lib/model-cache'
import {
  imageDataToTensor,
  tensorToImageData,
  downscaleImageData,
} from '../lib/image-utils'
import { dehaze as dehazeImpl } from '../lib/dehaze'
import type { DehazeOptions } from '../lib/dehaze/types'
import type { ExecutionProvider } from '../lib/backend-detect'

// Point ORT to its WASM binaries.
// Must use BASE_URL so the path is correct both in dev (/) and on GitHub Pages (/aquatune/).
ort.env.wasm.wasmPaths = `${import.meta.env.BASE_URL}onnxruntime-web/`

let session: ort.InferenceSession | null = null
let activeEP: ExecutionProvider = 'wasm'

/**
 * Longest side (px) fed to the model.  Images larger than this are downscaled
 * before inference; the model output is returned at the downscaled dimensions.
 */
const MAX_INFERENCE_DIM = 2048

export type LoadProgress = { loaded: number; total: number }

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Run a single ORT forward pass on an ImageData and return the result as a
 * new ImageData of the same dimensions.  Intermediate tensor references are
 * nulled out immediately after use so the GC can reclaim them promptly.
 */
/** Compute mean of a Float32Array slice. */
function sliceMean(arr: Float32Array, offset: number, len: number): number {
  let sum = 0
  for (let i = 0; i < len; i++) sum += arr[offset + i]
  return sum / len
}

/** Compute mean absolute difference between two Float32Array slices. */
function sliceMAD(a: Float32Array, aOff: number, b: Float32Array, bOff: number, len: number): number {
  let sum = 0
  for (let i = 0; i < len; i++) sum += Math.abs(a[aOff + i] - b[bOff + i])
  return sum / len
}

async function runInference(
  sess: ort.InferenceSession,
  imageData: ImageData,
): Promise<ImageData> {
  const { data, paddedW, paddedH } = imageDataToTensor(imageData)
  const planeSize = paddedW * paddedH

  const inputTensor = new ort.Tensor('float32', data, [1, 3, paddedH, paddedW])
  const feeds: Record<string, ort.Tensor> = { input: inputTensor }

  const results = await sess.run(feeds)
  const outputTensor = results['output']
  const outputData = outputTensor.data as Float32Array

  // --- Diagnostics ---
  const inR  = sliceMean(data,       0,           planeSize)
  const inG  = sliceMean(data,       planeSize,   planeSize)
  const inB  = sliceMean(data,       planeSize*2, planeSize)
  const outR = sliceMean(outputData, 0,           planeSize)
  const outG = sliceMean(outputData, planeSize,   planeSize)
  const outB = sliceMean(outputData, planeSize*2, planeSize)
  const mad  = (
    sliceMAD(data, 0,           outputData, 0,           planeSize) +
    sliceMAD(data, planeSize,   outputData, planeSize,   planeSize) +
    sliceMAD(data, planeSize*2, outputData, planeSize*2, planeSize)
  ) / 3

  const outDims = outputTensor.dims
  console.log(
    `[LU2Net] EP=${activeEP} | ` +
    `size=${imageData.width}×${imageData.height} (padded ${paddedW}×${paddedH}) | ` +
    `out dims=[${outDims.join(',')}] | ` +
    `in  R=${(inR*255).toFixed(1)} G=${(inG*255).toFixed(1)} B=${(inB*255).toFixed(1)} | ` +
    `out R=${(outR*255).toFixed(1)} G=${(outG*255).toFixed(1)} B=${(outB*255).toFixed(1)} | ` +
    `MAD=${(mad*255).toFixed(2)} counts`
  )
  // -------------------

  const out = tensorToImageData(outputData, imageData.width, imageData.height, paddedW, paddedH)

  // Dispose ORT-owned GPU/WASM buffers if the runtime supports it,
  // then drop all JS references so the GC can reclaim the Float32Arrays.
  if (typeof (outputTensor as { dispose?: () => void }).dispose === 'function') {
    (outputTensor as { dispose: () => void }).dispose()
  }
  // @ts-expect-error — clear reference for GC
  results['output'] = null

  return out
}

// ---------------------------------------------------------------------------
// Exposed worker API
// ---------------------------------------------------------------------------

const worker = {
  /**
   * Load the ONNX model. Tries EPs in order: webgpu → webgl → wasm.
   * Reports download progress via the callback.
   */
  async load(
    modelUrl: string,
    _preferredEP: ExecutionProvider,
    onProgress: (p: LoadProgress) => void,
  ): Promise<ExecutionProvider> {
    const buffer = await fetchModel(modelUrl, (loaded, total) =>
      onProgress({ loaded, total }),
    )

    // Pre-create a GPUDevice without shader-f16 so ORT uses f32 WGSL shaders.
    // Note: WebGPU still produces near-zero corrections due to a separate
    // precision issue in ORT's buffer layer — but we keep this in case a future
    // ORT version fixes it.
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        const adapter = await (navigator as any).gpu.requestAdapter()
        if (adapter) {
          const device = await adapter.requestDevice({ requiredFeatures: [] })
          ort.env.webgpu.device = device
        }
      } catch {
        // No GPU available.
      }
    }

    // Known limitations per EP (for user awareness):
    //   webgpu — near-zero corrections (MAD ≈ 0.23) due to ORT buffer precision bug
    //   webgl  — fails at session init: 'Erf' op not supported (used by GELU)
    //   wasm   — correct results, ~5–15s on modern hardware
    //
    // When a specific EP is requested we try it first; if it fails we fall back
    // to wasm so the user always gets a result.
    const epOrder: ExecutionProvider[] =
      _preferredEP === 'wasm'
        ? ['wasm']
        : [_preferredEP, 'wasm']

    for (const ep of epOrder) {
      try {
        session = await ort.InferenceSession.create(buffer, {
          executionProviders: [ep],
          graphOptimizationLevel: 'all',
        })
        activeEP = ep
        return ep
      } catch (err) {
        console.warn(`[ORT] EP failed: ${ep} —`, err)
      }
    }

    throw new Error('No supported execution provider found')
  },

  /**
   * Run LU2Net on an ImageData. The image is downscaled to MAX_INFERENCE_DIM
   * on its longest side before inference. Returns the enhanced ImageData at
   * the (possibly downscaled) dimensions.
   */
  async enhance(imageData: ImageData): Promise<ImageData> {
    if (!session) throw new Error('Model not loaded. Call load() first.')
    const scaled = downscaleImageData(imageData, MAX_INFERENCE_DIM)
    const out = await runInference(session, scaled)
    return Comlink.transfer(out, [out.data.buffer])
  },

  /** Return the currently active execution provider. */
  getEP(): ExecutionProvider {
    return activeEP
  },

  /**
   * Apply dehaze post-processing to an ImageData.
   * Runs in the worker to keep the main thread free.
   */
  dehaze(imageData: ImageData, options: DehazeOptions): ImageData {
    const out = dehazeImpl(imageData, options)
    return Comlink.transfer(out, [out.data.buffer])
  },
}

Comlink.expose(worker)

export type { DehazeOptions }
export type InferenceWorker = typeof worker
