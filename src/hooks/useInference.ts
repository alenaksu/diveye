import { useCallback, useEffect, useRef, useState } from 'react';
import * as Comlink from 'comlink';
import { type ExecutionProvider } from '../lib/backend-detect';
import { purgeOldCaches } from '../lib/model-cache';
import type { InferenceWorker, LoadProgress, DehazeOptions } from '../workers/inference.worker';
import workerUrl from '../workers/inference.worker.js?url';

const MODEL_URL = `${import.meta.env.BASE_URL}models/lu2net.onnx`;

export type InferenceStatus =
    | { phase: 'idle' }
    | { phase: 'loading'; loaded: number; total: number }
    | { phase: 'ready' }
    | { phase: 'processing' }
    | { phase: 'error'; message: string };

export function useInference(preferredEP: ExecutionProvider) {
    const workerRef = useRef<Worker | null>(null);
    const apiRef = useRef<Comlink.Remote<InferenceWorker> | null>(null);

    const [status, setStatus] = useState<InferenceStatus>({ phase: 'idle' });
    const [ep, setEP] = useState<ExecutionProvider | null>(null);

    // Boot (or re-boot) the worker + load model whenever preferredEP changes
    useEffect(() => {
        let cancelled = false;

        async function init() {
            try {
                await purgeOldCaches();

                const worker = new Worker(workerUrl, { type: 'module' });
                workerRef.current = worker;

                const api = Comlink.wrap<InferenceWorker>(worker);
                apiRef.current = api;

                setStatus({ phase: 'loading', loaded: 0, total: 0 });
                setEP(null);

                const usedEP = await api.load(
                    MODEL_URL,
                    preferredEP,
                    Comlink.proxy((p: LoadProgress) => {
                        if (!cancelled)
                            setStatus({ phase: 'loading', loaded: p.loaded, total: p.total });
                    })
                );

                if (!cancelled) {
                    setEP(usedEP);
                    setStatus({ phase: 'ready' });
                }
            } catch (err) {
                if (!cancelled) {
                    setStatus({
                        phase: 'error',
                        message: err instanceof Error ? err.message : String(err)
                    });
                }
            }
        }

        init();

        return () => {
            cancelled = true;
            apiRef.current?.[Comlink.releaseProxy]();
            apiRef.current = null;
            workerRef.current?.terminate();
            workerRef.current = null;
        };
    }, [preferredEP]);

    const enhance = useCallback(async (imageData: ImageData): Promise<ImageData> => {
        if (!apiRef.current) throw new Error('Worker not ready');
        setStatus({ phase: 'processing' });
        try {
            const result = await apiRef.current.enhance(
                Comlink.transfer(imageData, [imageData.data.buffer])
            );
            setStatus({ phase: 'ready' });
            return result;
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            setStatus({ phase: 'error', message });
            throw err;
        }
    }, []);

    const dehaze = useCallback(
        async (imageData: ImageData, options: DehazeOptions): Promise<ImageData> => {
            if (!apiRef.current) throw new Error('Worker not ready');
            const result = await apiRef.current.dehaze(
                Comlink.transfer(imageData, [imageData.data.buffer]),
                options
            );
            return result;
        },
        []
    );

    return { status, ep, enhance, dehaze };
}
