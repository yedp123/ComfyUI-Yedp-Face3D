let _visionPromise = null;
let _imageLandmarkerPromise = null;
let _landmarkerResetPromise = null;
let _videoLandmarkerPromise = null;
let _imageSegmenterPromise = null;
let _segWarned = false;
let _detectCanvas = null;
let _segInputCanvas = null;
let _segOutputCanvas = null;
let _lastVideoTimestampMs = 0;

const DETECT_CANVAS_SIZE = 1024;
const SEG_CANVAS_SIZE = 256;

function getModelsBaseUrl() {
    return new URL("../../models", import.meta.url).href.replace(/\/$/, "");
}

async function getVisionLib() {
    if (!_visionPromise) {
        const visionUrl = new URL("../lib/tasks_vision.js", import.meta.url).href;
        _visionPromise = import(visionUrl);
    }
    return _visionPromise;
}

async function createLandmarker(runningMode) {
    const vision = await getVisionLib();
    const modelsBase = getModelsBaseUrl();
    const fileset = await vision.FilesetResolver.forVisionTasks(modelsBase);
    return vision.FaceLandmarker.createFromOptions(fileset, {
        baseOptions: {
            modelAssetPath: `${modelsBase}/face_landmarker.task`,
            delegate: "CPU",
        },
        runningMode,
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
    });
}

async function getImageLandmarker() {
    if (_landmarkerResetPromise) {
        await _landmarkerResetPromise;
    }
    if (!_imageLandmarkerPromise) {
        _imageLandmarkerPromise = createLandmarker("IMAGE");
    }
    return _imageLandmarkerPromise;
}

async function getVideoLandmarker() {
    if (!_videoLandmarkerPromise) {
        _videoLandmarkerPromise = createLandmarker("VIDEO");
    }
    return _videoLandmarkerPromise;
}

function warnSegmentationFallback(err) {
    if (!_segWarned) {
        _segWarned = true;
        console.warn(
            "[YedpFace3D] SelfieMulticlass segmentation unavailable; falling back to visibility-only weights:",
            err?.message || err
        );
    }
}

async function getImageSegmenter() {
    if (!_imageSegmenterPromise) {
        _imageSegmenterPromise = (async () => {
            const vision = await getVisionLib();
            if (!vision?.ImageSegmenter) {
                throw new Error("ImageSegmenter not exported by tasks_vision.js");
            }
            const modelsBase = getModelsBaseUrl();
            const fileset = await vision.FilesetResolver.forVisionTasks(modelsBase);
            return vision.ImageSegmenter.createFromOptions(fileset, {
                baseOptions: {
                    modelAssetPath: `${modelsBase}/selfie_multiclass_256x256.tflite`,
                    delegate: "CPU",
                },
                runningMode: "IMAGE",
                outputCategoryMask: false,
                outputConfidenceMasks: true,
            });
        })().catch((err) => {
            warnSegmentationFallback(err);
            return null;
        });
    }
    return _imageSegmenterPromise;
}

/**
 * Dispose the cached IMAGE FaceLandmarker so full resets clear all internal WASM graph state.
 */
export async function resetImageLandmarker() {
    const prev = _imageLandmarkerPromise;
    _imageLandmarkerPromise = null;
    if (!prev) return;
    _landmarkerResetPromise = prev
        .then((lm) => lm?.close?.())
        .catch(() => {})
        .finally(() => {
            _landmarkerResetPromise = null;
        });
    await _landmarkerResetPromise;
}

function getFixedDetectCanvas() {
    if (!_detectCanvas) {
        _detectCanvas = document.createElement("canvas");
        _detectCanvas.width = DETECT_CANVAS_SIZE;
        _detectCanvas.height = DETECT_CANVAS_SIZE;
    }
    return _detectCanvas;
}

/**
 * Draw every still image onto a fixed 1024x1024 canvas so MediaPipe's internal
 * WebGL texture/canvas size never mutates across differently sized uploads.
 */
function prepareFixedDetectInput(img) {
    const w = img.naturalWidth || img.width || 512;
    const h = img.naturalHeight || img.height || 512;
    const size = DETECT_CANVAS_SIZE;
    const scale = Math.min(size / w, size / h);
    const drawW = w * scale;
    const drawH = h * scale;
    const padX = (size - drawW) * 0.5;
    const padY = (size - drawH) * 0.5;

    const canvas = getFixedDetectCanvas();
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#12151c";
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(img, padX, padY, drawW, drawH);

    return { canvas, width: w, height: h, padX, padY, drawW, drawH, size };
}

function unmapLetterboxLandmarks(formatted, box) {
    if (!formatted?.landmarks?.length) return formatted;
    const xNorm = box.drawW / box.size;
    const yNorm = box.drawH / box.size;
    const xOff = box.padX / box.size;
    const yOff = box.padY / box.size;
    const zScale = box.size / box.drawW;

    for (const lm of formatted.landmarks) {
        lm.x = (lm.x - xOff) / xNorm;
        lm.y = (lm.y - yOff) / yNorm;
        lm.z = lm.z * zScale;
    }
    formatted.width = box.width;
    formatted.height = box.height;
    return formatted;
}

export function loadImageElement(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("Failed to decode image."));
        img.src = src;
    });
}

function formatResult(result, width, height) {
    if (!result?.faceLandmarks?.length) return null;
    const landmarks = result.faceLandmarks[0].map((p) => ({
        x: p.x,
        y: p.y,
        z: p.z,
    }));
    const matrix = result.facialTransformationMatrixes?.[0]?.data
        ? Array.from(result.facialTransformationMatrixes[0].data)
        : null;
    const blendshapes = result.faceBlendshapes?.[0]?.categories || null;
    return { landmarks, matrix, blendshapes, width, height };
}

async function detectImageDataUrl(dataUrl, required) {
    if (!dataUrl) return null;
    const [landmarker, img] = await Promise.all([
        getImageLandmarker(),
        loadImageElement(dataUrl),
    ]);
    const box = prepareFixedDetectInput(img);
    const formatted = formatResult(landmarker.detect(box.canvas), box.size, box.size);
    if (!formatted) {
        if (required) {
            throw new Error("No face detected in reference image.");
        }
        return null;
    }
    return unmapLetterboxLandmarks(formatted, box);
}

/**
 * Extract 478 3D landmarks and 4x4 facial transformation matrix from a data URL (IMAGE mode).
 */
export async function extractFaceFromDataUrl(dataUrl) {
    return detectImageDataUrl(dataUrl, true);
}

function getFixedSegCanvases() {
    if (!_segInputCanvas) {
        _segInputCanvas = document.createElement("canvas");
        _segInputCanvas.width = SEG_CANVAS_SIZE;
        _segInputCanvas.height = SEG_CANVAS_SIZE;
    }
    if (!_segOutputCanvas) {
        _segOutputCanvas = document.createElement("canvas");
        _segOutputCanvas.width = SEG_CANVAS_SIZE;
        _segOutputCanvas.height = SEG_CANVAS_SIZE;
    }
    return { inputCanvas: _segInputCanvas, outputCanvas: _segOutputCanvas };
}

function sampleMaskBilinear(arr, w, h, x, y) {
    if (!arr) return 0;
    const cx = Math.max(0, Math.min(w - 1, x));
    const cy = Math.max(0, Math.min(h - 1, y));
    const x0 = Math.floor(cx);
    const y0 = Math.floor(cy);
    const x1 = Math.min(w - 1, x0 + 1);
    const y1 = Math.min(h - 1, y0 + 1);
    const fx = cx - x0;
    const fy = cy - y0;
    const r0 = y0 * w;
    const r1 = y1 * w;
    const v00 = arr[r0 + x0];
    const v10 = arr[r0 + x1];
    const v01 = arr[r1 + x0];
    const v11 = arr[r1 + x1];
    return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
}

/**
 * Run offline SelfieMulticlass segmentation on a reference image and pack soft probabilities
 * into a 256x256 RGB PNG data URL (R=face_skin, G=hair, B=non_skin [bg+clothes+others]).
 */
export async function segmentImageDataUrl(dataUrl) {
    if (!dataUrl) return null;
    let result = null;
    try {
        const [segmenter, img] = await Promise.all([
            getImageSegmenter(),
            loadImageElement(dataUrl),
        ]);
        if (!segmenter) return null;

        const w = img.naturalWidth || img.width || 512;
        const h = img.naturalHeight || img.height || 512;
        const size = SEG_CANVAS_SIZE;
        const scale = Math.min(size / w, size / h);
        const drawW = w * scale;
        const drawH = h * scale;
        const padX = (size - drawW) * 0.5;
        const padY = (size - drawH) * 0.5;

        const { inputCanvas, outputCanvas } = getFixedSegCanvases();
        const inCtx = inputCanvas.getContext("2d");
        inCtx.fillStyle = "#12151c";
        inCtx.fillRect(0, 0, size, size);
        inCtx.imageSmoothingEnabled = true;
        inCtx.imageSmoothingQuality = "high";
        inCtx.drawImage(img, padX, padY, drawW, drawH);

        result = segmenter.segment(inputCanvas);
        const masks = result?.confidenceMasks;
        if (!masks || masks.length < 4) return null;

        const maskW = masks[0].width || size;
        const maskH = masks[0].height || size;
        const c0 = masks[0].getAsFloat32Array();
        const c1 = masks[1].getAsFloat32Array();
        const c3 = masks[3].getAsFloat32Array();
        const c4 = masks[4]?.getAsFloat32Array?.() || null;
        const c5 = masks[5]?.getAsFloat32Array?.() || null;

        const outCtx = outputCanvas.getContext("2d");
        const imgData = outCtx.createImageData(size, size);
        const dst = imgData.data;
        const sxScale = maskW / size;
        const syScale = maskH / size;

        for (let oy = 0; oy < size; oy++) {
            const cy = padY + ((oy + 0.5) / size) * drawH;
            const my = cy * syScale - 0.5;
            const rowOff = oy * size;
            for (let ox = 0; ox < size; ox++) {
                const cx = padX + ((ox + 0.5) / size) * drawW;
                const mx = cx * sxScale - 0.5;

                const faceSkin = sampleMaskBilinear(c3, maskW, maskH, mx, my);
                const hair = sampleMaskBilinear(c1, maskW, maskH, mx, my);
                const bg = sampleMaskBilinear(c0, maskW, maskH, mx, my);
                const clothes = c4 ? sampleMaskBilinear(c4, maskW, maskH, mx, my) : 0;
                const others = c5 ? sampleMaskBilinear(c5, maskW, maskH, mx, my) : 0;
                const nonSkin = Math.max(0, Math.min(1, bg + clothes + others));

                const p = (rowOff + ox) * 4;
                dst[p] = Math.round(Math.max(0, Math.min(1, faceSkin)) * 255);
                dst[p + 1] = Math.round(Math.max(0, Math.min(1, hair)) * 255);
                dst[p + 2] = Math.round(nonSkin * 255);
                dst[p + 3] = 255;
            }
        }

        outCtx.putImageData(imgData, 0, 0);
        return outputCanvas.toDataURL("image/png");
    } catch (err) {
        warnSegmentationFallback(err);
        return null;
    } finally {
        if (result?.confidenceMasks) {
            for (const m of result.confidenceMasks) {
                m?.close?.();
            }
        }
    }
}

/**
 * Extract landmarks, pose matrix, and 52 ARKit blendshapes from a video/canvas frame (VIDEO mode).
 */
export async function detectFaceForVideo(source, timestampMs) {
    const landmarker = await getVideoLandmarker();
    const width = source.videoWidth || source.naturalWidth || source.width || 640;
    const height = source.videoHeight || source.naturalHeight || source.height || 480;
    const ts = Math.max(_lastVideoTimestampMs + 1, Math.floor(timestampMs || performance.now()));
    _lastVideoTimestampMs = ts;
    const result = landmarker.detectForVideo(source, ts);
    return formatResult(result, width, height);
}

/**
 * Extract landmarks, pose matrix, and 52 ARKit blendshapes from an image data URL without throwing.
 */
export async function detectOptionalFaceFromDataUrl(dataUrl) {
    return detectImageDataUrl(dataUrl, false);
}

/**
 * Convert MediaPipe's FACE_LANDMARKS_TESSELATION edge triples into flat triangle indices.
 */
export async function getTesselationTriangles() {
    const vision = await getVisionLib();
    const edges = vision.FaceLandmarker.FACE_LANDMARKS_TESSELATION;
    const indices = [];
    for (let i = 0; i + 2 < edges.length; i += 3) {
        indices.push(edges[i].start, edges[i + 1].start, edges[i + 2].start);
    }
    return indices;
}
