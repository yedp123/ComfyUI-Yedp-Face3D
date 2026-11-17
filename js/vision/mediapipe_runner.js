let _visionPromise = null;
let _imageLandmarkerPromise = null;
let _landmarkerResetPromise = null;
let _videoLandmarkerPromise = null;
let _detectCanvas = null;
let _lastVideoTimestampMs = 0;

const DETECT_CANVAS_SIZE = 1024;

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
