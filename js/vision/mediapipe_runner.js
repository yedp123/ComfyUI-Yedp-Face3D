let _visionPromise = null;
let _imageLandmarkerPromise = null;
let _videoLandmarkerPromise = null;

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

/**
 * Extract 478 3D landmarks and 4x4 facial transformation matrix from a data URL (IMAGE mode).
 */
export async function extractFaceFromDataUrl(dataUrl) {
    if (!dataUrl) return null;
    const [landmarker, img] = await Promise.all([
        getImageLandmarker(),
        loadImageElement(dataUrl),
    ]);
    const formatted = formatResult(
        landmarker.detect(img),
        img.naturalWidth || img.width,
        img.naturalHeight || img.height
    );
    if (!formatted) {
        throw new Error("No face detected in reference image.");
    }
    return formatted;
}

/**
 * Extract landmarks, pose matrix, and 52 ARKit blendshapes from a video/canvas frame (VIDEO mode).
 */
export async function detectFaceForVideo(source, timestampMs) {
    const landmarker = await getVideoLandmarker();
    const width = source.videoWidth || source.naturalWidth || source.width || 640;
    const height = source.videoHeight || source.naturalHeight || source.height || 480;
    const result = landmarker.detectForVideo(source, timestampMs);
    return formatResult(result, width, height);
}

/**
 * Extract landmarks, pose matrix, and 52 ARKit blendshapes from an image data URL without throwing.
 */
export async function detectOptionalFaceFromDataUrl(dataUrl) {
    if (!dataUrl) return null;
    const [landmarker, img] = await Promise.all([
        getImageLandmarker(),
        loadImageElement(dataUrl),
    ]);
    return formatResult(
        landmarker.detect(img),
        img.naturalWidth || img.width,
        img.naturalHeight || img.height
    );
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
