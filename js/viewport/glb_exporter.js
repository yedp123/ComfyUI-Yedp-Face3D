import { FaceRetargetRig } from "../vision/retarget_rig.js";
import { EYE_SPECS, fitEyeSocket, clampEyelidsOverEyeball } from "./eyeball_rig.js";

export const ARKIT_52_NAMES = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "tongueOut",
];

const UPPER_LIP_IDS = [0, 11, 12, 13, 37, 38, 39, 40, 41, 42, 62, 72, 73, 74, 76, 77, 80, 81, 82, 183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 292, 302, 303, 304, 306, 307, 310, 311, 312, 407, 408, 409, 415];
const LOWER_LIP_IDS = [14, 15, 16, 17, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96, 146, 178, 179, 180, 181, 314, 315, 316, 317, 318, 319, 320, 321, 324, 325, 375, 402, 403, 404, 405];
const CORNER_LIP_IDS = [61, 78, 291, 308];
const BROW_IDS = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107, 276, 283, 282, 295, 285, 300, 293, 334, 296, 336];
const JAW_IDS = [152, 148, 176, 149, 150, 136, 172, 58, 132, 377, 400, 378, 379, 365, 397, 288, 361, 175, 199, 200, 201, 421, 418, 262, 428, 32, 140, 171, 396, 369];
const CHEEK_IDS = [50, 280, 101, 330, 118, 347, 123, 352, 187, 411, 205, 425, 207, 427];
const NOSE_WING_IDS = [48, 115, 220, 45, 278, 344, 440, 275];
const NOSE_BONE_LOCK = new Set([168, 6, 197, 195, 5, 4, 1]);

function smoothstep(edge0, edge1, x) {
    const t = Math.max(0, Math.min(1, (x - edge0) / Math.max(edge1 - edge0, 1e-6)));
    return t * t * (3 - 2 * t);
}

function computeVec3MinMax(arr) {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i + 2 < arr.length; i += 3) {
        const x = arr[i];
        const y = arr[i + 1];
        const z = arr[i + 2];
        if (x < min[0]) min[0] = x;
        if (y < min[1]) min[1] = y;
        if (z < min[2]) min[2] = z;
        if (x > max[0]) max[0] = x;
        if (y > max[1]) max[1] = y;
        if (z > max[2]) max[2] = z;
    }
    if (!Number.isFinite(min[0])) return { min: [0, 0, 0], max: [0, 0, 0] };
    return { min, max };
}

function computeNormals(positions, indices) {
    const nVerts = Math.floor(positions.length / 3);
    const norms = new Float32Array(nVerts * 3);
    for (let i = 0; i + 2 < indices.length; i += 3) {
        const ia = indices[i] * 3;
        const ib = indices[i + 1] * 3;
        const ic = indices[i + 2] * 3;
        const ax = positions[ia], ay = positions[ia + 1], az = positions[ia + 2];
        const ux = positions[ib] - ax, uy = positions[ib + 1] - ay, uz = positions[ib + 2] - az;
        const vx = positions[ic] - ax, vy = positions[ic + 1] - ay, vz = positions[ic + 2] - az;
        const nx = uy * vz - uz * vy;
        const ny = uz * vx - ux * vz;
        const nz = ux * vy - uy * vx;
        norms[ia] += nx; norms[ia + 1] += ny; norms[ia + 2] += nz;
        norms[ib] += nx; norms[ib + 1] += ny; norms[ib + 2] += nz;
        norms[ic] += nx; norms[ic + 1] += ny; norms[ic + 2] += nz;
    }
    for (let i = 0; i < norms.length; i += 3) {
        const len = Math.hypot(norms[i], norms[i + 1], norms[i + 2]);
        if (len > 1e-6) {
            const inv = 1.0 / len;
            norms[i] *= inv;
            norms[i + 1] *= inv;
            norms[i + 2] *= inv;
        } else {
            norms[i + 2] = 1.0;
        }
    }
    return norms;
}

function diffuseMask(nVerts, neighbors, seedFn, lockSet = NOSE_BONE_LOCK, passes = 5) {
    const cur = new Float32Array(nVerts);
    const nxt = new Float32Array(nVerts);
    const seedKind = new Uint8Array(nVerts);
    for (let i = 0; i < nVerts; i++) {
        if (lockSet && lockSet.has(i)) {
            seedKind[i] = 1;
            continue;
        }
        const v = seedFn(i);
        if (v !== 0) {
            cur[i] = v;
            seedKind[i] = 2;
        }
    }
    for (let iter = 0; iter < passes; iter++) {
        for (let i = 0; i < nVerts; i++) {
            if (seedKind[i] === 1) {
                nxt[i] = 0;
                continue;
            }
            if (seedKind[i] === 2) {
                nxt[i] = cur[i];
                continue;
            }
            const nbrs = neighbors[i];
            if (!nbrs || !nbrs.length) {
                nxt[i] = cur[i];
                continue;
            }
            let sum = 0;
            for (let k = 0; k < nbrs.length; k++) sum += cur[nbrs[k]];
            nxt[i] = 0.25 * cur[i] + 0.75 * (sum / nbrs.length);
        }
        cur.set(nxt);
    }
    return cur;
}

/**
 * Extract target export topology (Face Only vs Face + Head, Base vs Smooth) with vertex compaction.
 */
export function extractExportTopology(faceMesh, { scope = "head", subdiv = "smooth" } = {}) {
    const isHead = scope === "head" && Boolean(faceMesh.shellRig);
    const isSmooth = subdiv === "smooth" && Boolean(faceMesh.subdivider);
    const faceIndicesCount = faceMesh.faceIndicesCount || faceMesh.baseIndices.length;

    if (isHead) {
        const rawIndices = isSmooth ? faceMesh.subdivider.subdivIndices : faceMesh.mergedIndices;
        const vertCount = isSmooth
            ? faceMesh.subdivider.totalVerts
            : Math.floor(faceMesh.mergedPositions.length / 3);
        const mergedBaseCount = Math.floor(faceMesh.mergedPositions.length / 3);
        const pureShellMask = new Uint8Array(vertCount);
        for (let v = 478; v < Math.min(mergedBaseCount, vertCount); v++) {
            if (faceMesh.shellRig.nearestSocketMp[v - 478] < 0) {
                pureShellMask[v] = 1;
            }
        }
        if (isSmooth && faceMesh.subdivider?.edgePairs) {
            const edges = faceMesh.subdivider.edgePairs;
            for (let e = 0; e < edges.length; e += 3) {
                const a = edges[e];
                const b = edges[e + 1];
                const m = edges[e + 2];
                if (m < vertCount && pureShellMask[a] && pureShellMask[b]) {
                    pureShellMask[m] = 1;
                }
            }
        }
        return {
            isHead: true,
            isSmooth,
            vertCount,
            indices: new Uint16Array(rawIndices),
            remapIndices: null,
            pureShellMask,
        };
    }

    if (!isSmooth) {
        return {
            isHead: false,
            isSmooth: false,
            vertCount: 478,
            indices: new Uint16Array(faceMesh.baseIndices.slice(0, faceIndicesCount)),
            remapIndices: null,
            pureShellMask: new Uint8Array(478),
        };
    }

    const faceSubdivLimit = Math.min(faceMesh.subdivider.subdivIndices.length, faceIndicesCount * 4);
    const rawSlice = faceMesh.subdivider.subdivIndices.slice(0, faceSubdivLimit);
    const vertMap = new Int32Array(faceMesh.subdivider.totalVerts).fill(-1);
    const remap = [];
    const compacted = new Uint16Array(rawSlice.length);
    for (let i = 0; i < rawSlice.length; i++) {
        const orig = rawSlice[i];
        let mapped = vertMap[orig];
        if (mapped === -1) {
            mapped = remap.length;
            vertMap[orig] = mapped;
            remap.push(orig);
        }
        compacted[i] = mapped;
    }

    return {
        isHead: false,
        isSmooth: true,
        vertCount: remap.length,
        indices: compacted,
        remapIndices: new Int32Array(remap),
        pureShellMask: new Uint8Array(remap.length),
    };
}

export function expandFacePositionsToTopology(faceMesh, facePos478, topo) {
    let source = facePos478;
    if ((topo.isHead || topo.isSmooth) && faceMesh.shellRig && faceMesh.shellFitState && faceMesh.mergedScratch) {
        faceMesh.shellRig.deformInto(facePos478, faceMesh.shellFitState, faceMesh.mergedScratch);
        source = faceMesh.mergedScratch;
    }
    if (topo.isSmooth && faceMesh.subdivider) {
        source = faceMesh.subdivider.evaluate(source);
    }
    if (!topo.remapIndices) {
        return new Float32Array(source.subarray(0, topo.vertCount * 3));
    }
    const out = new Float32Array(topo.vertCount * 3);
    const remap = topo.remapIndices;
    for (let i = 0; i < remap.length; i++) {
        const s = remap[i] * 3;
        const d = i * 3;
        out[d] = source[s];
        out[d + 1] = source[s + 1];
        out[d + 2] = source[s + 2];
    }
    return out;
}

function getSkinRgbBytes(faceMesh) {
    const c = faceMesh.material?.uniforms?.uSkinColor?.value;
    if (!c) return [216, 180, 160];
    // Convert linear uniform back to sRGB bytes for 2D canvas PNG export
    const toSrgb = (lin) => {
        const v = Math.max(0, Math.min(1, lin));
        const s = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
        return Math.max(0, Math.min(255, Math.round(s * 255)));
    };
    return [toSrgb(c.r), toSrgb(c.g), toSrgb(c.b)];
}

function getScleraRgbBytes(faceMesh) {
    const c = faceMesh.material?.uniforms?.uScleraColor?.value;
    if (!c) return [214, 209, 203];
    const toSrgb = (lin) => {
        const v = Math.max(0, Math.min(1, lin));
        const s = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
        return Math.max(0, Math.min(255, Math.round(s * 255)));
    };
    return [toSrgb(c.r), toSrgb(c.g), toSrgb(c.b)];
}

async function canvasToPngBytes(canvas) {
    if (typeof canvas.toBlob === "function") {
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
        if (blob) {
            return new Uint8Array(await blob.arrayBuffer());
        }
    }
    // Minimal 1x1 transparent/solid PNG fallback for headless Node test environments without canvas.toBlob
    return new Uint8Array([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
        0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
        0xB0, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
        0x44, 0xAE, 0x42, 0x60, 0x82,
    ]);
}

function getActiveTextureImage(faceMesh) {
    if (faceMesh.isMultiViewTextureActive?.() && faceMesh.bakedTexture?.image) {
        return faceMesh.bakedTexture.image;
    }
    if (faceMesh.hasTexture && faceMesh.frontTexture?.image) {
        return faceMesh.frontTexture.image;
    }
    return null;
}

/**
 * Bake the active face UV texture (or solid clay/skin fallback) with a top-left 12x12 skin-color
 * corner patch so non-socket head_shell vertices render seamlessly in Maya/Blender/Unreal.
 */
export async function bakeFaceTexturePngBytes(faceMesh, size = 1024) {
    const [sr, sg, sb] = getSkinRgbBytes(faceMesh);
    const skinCss = `rgb(${sr}, ${sg}, ${sb})`;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (ctx) {
        ctx.fillStyle = skinCss;
        ctx.fillRect(0, 0, size, size);
        const img = getActiveTextureImage(faceMesh);
        if (img) {
            ctx.drawImage(img, 0, 0, size, size);
        }
        ctx.fillStyle = skinCss;
        ctx.fillRect(0, 0, 12, 12);
    }
    return canvasToPngBytes(canvas);
}

/**
 * Bake a 256x256 eyeball iris+sclera PNG matching face_shader.js's radial uIrisMaskScale blend.
 */
async function bakeEyeballTexturePngBytes(faceMesh, eye, size = 256) {
    const [scr, scg, scb] = getScleraRgbBytes(faceMesh);
    const scleraCss = `rgb(${scr}, ${scg}, ${scb})`;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (!ctx) return canvasToPngBytes(canvas);

    ctx.fillStyle = scleraCss;
    ctx.fillRect(0, 0, size, size);

    const img = getActiveTextureImage(faceMesh);
    if (img && eye.restSocket) {
        const irisMaskScale = Number(faceMesh.material?.uniforms?.uIrisMaskScale?.value ?? 1.08);
        const srcCanvas = document.createElement("canvas");
        srcCanvas.width = size;
        srcCanvas.height = size;
        const sctx = srcCanvas.getContext("2d");
        sctx.drawImage(img, 0, 0, size, size);
        const srcData = sctx.getImageData(0, 0, size, size).data;
        const outImg = ctx.createImageData(size, size);
        const dst = outImg.data;

        // Paint UV texture sampled at the eyeball's UV strip and blend outer sclera using uIrisMaskScale
        const uvs = eye.uv;
        const sbArr = eye.scleraBlend;
        // Build a fast lookup: draw portrait directly, then blend outer pixels radially from iris center
        const irisU = uvs[0];
        const irisV = 1.0 - uvs[1];
        // Estimate UV radius of r_norm = 1.0 from ring 2..3 samples
        let uvRadSum = 0;
        let uvRadCnt = 0;
        for (let i = 1; i < sbArr.length; i++) {
            if (sbArr[i] > 0.25 && sbArr[i] < 1.8) {
                const du = uvs[i * 2] - irisU;
                const dv = (1.0 - uvs[i * 2 + 1]) - irisV;
                uvRadSum += Math.hypot(du, dv) / sbArr[i];
                uvRadCnt++;
            }
        }
        const unitUvRad = uvRadCnt > 0 ? Math.max(uvRadSum / uvRadCnt, 1e-3) : 0.02;

        for (let py = 0; py < size; py++) {
            const v = (py + 0.5) / size;
            for (let px = 0; px < size; px++) {
                const u = (px + 0.5) / size;
                const rNorm = Math.hypot(u - irisU, v - irisV) / unitUvRad;
                const scleraF = smoothstep(irisMaskScale, irisMaskScale + 0.18, rNorm);
                const socketShade = 1.0 + (0.86 - 1.0) * smoothstep(irisMaskScale, irisMaskScale + 1.4, rNorm);
                const o = (py * size + px) * 4;
                dst[o] = Math.round(srcData[o] * (1 - scleraF) + scr * socketShade * scleraF);
                dst[o + 1] = Math.round(srcData[o + 1] * (1 - scleraF) + scg * socketShade * scleraF);
                dst[o + 2] = Math.round(srcData[o + 2] * (1 - scleraF) + scb * socketShade * scleraF);
                dst[o + 3] = 255;
            }
        }
        ctx.putImageData(outImg, 0, 0);
    }
    return canvasToPngBytes(canvas);
}

/**
 * Generate all 52 standard ARKit morph target deltas in the target export topology.
 */
export function buildArkit52MorphTargets(faceMesh, topo) {
    const rest478 = faceMesh.restPositions;
    const n478 = 478;
    const rig = faceMesh.morphRig || new FaceRetargetRig(rest478, faceMesh.baseIndices, faceMesh.regionWeights);
    const nbrs = rig.neighbors;
    const eyeSpan = rig.charEyeSpan || 0.24;
    const chinDrop = rig.charChinDrop || 0.28;

    const neutralExpanded = expandFacePositionsToTopology(faceMesh, rest478, topo);

    const upperLipSet = new Set(UPPER_LIP_IDS);
    const lowerLipSet = new Set(LOWER_LIP_IDS);
    const cornerLipSet = new Set(CORNER_LIP_IDS);
    const browSet = new Set(BROW_IDS);
    const jawSet = new Set(JAW_IDS);
    const cheekSet = new Set(CHEEK_IDS);
    const noseWingSet = new Set(NOSE_WING_IDS);

    const sideSign = (i, isLeft) => {
        const x = rest478[i * 3];
        if (isLeft) return x >= -0.04 * eyeSpan ? smoothstep(-0.04 * eyeSpan, 0.18 * eyeSpan, x) : 0;
        return x <= 0.04 * eyeSpan ? smoothstep(0.04 * eyeSpan, -0.18 * eyeSpan, x) : 0;
    };

    const jawField = diffuseMask(n478, nbrs, (i) => (jawSet.has(i) ? 1 : (lowerLipSet.has(i) ? 0.72 : 0)));
    const upperLipField = diffuseMask(n478, nbrs, (i) => (upperLipSet.has(i) ? 1 : 0));
    const lowerLipField = diffuseMask(n478, nbrs, (i) => (lowerLipSet.has(i) ? 1 : 0));
    const cornerLField = diffuseMask(n478, nbrs, (i) => (cornerLipSet.has(i) ? sideSign(i, true) : 0));
    const cornerRField = diffuseMask(n478, nbrs, (i) => (cornerLipSet.has(i) ? sideSign(i, false) : 0));
    const browField = diffuseMask(n478, nbrs, (i) => (browSet.has(i) ? 1 : 0));
    const cheekField = diffuseMask(n478, nbrs, (i) => (cheekSet.has(i) ? 1 : 0));
    const noseField = diffuseMask(n478, nbrs, (i) => (noseWingSet.has(i) ? 1 : 0));

    const wrapEyes = (arr) => {
        if (rig.rightEyeSocket) {
            clampEyelidsOverEyeball(arr, rig.rightEyeSocket, rig.rightEyeWrapIds);
        }
        if (rig.leftEyeSocket) {
            clampEyelidsOverEyeball(arr, rig.leftEyeSocket, rig.leftEyeWrapIds);
        }
    };

    const build478 = (fn) => {
        const arr = new Float32Array(rest478.subarray(0, n478 * 3));
        fn(arr);
        return arr;
    };

    const targetMap = {
        eyeBlinkLeft: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3 + 1] -= rig.eyeLeft.upperWeights[i] * rig.eyeLeft.lidGap * 1.02;
                a[i * 3 + 1] += rig.eyeLeft.lowerWeights[i] * rig.eyeLeft.lidGap * 0.26;
            }
            wrapEyes(a);
        }),
        eyeBlinkRight: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3 + 1] -= rig.eyeRight.upperWeights[i] * rig.eyeRight.lidGap * 1.02;
                a[i * 3 + 1] += rig.eyeRight.lowerWeights[i] * rig.eyeRight.lidGap * 0.26;
            }
            wrapEyes(a);
        }),
        eyeWideLeft: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3 + 1] += rig.eyeLeft.upperWeights[i] * rig.eyeLeft.lidGap * 0.45;
            }
            wrapEyes(a);
        }),
        eyeWideRight: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3 + 1] += rig.eyeRight.upperWeights[i] * rig.eyeRight.lidGap * 0.45;
            }
            wrapEyes(a);
        }),
        eyeSquintLeft: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3 + 1] += rig.eyeLeft.lowerWeights[i] * rig.eyeLeft.lidGap * 0.42;
                a[i * 3 + 1] += cheekField[i] * sideSign(i, true) * 0.04 * chinDrop;
            }
            wrapEyes(a);
        }),
        eyeSquintRight: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3 + 1] += rig.eyeRight.lowerWeights[i] * rig.eyeRight.lidGap * 0.42;
                a[i * 3 + 1] += cheekField[i] * sideSign(i, false) * 0.04 * chinDrop;
            }
            wrapEyes(a);
        }),
        eyeLookDownLeft: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] -= rig.eyeLeft.upperWeights[i] * rig.eyeLeft.lidGap * 0.30;
            wrapEyes(a);
        }),
        eyeLookUpLeft: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] += rig.eyeLeft.upperWeights[i] * rig.eyeLeft.lidGap * 0.22;
            wrapEyes(a);
        }),
        eyeLookInLeft: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3] -= rig.eyeLeft.upperWeights[i] * 0.012 * eyeSpan;
        }),
        eyeLookOutLeft: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3] += rig.eyeLeft.upperWeights[i] * 0.012 * eyeSpan;
        }),
        eyeLookDownRight: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] -= rig.eyeRight.upperWeights[i] * rig.eyeRight.lidGap * 0.30;
            wrapEyes(a);
        }),
        eyeLookUpRight: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] += rig.eyeRight.upperWeights[i] * rig.eyeRight.lidGap * 0.22;
            wrapEyes(a);
        }),
        eyeLookInRight: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3] += rig.eyeRight.upperWeights[i] * 0.012 * eyeSpan;
        }),
        eyeLookOutRight: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3] -= rig.eyeRight.upperWeights[i] * 0.012 * eyeSpan;
        }),
        jawOpen: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3 + 1] -= jawField[i] * 0.26 * chinDrop;
                a[i * 3 + 2] -= jawField[i] * 0.05 * eyeSpan;
            }
        }),
        jawForward: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 2] += jawField[i] * 0.08 * eyeSpan;
        }),
        jawLeft: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3] += jawField[i] * 0.07 * eyeSpan;
        }),
        jawRight: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3] -= jawField[i] * 0.07 * eyeSpan;
        }),
        mouthClose: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] += lowerLipField[i] * 0.04 * chinDrop;
        }),
        mouthFunnel: build478((a) => {
            const { mouthHalfW, inwardX, archY, protrudeZ } = rig.lipRound;
            for (let i = 0; i < n478; i++) {
                a[i * 3] += inwardX[i] * mouthHalfW * 0.26;
                a[i * 3 + 1] += archY[i] * mouthHalfW * 0.16;
                a[i * 3 + 2] += protrudeZ[i] * eyeSpan * 0.024;
            }
        }),
        mouthPucker: build478((a) => {
            const { mouthHalfW, inwardX, protrudeZ } = rig.lipRound;
            for (let i = 0; i < n478; i++) {
                a[i * 3] += inwardX[i] * mouthHalfW * 0.34;
                a[i * 3 + 2] += protrudeZ[i] * eyeSpan * 0.036;
            }
        }),
        mouthLeft: build478((a) => {
            for (let i = 0; i < n478; i++) {
                const w = Math.max(upperLipField[i], lowerLipField[i], cornerLField[i], cornerRField[i]);
                a[i * 3] += w * 0.07 * eyeSpan;
            }
        }),
        mouthRight: build478((a) => {
            for (let i = 0; i < n478; i++) {
                const w = Math.max(upperLipField[i], lowerLipField[i], cornerLField[i], cornerRField[i]);
                a[i * 3] -= w * 0.07 * eyeSpan;
            }
        }),
        mouthSmileLeft: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3] += cornerLField[i] * 0.065 * eyeSpan;
                a[i * 3 + 1] += cornerLField[i] * 0.075 * chinDrop;
                a[i * 3 + 2] -= cornerLField[i] * 0.025 * eyeSpan;
            }
        }),
        mouthSmileRight: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3] -= cornerRField[i] * 0.065 * eyeSpan;
                a[i * 3 + 1] += cornerRField[i] * 0.075 * chinDrop;
                a[i * 3 + 2] -= cornerRField[i] * 0.025 * eyeSpan;
            }
        }),
        mouthFrownLeft: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] -= cornerLField[i] * 0.06 * chinDrop;
        }),
        mouthFrownRight: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] -= cornerRField[i] * 0.06 * chinDrop;
        }),
        mouthDimpleLeft: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3] += cornerLField[i] * 0.035 * eyeSpan;
                a[i * 3 + 2] -= cornerLField[i] * 0.02 * eyeSpan;
            }
        }),
        mouthDimpleRight: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3] -= cornerRField[i] * 0.035 * eyeSpan;
                a[i * 3 + 2] -= cornerRField[i] * 0.02 * eyeSpan;
            }
        }),
        mouthStretchLeft: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3] += cornerLField[i] * 0.075 * eyeSpan;
        }),
        mouthStretchRight: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3] -= cornerRField[i] * 0.075 * eyeSpan;
        }),
        mouthRollLower: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3 + 1] += lowerLipField[i] * 0.025 * chinDrop;
                a[i * 3 + 2] -= lowerLipField[i] * 0.03 * eyeSpan;
            }
        }),
        mouthRollUpper: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3 + 1] -= upperLipField[i] * 0.025 * chinDrop;
                a[i * 3 + 2] -= upperLipField[i] * 0.03 * eyeSpan;
            }
        }),
        mouthShrugLower: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3 + 1] += lowerLipField[i] * 0.045 * chinDrop;
                a[i * 3 + 2] += lowerLipField[i] * 0.02 * eyeSpan;
            }
        }),
        mouthShrugUpper: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3 + 1] += upperLipField[i] * 0.04 * chinDrop;
                a[i * 3 + 2] += upperLipField[i] * 0.015 * eyeSpan;
            }
        }),
        mouthPressLeft: build478((a) => {
            for (let i = 0; i < n478; i++) {
                const s = sideSign(i, true);
                a[i * 3 + 1] -= upperLipField[i] * s * 0.018 * chinDrop;
                a[i * 3 + 1] += lowerLipField[i] * s * 0.018 * chinDrop;
            }
        }),
        mouthPressRight: build478((a) => {
            for (let i = 0; i < n478; i++) {
                const s = sideSign(i, false);
                a[i * 3 + 1] -= upperLipField[i] * s * 0.018 * chinDrop;
                a[i * 3 + 1] += lowerLipField[i] * s * 0.018 * chinDrop;
            }
        }),
        mouthLowerDownLeft: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] -= lowerLipField[i] * sideSign(i, true) * 0.065 * chinDrop;
        }),
        mouthLowerDownRight: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] -= lowerLipField[i] * sideSign(i, false) * 0.065 * chinDrop;
        }),
        mouthUpperUpLeft: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] += upperLipField[i] * sideSign(i, true) * 0.055 * chinDrop;
        }),
        mouthUpperUpRight: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] += upperLipField[i] * sideSign(i, false) * 0.055 * chinDrop;
        }),
        cheekPuff: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3] += Math.sign(rest478[i * 3]) * cheekField[i] * 0.05 * eyeSpan;
                a[i * 3 + 2] += cheekField[i] * 0.06 * eyeSpan;
            }
        }),
        cheekSquintLeft: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] += cheekField[i] * sideSign(i, true) * 0.05 * chinDrop;
        }),
        cheekSquintRight: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] += cheekField[i] * sideSign(i, false) * 0.05 * chinDrop;
        }),
        noseSneerLeft: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] += noseField[i] * sideSign(i, true) * 0.045 * chinDrop;
        }),
        noseSneerRight: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] += noseField[i] * sideSign(i, false) * 0.045 * chinDrop;
        }),
        browDownLeft: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] -= browField[i] * sideSign(i, true) * 0.055 * chinDrop;
        }),
        browDownRight: build478((a) => {
            for (let i = 0; i < n478; i++) a[i * 3 + 1] -= browField[i] * sideSign(i, false) * 0.055 * chinDrop;
        }),
        browInnerUp: build478((a) => {
            for (let i = 0; i < n478; i++) {
                const inner = 1.0 - smoothstep(0.08 * eyeSpan, 0.35 * eyeSpan, Math.abs(rest478[i * 3]));
                a[i * 3 + 1] += browField[i] * inner * 0.06 * chinDrop;
            }
        }),
        browOuterUpLeft: build478((a) => {
            for (let i = 0; i < n478; i++) {
                const outer = smoothstep(0.18 * eyeSpan, 0.42 * eyeSpan, rest478[i * 3]);
                a[i * 3 + 1] += browField[i] * outer * 0.055 * chinDrop;
            }
        }),
        browOuterUpRight: build478((a) => {
            for (let i = 0; i < n478; i++) {
                const outer = smoothstep(-0.18 * eyeSpan, -0.42 * eyeSpan, rest478[i * 3]);
                a[i * 3 + 1] += browField[i] * outer * 0.055 * chinDrop;
            }
        }),
        tongueOut: build478((a) => {
            for (let i = 0; i < n478; i++) {
                a[i * 3 + 1] -= lowerLipField[i] * 0.04 * chinDrop;
                a[i * 3 + 2] += lowerLipField[i] * 0.02 * eyeSpan;
            }
        }),
    };

    const targets = [];
    for (const name of ARKIT_52_NAMES) {
        const expanded = expandFacePositionsToTopology(faceMesh, targetMap[name], topo);
        const deltas = new Float32Array(expanded.length);
        for (let i = 0; i < expanded.length; i++) {
            deltas[i] = expanded[i] - neutralExpanded[i];
        }
        targets.push({ name, deltas });
    }
    return { neutralPositions: neutralExpanded, targets };
}

function gazeToQuaternion(gx, gy) {
    // Match EyeballRig pitch (around X: sinP = -gy) then yaw (around Y: sinY = gx)
    const yaw = Math.asin(Math.max(-0.30, Math.min(0.30, gx)));
    const pitch = -Math.asin(Math.max(-0.20, Math.min(0.20, gy)));
    const cy = Math.cos(yaw * 0.5);
    const sy = Math.sin(yaw * 0.5);
    const cp = Math.cos(pitch * 0.5);
    const sp = Math.sin(pitch * 0.5);
    // qY * qX
    return [
        cy * sp,
        sy * cp,
        -sy * sp,
        cy * cp,
    ];
}

function packGlb(gltfJson, binParts) {
    let binLen = 0;
    for (const p of binParts) binLen += p.byteLength;
    const binPadLen = (binLen + 3) & ~3;
    const binBuf = new Uint8Array(binPadLen);
    let off = 0;
    for (const p of binParts) {
        binBuf.set(new Uint8Array(p.buffer, p.byteOffset, p.byteLength), off);
        off += p.byteLength;
    }

    const jsonBytes = new TextEncoder().encode(JSON.stringify(gltfJson));
    const jsonPadLen = (jsonBytes.length + 3) & ~3;
    const totalLen = 12 + 8 + jsonPadLen + 8 + binPadLen;
    const out = new Uint8Array(totalLen);
    const view = new DataView(out.buffer);

    // Header
    view.setUint32(0, 0x46546C67, true); // glTF
    view.setUint32(4, 2, true);
    view.setUint32(8, totalLen, true);

    // JSON chunk
    view.setUint32(12, jsonPadLen, true);
    view.setUint32(16, 0x4E4F534A, true); // JSON
    out.set(jsonBytes, 20);
    for (let i = jsonBytes.length; i < jsonPadLen; i++) out[20 + i] = 0x20;

    // BIN chunk
    const binHeaderOff = 20 + jsonPadLen;
    view.setUint32(binHeaderOff, binPadLen, true);
    view.setUint32(binHeaderOff + 4, 0x004E4942, true); // BIN\0
    out.set(binBuf, binHeaderOff + 8);
    return out;
}

/**
 * Build a self-contained binary .glb (glTF 2.0) buffer for Static Mesh,
 * Mesh + 52 ARKit Blendshapes, or Mesh + Mocap Animation.
 */
export async function exportFaceGlbBuffer(faceMesh, options = {}) {
    if (!faceMesh?.hasMesh || !faceMesh.restPositions) {
        throw new Error("No 3D face mesh is loaded to export.");
    }
    const {
        mode = "static",          // "static" | "blendshapes" | "animation"
        scope = "head",           // "head" | "face"
        subdiv = "smooth",        // "smooth" | "base"
        pose = "rest",            // "rest" | "current"
        includeEyeballs = true,
        animMode = "arkit",       // "arkit" | "exact"
        motionClip = [],
        clipFps = 24,
    } = options;

    if (mode === "animation" && (!Array.isArray(motionClip) || motionClip.length === 0)) {
        throw new Error("No mocap clip recorded or loaded to export.");
    }

    const topo = extractExportTopology(faceMesh, { scope, subdiv });
    const binParts = [];
    const bufferViews = [];
    const accessors = [];
    let binOffset = 0;

    const pushBin = (typedArr, target = undefined) => {
        const byteOffset = binOffset;
        const pad = (4 - (typedArr.byteLength % 4)) % 4;
        const rawBytes = new Uint8Array(typedArr.byteLength + pad);
        rawBytes.set(new Uint8Array(typedArr.buffer, typedArr.byteOffset, typedArr.byteLength));
        binParts.push(rawBytes);
        binOffset += rawBytes.byteLength;
        const bv = { buffer: 0, byteOffset, byteLength: typedArr.byteLength };
        if (target !== undefined) bv.target = target;
        const idx = bufferViews.length;
        bufferViews.push(bv);
        return idx;
    };

    const pushAccessor = (bvIdx, componentType, count, type, bounds = null) => {
        const acc = { bufferView: bvIdx, componentType, count, type };
        if (bounds) {
            acc.min = bounds.min;
            acc.max = bounds.max;
        }
        const idx = accessors.length;
        accessors.push(acc);
        return idx;
    };

    // 1. Base neutral or current-pose face positions
    let facePos478 = new Float32Array(faceMesh.restPositions.subarray(0, 478 * 3));
    if (mode === "static" && pose === "current" && faceMesh.latestActivePositions) {
        facePos478 = new Float32Array(faceMesh.latestActivePositions.subarray(0, 478 * 3));
    }
    const basePositions = expandFacePositionsToTopology(faceMesh, facePos478, topo);
    const baseNormals = computeNormals(basePositions, topo.indices);

    // 2. UVs (v_gltf = 1 - v_webgl, pure shell vertices mapped to top-left 12x12 skin patch)
    const activeWebglUvs = faceMesh.getActiveDisplayUvs();
    const uvs = new Float32Array(topo.vertCount * 2);
    const patchCenter = 6.0 / 1024.0;
    for (let i = 0; i < topo.vertCount; i++) {
        if (topo.pureShellMask[i]) {
            uvs[i * 2] = patchCenter;
            uvs[i * 2 + 1] = patchCenter;
        } else {
            const srcIdx = topo.remapIndices ? topo.remapIndices[i] : i;
            uvs[i * 2] = activeWebglUvs[srcIdx * 2];
            uvs[i * 2 + 1] = 1.0 - activeWebglUvs[srcIdx * 2 + 1];
        }
    }

    const posBv = pushBin(basePositions, 34962);
    const normBv = pushBin(baseNormals, 34962);
    const uvBv = pushBin(uvs, 34962);
    const idxBv = pushBin(topo.indices, 34963);

    const posAcc = pushAccessor(posBv, 5126, topo.vertCount, "VEC3", computeVec3MinMax(basePositions));
    const normAcc = pushAccessor(normBv, 5126, topo.vertCount, "VEC3");
    const uvAcc = pushAccessor(uvBv, 5126, topo.vertCount, "VEC2");
    const idxAcc = pushAccessor(idxBv, 5123, topo.indices.length, "SCALAR", {
        min: [0],
        max: [topo.vertCount - 1],
    });

    // 3. Textures & Materials
    const facePng = await bakeFaceTexturePngBytes(faceMesh, 1024);
    const faceImgBv = pushBin(facePng);
    const images = [{ bufferView: faceImgBv, mimeType: "image/png" }];
    const samplers = [{ magFilter: 9729, minFilter: 9987, wrapS: 33071, wrapT: 33071 }];
    const textures = [{ sampler: 0, source: 0 }];
    const materials = [{
        name: "Face_Mat",
        pbrMetallicRoughness: {
            baseColorTexture: { index: 0 },
            metallicFactor: 0.0,
            roughnessFactor: 0.72,
        },
        doubleSided: !topo.isHead,
    }];

    const primitive = {
        attributes: { POSITION: posAcc, NORMAL: normAcc, TEXCOORD_0: uvAcc },
        indices: idxAcc,
        material: 0,
    };

    // 4. Morph targets (Blendshapes or Animation)
    let targetNames = null;
    if (mode === "blendshapes" || (mode === "animation" && animMode === "arkit")) {
        const arkit = buildArkit52MorphTargets(faceMesh, topo);
        targetNames = arkit.targets.map((t) => t.name);
        primitive.targets = arkit.targets.map((t) => {
            const bv = pushBin(t.deltas, 34962);
            const acc = pushAccessor(bv, 5126, topo.vertCount, "VEC3", computeVec3MinMax(t.deltas));
            return { POSITION: acc };
        });
        primitive.extras = { targetNames };
    } else if (mode === "animation" && animMode === "exact") {
        const estBytes = motionClip.length * topo.vertCount * 12;
        if (estBytes > 150 * 1024 * 1024) {
            throw new Error(
                `Exact Bake (${motionClip.length}f × ${topo.vertCount} verts) exceeds 150 MB. Switch to 'Base' subdiv or 'ARKit 52' mode.`
            );
        }
        const rig = faceMesh.morphRig || new FaceRetargetRig(facePos478, faceMesh.baseIndices, faceMesh.regionWeights);
        const scratch = new Float32Array(478 * 3);
        targetNames = [];
        primitive.targets = [];
        for (let f = 0; f < motionClip.length; f++) {
            rig.evaluate(motionClip[f], scratch);
            const expanded = expandFacePositionsToTopology(faceMesh, scratch, topo);
            const deltas = new Float32Array(expanded.length);
            for (let i = 0; i < expanded.length; i++) {
                deltas[i] = expanded[i] - basePositions[i];
            }
            const bv = pushBin(deltas, 34962);
            const acc = pushAccessor(bv, 5126, topo.vertCount, "VEC3", computeVec3MinMax(deltas));
            primitive.targets.push({ POSITION: acc });
            targetNames.push(`frame_${String(f).padStart(4, "0")}`);
        }
        primitive.extras = { targetNames };
    }

    const headMeshDef = { name: topo.isHead ? "Face_Head_Mesh" : "Face_Only_Mesh", primitives: [primitive] };
    if (targetNames) {
        headMeshDef.extras = { targetNames };
        headMeshDef.weights = new Array(targetNames.length).fill(0.0);
    }

    const meshes = [headMeshDef];
    const nodes = [];
    const rootChildren = [1];
    const eyeNames = ["Eye_R", "Eye_L"];
    const eyeNodeIndices = [];

    // Node 0: Face3D_Root (carries head rotation during animation or current pose)
    const rootNode = { name: "Face3D_Root", children: rootChildren };
    if (mode === "static" && pose === "current" && faceMesh.group?.quaternion) {
        const q = faceMesh.group.quaternion;
        if (Math.abs(q.x) + Math.abs(q.y) + Math.abs(q.z) > 1e-6) {
            rootNode.rotation = [q.x, q.y, q.z, q.w];
        }
    }
    nodes.push(rootNode);
    // Node 1: Face mesh
    nodes.push({ name: headMeshDef.name, mesh: 0 });

    // 5. Optional Eyeball Child Nodes (pivoted at socket centers)
    const eyeSockets = [];
    if (includeEyeballs && faceMesh.eyeballRig?.eyes?.length === 2) {
        for (let eIdx = 0; eIdx < 2; eIdx++) {
            const eye = faceMesh.eyeballRig.eyes[eIdx];
            const socket = fitEyeSocket(facePos478, EYE_SPECS[eIdx]);
            eyeSockets.push(socket);
            const eyeVertCount = Math.floor(eye.pos.length / 3);
            const localPos = new Float32Array(eye.pos.length);
            for (let i = 0; i < eyeVertCount; i++) {
                localPos[i * 3] = eye.pos[i * 3] - socket.cx;
                localPos[i * 3 + 1] = eye.pos[i * 3 + 1] - socket.cy;
                localPos[i * 3 + 2] = eye.pos[i * 3 + 2] - socket.cz;
            }
            const eyeUvs = new Float32Array(eye.uv.length);
            for (let i = 0; i < eyeVertCount; i++) {
                eyeUvs[i * 2] = eye.uv[i * 2];
                eyeUvs[i * 2 + 1] = 1.0 - eye.uv[i * 2 + 1];
            }
            const eyeIdxArr = new Uint16Array(faceMesh.eyeballRig.topo.indices);

            const ePosBv = pushBin(localPos, 34962);
            const eNormBv = pushBin(new Float32Array(eye.norm), 34962);
            const eUvBv = pushBin(eyeUvs, 34962);
            const eIdxBv = pushBin(eyeIdxArr, 34963);

            const ePosAcc = pushAccessor(ePosBv, 5126, eyeVertCount, "VEC3", computeVec3MinMax(localPos));
            const eNormAcc = pushAccessor(eNormBv, 5126, eyeVertCount, "VEC3");
            const eUvAcc = pushAccessor(eUvBv, 5126, eyeVertCount, "VEC2");
            const eIdxAcc = pushAccessor(eIdxBv, 5123, eyeIdxArr.length, "SCALAR", {
                min: [0],
                max: [eyeVertCount - 1],
            });

            const eyePng = await bakeEyeballTexturePngBytes(faceMesh, eye, 256);
            const eyeImgBv = pushBin(eyePng);
            const imgIdx = images.length;
            images.push({ bufferView: eyeImgBv, mimeType: "image/png" });
            const texIdx = textures.length;
            textures.push({ sampler: 0, source: imgIdx });
            const matIdx = materials.length;
            materials.push({
                name: `${eyeNames[eIdx]}_Mat`,
                pbrMetallicRoughness: {
                    baseColorTexture: { index: texIdx },
                    metallicFactor: 0.0,
                    roughnessFactor: 0.25,
                },
            });

            const meshIdx = meshes.length;
            meshes.push({
                name: `${eyeNames[eIdx]}_Mesh`,
                primitives: [{
                    attributes: { POSITION: ePosAcc, NORMAL: eNormAcc, TEXCOORD_0: eUvAcc },
                    indices: eIdxAcc,
                    material: matIdx,
                }],
            });

            const nodeIdx = nodes.length;
            nodes.push({
                name: eyeNames[eIdx],
                mesh: meshIdx,
                translation: [socket.cx, socket.cy, socket.cz],
            });
            rootChildren.push(nodeIdx);
            eyeNodeIndices.push(nodeIdx);
        }
    }

    // 6. Mocap Animation Tracks
    const animations = [];
    if (mode === "animation") {
        const nFrames = motionClip.length;
        const fps = Math.max(1, Number(clipFps || 24));
        const times = new Float32Array(nFrames);
        const rootQuats = new Float32Array(nFrames * 4);
        for (let f = 0; f < nFrames; f++) {
            times[f] = f / fps;
            const q = motionClip[f]?.quat || [0, 0, 0, 1];
            rootQuats[f * 4] = q[0];
            rootQuats[f * 4 + 1] = q[1];
            rootQuats[f * 4 + 2] = q[2];
            rootQuats[f * 4 + 3] = q[3];
        }

        const timeBv = pushBin(times);
        const timeAcc = pushAccessor(timeBv, 5126, nFrames, "SCALAR", {
            min: [times[0]],
            max: [times[nFrames - 1]],
        });
        const quatBv = pushBin(rootQuats);
        const quatAcc = pushAccessor(quatBv, 5126, nFrames, "VEC4");

        const samplersAnim = [{ input: timeAcc, interpolation: "LINEAR", output: quatAcc }];
        const channels = [{ sampler: 0, target: { node: 0, path: "rotation" } }];

        const nTargets = targetNames.length;
        const weights = new Float32Array(nFrames * nTargets);
        const rw = faceMesh.regionWeights || { eyes: 1.0, brows: 1.0, mouth: 1.0, jaw: 0.85, cheeks: 0.42 };

        if (animMode === "arkit") {
            const scaleForName = (name) => {
                if (name.startsWith("eye")) return Number(rw.eyes ?? 1.0);
                if (name.startsWith("brow")) return Number(rw.brows ?? 1.0);
                if (name.startsWith("jaw")) return Number(rw.jaw ?? 0.85);
                if (name.startsWith("cheek") || name.startsWith("nose")) return Number(rw.cheeks ?? 0.42);
                return Number(rw.mouth ?? 1.0);
            };
            for (let f = 0; f < nFrames; f++) {
                const bs = motionClip[f]?.bs || {};
                const base = f * nTargets;
                for (let t = 0; t < nTargets; t++) {
                    const name = targetNames[t];
                    weights[base + t] = Math.max(0, Math.min(1.2, (bs[name] || 0) * scaleForName(name)));
                }
            }
        } else {
            for (let f = 0; f < nFrames; f++) {
                weights[f * nTargets + f] = 1.0;
            }
        }

        const weightBv = pushBin(weights);
        const weightAcc = pushAccessor(weightBv, 5126, nFrames * nTargets, "SCALAR");
        const wSamplerIdx = samplersAnim.length;
        samplersAnim.push({ input: timeAcc, interpolation: "LINEAR", output: weightAcc });
        channels.push({ sampler: wSamplerIdx, target: { node: 1, path: "weights" } });

        // Eye gaze rotation channels when eyeballs are included
        if (eyeNodeIndices.length === 2 && faceMesh.eyeballRig?.eyes?.length === 2) {
            const eyeQuatL = new Float32Array(nFrames * 4);
            const eyeQuatR = new Float32Array(nFrames * 4);
            for (let f = 0; f < nFrames; f++) {
                const mf = motionClip[f];
                const d = mf?.deltas;
                let gx = 0;
                let gy = 0;
                if (d && d.length >= 474 * 3) {
                    const sgn = mf.mirror ? -1.0 : 1.0;
                    const gxR = sgn * d[468 * 3] * 1.4;
                    const gyR = d[468 * 3 + 1] * 1.4;
                    const gxL = sgn * d[473 * 3] * 1.4;
                    const gyL = d[473 * 3 + 1] * 1.4;
                    gx = 0.5 * (gxR + gxL);
                    gy = 0.5 * (gyR + gyL);
                }
                const q = gazeToQuaternion(gx, gy);
                eyeQuatR.set(q, f * 4);
                eyeQuatL.set(q, f * 4);
            }
            const rBv = pushBin(eyeQuatR);
            const rAcc = pushAccessor(rBv, 5126, nFrames, "VEC4");
            const rSamp = samplersAnim.length;
            samplersAnim.push({ input: timeAcc, interpolation: "LINEAR", output: rAcc });
            channels.push({ sampler: rSamp, target: { node: eyeNodeIndices[0], path: "rotation" } });

            const lBv = pushBin(eyeQuatL);
            const lAcc = pushAccessor(lBv, 5126, nFrames, "VEC4");
            const lSamp = samplersAnim.length;
            samplersAnim.push({ input: timeAcc, interpolation: "LINEAR", output: lAcc });
            channels.push({ sampler: lSamp, target: { node: eyeNodeIndices[1], path: "rotation" } });
        }

        animations.push({
            name: "Face3D_Mocap",
            samplers: samplersAnim,
            channels,
        });
    }

    const gltf = {
        asset: { version: "2.0", generator: "ComfyUI-Yedp-Face3D" },
        scene: 0,
        scenes: [{ name: "Scene", nodes: [0] }],
        nodes,
        meshes,
        materials,
        textures,
        images,
        samplers,
        bufferViews,
        accessors,
        buffers: [{ byteLength: binOffset }],
    };
    if (animations.length > 0) {
        gltf.animations = animations;
    }

    return packGlb(gltf, binParts);
}
