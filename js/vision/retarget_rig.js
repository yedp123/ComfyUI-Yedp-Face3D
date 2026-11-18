/**
 * Continuous 478-vertex intrinsic skull-space retargeting rig with:
 * - Topological group-isolated bilateral symmetry (prevents upper/lower lip cross-pairing)
 * - Graph Laplacian diffused regional weight blending (Eyes, Brows, Mouth, Jaw, Cheeks)
 * - Camera-angle foreshortening damping + 1-pass mesh delta smoothing
 * - EAR-fused eyelid blink closure
 */

import { EYE_SPECS, fitEyeSocket, clampEyelidsOverEyeball } from "../viewport/eyeball_rig.js";

const NOSE_BRIDGE_IDX = 168;
const CHIN_IDX = 152;
const LEFT_EYE_OUTER_IDX = 33;
const LEFT_EYE_INNER_IDX = 133;
const RIGHT_EYE_OUTER_IDX = 263;
const RIGHT_EYE_INNER_IDX = 362;

const RIGHT_UPPER_LID = [246, 161, 160, 159, 158, 157, 173];
const RIGHT_UPPER_FOLD = [27, 28, 29, 30, 56, 190, 247];
const RIGHT_LOWER_LID = [7, 163, 144, 145, 153, 154, 155];

const LEFT_UPPER_LID = [466, 388, 387, 386, 385, 384, 398];
const LEFT_UPPER_FOLD = [257, 258, 259, 260, 286, 414, 467];
const LEFT_LOWER_LID = [249, 390, 373, 374, 380, 381, 382];

// Complete lip partitions (outer, mid-vermilion, and inner loop) from MediaPipe topology
const UPPER_LIP_SET = new Set([
    0, 11, 12, 13, 37, 38, 39, 40, 41, 42, 62, 72, 73, 74, 76, 77,
    80, 81, 82, 183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 292,
    302, 303, 304, 306, 307, 310, 311, 312, 407, 408, 409, 415,
]);

const LOWER_LIP_SET = new Set([
    14, 15, 16, 17, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96, 146,
    178, 179, 180, 181, 314, 315, 316, 317, 318, 319, 320, 321, 324, 325,
    375, 402, 403, 404, 405,
]);

const CORNER_LIP_SET = new Set([61, 78, 291, 308]);
const LIP_VERTEX_SET = new Set([...UPPER_LIP_SET, ...LOWER_LIP_SET, ...CORNER_LIP_SET]);

const BROW_SET = new Set([
    46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
    276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
]);

const JAW_SET = new Set([
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    377, 400, 378, 379, 365, 397, 288, 361,
    175, 199, 200, 201, 421, 418, 262, 428, 32, 140, 171, 396, 369,
]);

const UPPER_EYE_SET = new Set([
    ...RIGHT_UPPER_LID, ...RIGHT_UPPER_FOLD,
    ...LEFT_UPPER_LID, ...LEFT_UPPER_FOLD,
]);
const LOWER_EYE_SET = new Set([...RIGHT_LOWER_LID, ...LEFT_LOWER_LID]);
const PUPIL_SET = new Set([468, 469, 470, 471, 472, 473, 474, 475, 476, 477]);

// Rigid nose ridge landmarks stay anchored to preserve character identity
const NOSE_BONE_INDICES = new Set([168, 6, 197, 195, 5, 4, 1]);

function smoothstep(edge0, edge1, x) {
    const t = Math.max(0, Math.min(1, (x - edge0) / Math.max(edge1 - edge0, 1e-6)));
    return t * t * (3 - 2 * t);
}

function midPt(pos, a, b) {
    return [
        0.5 * (pos[a * 3] + pos[b * 3]),
        0.5 * (pos[a * 3 + 1] + pos[b * 3 + 1]),
        0.5 * (pos[a * 3 + 2] + pos[b * 3 + 2]),
    ];
}

function getSymmetryGroup(idx) {
    if (UPPER_LIP_SET.has(idx)) return 1;
    if (LOWER_LIP_SET.has(idx)) return 2;
    if (CORNER_LIP_SET.has(idx)) return 3;
    if (UPPER_EYE_SET.has(idx)) return 4;
    if (LOWER_EYE_SET.has(idx)) return 5;
    if (PUPIL_SET.has(idx)) return 6;
    return 0;
}

function buildSymmetryMap(rest) {
    const n = Math.floor(rest.length / 3);
    const map = new Int32Array(n);
    const groups = new Int8Array(n);
    for (let i = 0; i < n; i++) groups[i] = getSymmetryGroup(i);

    for (let i = 0; i < n; i++) {
        const mx = -rest[i * 3];
        const my = rest[i * 3 + 1];
        const mz = rest[i * 3 + 2];
        const g = groups[i];
        let bestIdx = i;
        let bestDist = Infinity;
        for (let j = 0; j < n; j++) {
            // Strictly match within the same anatomical loop so upper lip never mirrors to lower lip
            if (groups[j] !== g) continue;
            const dx = mx - rest[j * 3];
            const dy = my - rest[j * 3 + 1];
            const dz = mz - rest[j * 3 + 2];
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < bestDist) {
                bestDist = d2;
                bestIdx = j;
            }
        }
        map[i] = bestIdx;
    }
    return map;
}

function buildAdjacency(nVerts, indices) {
    const adj = Array.from({ length: nVerts }, () => new Set());
    if (indices && indices.length >= 3) {
        for (let i = 0; i + 2 < indices.length; i += 3) {
            const a = indices[i];
            const b = indices[i + 1];
            const c = indices[i + 2];
            if (a < nVerts && b < nVerts && c < nVerts) {
                adj[a].add(b).add(c);
                adj[b].add(a).add(c);
                adj[c].add(a).add(b);
            }
        }
    }
    return adj.map((s) => Array.from(s));
}

function buildEyeRig(pos, upperIds, upperFoldIds, lowerIds, outerIdx, innerIdx) {
    const upperSet = new Set(upperIds);
    const upperFoldSet = new Set(upperFoldIds);
    const lowerSet = new Set(lowerIds);

    const ox = pos[outerIdx * 3];
    const oy = pos[outerIdx * 3 + 1];
    const ix = pos[innerIdx * 3];
    const iy = pos[innerIdx * 3 + 1];
    const cx = 0.5 * (ox + ix);
    const cy = 0.5 * (oy + iy);
    const halfW = Math.max(Math.abs(ox - ix) * 0.58, 0.03);

    let uy = 0;
    let ly = 0;
    for (const id of upperIds) uy += pos[id * 3 + 1];
    for (const id of lowerIds) ly += pos[id * 3 + 1];
    uy /= upperIds.length;
    ly /= lowerIds.length;
    const lidGap = Math.max(uy - ly, 0.016);

    const n = pos.length / 3;
    const upperWeights = new Float32Array(n);
    const lowerWeights = new Float32Array(n);
    const isLidMask = new Uint8Array(n);

    for (let i = 0; i < n; i++) {
        // Keep medial/lateral eye corners anchored during blink weights (deltas still track expressions)
        if (i === outerIdx || i === innerIdx) continue;
        const vx = pos[i * 3];
        const vy = pos[i * 3 + 1];
        const dx = Math.abs(vx - cx) / halfW;
        if (dx > 1.4) continue;

        const horiz = Math.exp(-1.25 * dx * dx);
        if (upperSet.has(i)) {
            upperWeights[i] = Math.max(horiz, 0.45);
            isLidMask[i] = 1;
        } else if (upperFoldSet.has(i)) {
            upperWeights[i] = horiz * 0.78;
            isLidMask[i] = 1;
        } else if (lowerSet.has(i)) {
            lowerWeights[i] = Math.max(horiz, 0.40);
            isLidMask[i] = 1;
        } else if (vy >= cy - lidGap * 0.2 && vy <= cy + lidGap * 2.2) {
            const dy = Math.abs(vy - uy) / (lidGap * 1.8);
            upperWeights[i] = horiz * Math.exp(-2.2 * dy * dy) * 0.45;
        }
    }

    return { upperWeights, lowerWeights, lidGap, isLidMask };
}

/**
 * Precompute Graph-Laplacian-diffused lip shaping fields for rounded "O" (mouthFunnel)
 * and kissing pucker (mouthPucker) expressions without hard boundary seams.
 */
function buildLipRoundRig(pos, neighbors) {
    const n = Math.floor(pos.length / 3);
    const mouthCenterX = n > 291 ? 0.5 * (pos[61 * 3] + pos[291 * 3]) : 0;
    const mouthCenterY = n > 14 ? 0.5 * (pos[13 * 3 + 1] + pos[14 * 3 + 1]) : 0;
    const mouthHalfW = n > 291 ? Math.max(0.5 * Math.abs(pos[291 * 3] - pos[61 * 3]), 0.03) : 0.05;

    const inwardX = new Float32Array(n);
    const archY = new Float32Array(n);
    const protrudeZ = new Float32Array(n);
    const locked = new Uint8Array(n);

    for (let i = 0; i < n; i++) {
        if (NOSE_BONE_INDICES.has(i)) {
            locked[i] = 1;
            continue;
        }
        if (!LIP_VERTEX_SET.has(i)) continue;
        locked[i] = 2;

        const relX = (pos[i * 3] - mouthCenterX) / mouthHalfW;
        const u = Math.min(1.0, Math.abs(relX));
        // Pull outer corners strongly inward while keeping midline x=0 anchored
        inwardX[i] = -Math.sign(relX || 0) * Math.pow(u, 0.9);
        protrudeZ[i] = Math.cos(0.32 * Math.PI * u);

        // Arch lip shoulders (~u = 0.4..0.85) into a smooth circle on mouthFunnel
        const shoulder = Math.sin(Math.PI * Math.min(u, 0.96)) * (0.45 + 0.55 * u);
        if (UPPER_LIP_SET.has(i)) {
            archY[i] = shoulder;
        } else if (LOWER_LIP_SET.has(i)) {
            archY[i] = -shoulder;
        } else {
            // Pull extreme corners gently toward vertical mouth center
            archY[i] = -Math.max(-0.5, Math.min(0.5, (pos[i * 3 + 1] - mouthCenterY) / mouthHalfW));
        }
    }

    const diffuseField = (arr) => {
        const nxt = new Float32Array(n);
        // 5 seeded diffusion passes to spread influence into perioral cheeks/chin
        for (let iter = 0; iter < 5; iter++) {
            for (let i = 0; i < n; i++) {
                if (locked[i] === 1) {
                    nxt[i] = 0;
                    continue;
                }
                if (locked[i] === 2) {
                    nxt[i] = arr[i];
                    continue;
                }
                const nbrs = neighbors[i];
                if (!nbrs.length) {
                    nxt[i] = arr[i];
                    continue;
                }
                let sum = 0;
                for (let k = 0; k < nbrs.length; k++) sum += arr[nbrs[k]];
                nxt[i] = 0.30 * arr[i] + 0.70 * (sum / nbrs.length);
            }
            arr.set(nxt);
        }
        // 2 unpinned fairing passes eliminate any velocity step at the outer lip vermilion border
        for (let iter = 0; iter < 2; iter++) {
            for (let i = 0; i < n; i++) {
                if (locked[i] === 1) {
                    nxt[i] = 0;
                    continue;
                }
                const nbrs = neighbors[i];
                if (!nbrs.length) {
                    nxt[i] = arr[i];
                    continue;
                }
                let sum = 0;
                for (let k = 0; k < nbrs.length; k++) sum += arr[nbrs[k]];
                nxt[i] = 0.68 * arr[i] + 0.32 * (sum / nbrs.length);
            }
            arr.set(nxt);
        }
    };

    diffuseField(inwardX);
    diffuseField(archY);
    diffuseField(protrudeZ);

    return { mouthCenterX, mouthHalfW, inwardX, archY, protrudeZ };
}

export class FaceRetargetRig {
    constructor(restPositions, indices = [], initialWeights = null) {
        this.n = Math.floor(restPositions.length / 3);
        this.neighbors = buildAdjacency(this.n, indices);
        this.rest = new Float32Array(restPositions);
        this.symMap = buildSymmetryMap(this.rest);

        this.regionWeights = {
            eyes: 1.15,
            brows: 1.0,
            mouth: 1.0,
            mouthWidth: 0.55,
            jaw: 0.85,
            cheeks: 0.45,
        };
        if (initialWeights) {
            this.regionWeights = { ...this.regionWeights, ...initialWeights };
        }
        this.retargWeights = new Float32Array(this.n);
        this.retargWeightsX = new Float32Array(this.n);
        this.isOuterCheek = new Uint8Array(this.n);
        this._tempDx = new Float32Array(this.n);
        this._tempDy = new Float32Array(this.n);
        this._tempDz = new Float32Array(this.n);

        this.updateRestPositions(restPositions);
    }

    /**
     * Refresh rest metrics, eyelid bounds, lip rounding fields, and diffused region weights
     * when Sculpt sliders change without re-running the O(N^2) symmetry search.
     */
    updateRestPositions(restPositions) {
        this.rest.set(restPositions);

        const eyeL = midPt(this.rest, LEFT_EYE_OUTER_IDX, LEFT_EYE_INNER_IDX);
        const eyeR = midPt(this.rest, RIGHT_EYE_OUTER_IDX, RIGHT_EYE_INNER_IDX);
        this.charEyeSpan = Math.max(
            Math.hypot(eyeR[0] - eyeL[0], eyeR[1] - eyeL[1], eyeR[2] - eyeL[2]),
            0.05
        );

        const bridgeY = this.rest[NOSE_BRIDGE_IDX * 3 + 1];
        const chinY = this.rest[CHIN_IDX * 3 + 1];
        this.charChinDrop = Math.max(Math.abs(bridgeY - chinY), 0.12);

        this.eyeRight = buildEyeRig(this.rest, RIGHT_UPPER_LID, RIGHT_UPPER_FOLD, RIGHT_LOWER_LID, 33, 133);
        this.eyeLeft = buildEyeRig(this.rest, LEFT_UPPER_LID, LEFT_UPPER_FOLD, LEFT_LOWER_LID, 263, 362);
        this.rightEyeSocket = fitEyeSocket(this.rest, EYE_SPECS[0]);
        this.leftEyeSocket = fitEyeSocket(this.rest, EYE_SPECS[1]);
        this.rightEyeWrapIds = [...RIGHT_UPPER_LID, ...RIGHT_UPPER_FOLD, ...RIGHT_LOWER_LID];
        this.leftEyeWrapIds = [...LEFT_UPPER_LID, ...LEFT_UPPER_FOLD, ...LEFT_LOWER_LID];
        this.lipRound = buildLipRoundRig(this.rest, this.neighbors);

        this.eyeLineY = 0.5 * (eyeL[1] + eyeR[1]);
        this.maxAbsX = 1e-4;
        this.maxY = -Infinity;
        for (let i = 0; i < this.n; i++) {
            const x = Math.abs(this.rest[i * 3]);
            const y = this.rest[i * 3 + 1];
            if (x > this.maxAbsX) this.maxAbsX = x;
            if (y > this.maxY) this.maxY = y;
        }

        this.isOuterCheek.fill(0);
        for (let i = 0; i < this.n; i++) {
            const normX = Math.abs(this.rest[i * 3]) / this.maxAbsX;
            const vy = this.rest[i * 3 + 1];
            if (normX > 0.48 && vy < this.eyeLineY + 0.02) {
                this.isOuterCheek[i] = 1;
            }
        }

        this.setRegionWeights(this.regionWeights);
    }

    /**
     * Diffuse the 5 regional weight sliders smoothly across the 852-triangle mesh graph
     * so intermediate vertices blend naturally without boundary seams.
     */
    setRegionWeights(weights) {
        this.regionWeights = { ...this.regionWeights, ...weights };
        const { eyes, brows, mouth, jaw, cheeks } = this.regionWeights;
        const mw = Math.max(0.0, Math.min(1.5, this.regionWeights.mouthWidth ?? 0.55));
        const mouthX = mouth * mw;
        const jawX = jaw * Math.min(1.0, 0.45 + 0.55 * mw);

        const cur = new Float32Array(this.n);
        const curX = new Float32Array(this.n);
        const nxt = new Float32Array(this.n);
        const nxtX = new Float32Array(this.n);
        const seedKind = new Uint8Array(this.n); // 0=free, 1=locked(0), 2=region seed

        const browLineY = this.eyeLineY + (this.maxY - this.eyeLineY) * 0.28;

        for (let i = 0; i < this.n; i++) {
            const vx = this.rest[i * 3];
            const vy = this.rest[i * 3 + 1];
            const normX = Math.abs(vx) / this.maxAbsX;

            if (NOSE_BONE_INDICES.has(i)) {
                cur[i] = 0.08 * cheeks;
                curX[i] = 0.08 * cheeks;
                seedKind[i] = 1;
            } else if ((vy > this.eyeLineY - 0.04 && normX > 0.80) || vy > this.maxY * 0.90) {
                cur[i] = 0.0;
                curX[i] = 0.0;
                seedKind[i] = 1;
            } else if (this.eyeRight.isLidMask[i] || this.eyeLeft.isLidMask[i]) {
                cur[i] = eyes;
                curX[i] = eyes;
                seedKind[i] = 2;
            } else if (PUPIL_SET.has(i)) {
                cur[i] = eyes;
                curX[i] = eyes;
                seedKind[i] = 1;
            } else if (BROW_SET.has(i)) {
                cur[i] = brows;
                curX[i] = brows;
                seedKind[i] = 2;
            } else if (UPPER_LIP_SET.has(i) || LOWER_LIP_SET.has(i) || CORNER_LIP_SET.has(i)) {
                cur[i] = mouth;
                curX[i] = mouthX;
                seedKind[i] = 2;
            } else if (JAW_SET.has(i)) {
                cur[i] = jaw;
                curX[i] = jawX;
                seedKind[i] = 2;
            } else if (vy > browLineY) {
                cur[i] = brows * 0.35;
                curX[i] = brows * 0.35;
            } else {
                cur[i] = cheeks;
                curX[i] = cheeks;
            }
        }

        // 8 Laplacian diffusion passes across triangle edges keep both Y/Z and X fields strictly harmonic
        for (let iter = 0; iter < 8; iter++) {
            for (let i = 0; i < this.n; i++) {
                if (seedKind[i] === 1) {
                    nxt[i] = cur[i];
                    nxtX[i] = curX[i];
                    continue;
                }
                const nbrs = this.neighbors[i];
                if (!nbrs.length) {
                    nxt[i] = cur[i];
                    nxtX[i] = curX[i];
                    continue;
                }
                let sum = 0;
                let sumX = 0;
                for (let k = 0; k < nbrs.length; k++) {
                    const nb = nbrs[k];
                    sum += cur[nb];
                    sumX += curX[nb];
                }
                const inv = 1.0 / nbrs.length;
                const selfKeep = seedKind[i] === 2 ? 0.65 : 0.30;
                const nbrKeep = 1.0 - selfKeep;
                nxt[i] = selfKeep * cur[i] + nbrKeep * (sum * inv);
                nxtX[i] = selfKeep * curX[i] + nbrKeep * (sumX * inv);
            }
            cur.set(nxt);
            curX.set(nxtX);
        }

        this.retargWeights.set(cur);
        this.retargWeightsX.set(curX);
    }

    evaluate(motionFrame, outArray) {
        outArray.set(this.rest);
        if (!motionFrame) return;

        const { deltas, neutralAspect, mirror, bs, poseAngleMag = 0 } = motionFrame;

        // Damp outer cheek lateral foreshortening when head is turned strongly sideways
        const angleDamp = 1.0 - 0.65 * smoothstep(0.24, 0.72, poseAngleMag);

        let centeredGate = 1.0;

        if (deltas && deltas.length >= this.n * 3) {
            const scaleX = this.charEyeSpan;
            const charAspect = this.charChinDrop / this.charEyeSpan;
            const scaleY = this.charEyeSpan * (charAspect / Math.max(neutralAspect || charAspect, 0.6));
            const scaleZ = this.charEyeSpan * 0.32;
            const maxStep = this.charChinDrop * 0.78;

            // Separate whole-mouth sideways translation from symmetric inward corner pinch ("O"/pucker)
            const sgn = mirror ? -1.0 : 1.0;
            const idxL = mirror ? this.symMap[61] : 61;
            const idxR = mirror ? this.symMap[291] : 291;
            const idxU = mirror ? this.symMap[0] : 0;
            const idxD = mirror ? this.symMap[17] : 17;
            const dxL = sgn * deltas[idxL * 3];
            const dxR = sgn * deltas[idxR * 3];
            const dxU = sgn * deltas[idxU * 3];
            const dxD = sgn * deltas[idxD * 3];

            const mouthShiftX = 0.25 * (dxL + dxR + dxU + dxD);
            const cornerPinch = dxL - dxR; // > 0 when both corners pull inward toward midline
            centeredGate = 1.0 - smoothstep(0.018, 0.050, Math.abs(mouthShiftX));
            const pinchBoost = centeredGate * smoothstep(0.005, 0.045, cornerPinch);

            for (let i = 0; i < this.n; i++) {
                if (this.eyeRight.isLidMask[i] || this.eyeLeft.isLidMask[i]) {
                    this._tempDx[i] = 0;
                    this._tempDy[i] = 0;
                    this._tempDz[i] = 0;
                    continue;
                }
                const w = this.retargWeights[i];
                if (w <= 0.001) {
                    this._tempDx[i] = 0;
                    this._tempDy[i] = 0;
                    this._tempDz[i] = 0;
                    continue;
                }

                const srcIdx = mirror ? this.symMap[i] : i;
                const sOff = srcIdx * 3;
                const cAtten = this.isOuterCheek[i] ? angleDamp : 1.0;
                const wx = this.retargWeightsX[i] + pinchBoost * (w - this.retargWeightsX[i]);

                const dx = Math.max(-maxStep, Math.min(maxStep, (mirror ? -deltas[sOff] : deltas[sOff]) * scaleX * wx * cAtten));
                const dy = Math.max(-maxStep, Math.min(maxStep, deltas[sOff + 1] * scaleY * w));
                const dz = Math.max(-0.04, Math.min(0.04, deltas[sOff + 2] * scaleZ * w * cAtten));

                if (PUPIL_SET.has(i)) {
                    const maxGazeX = this.charEyeSpan * 0.09;
                    const maxGazeY = this.charEyeSpan * 0.055;
                    this._tempDx[i] = Math.max(-maxGazeX, Math.min(maxGazeX, dx));
                    this._tempDy[i] = Math.max(-maxGazeY, Math.min(maxGazeY, dy));
                    this._tempDz[i] = Math.max(-0.015, Math.min(0.015, dz));
                } else {
                    this._tempDx[i] = dx;
                    this._tempDy[i] = dy;
                    this._tempDz[i] = dz;
                }
            }

            // 1 gentle edge-smoothing pass prevents isolated outlier vertex spikes
            for (let i = 0; i < this.n; i++) {
                const dOff = i * 3;
                if (this.eyeRight.isLidMask[i] || this.eyeLeft.isLidMask[i] || this.retargWeights[i] <= 0.001) continue;
                const nbrs = this.neighbors[i];
                if (nbrs.length > 0) {
                    let sx = 0;
                    let sy = 0;
                    let sz = 0;
                    for (let k = 0; k < nbrs.length; k++) {
                        const nb = nbrs[k];
                        sx += this._tempDx[nb];
                        sy += this._tempDy[nb];
                        sz += this._tempDz[nb];
                    }
                    const inv = 1.0 / nbrs.length;
                    outArray[dOff] += 0.78 * this._tempDx[i] + 0.22 * sx * inv;
                    outArray[dOff + 1] += 0.78 * this._tempDy[i] + 0.22 * sy * inv;
                    outArray[dOff + 2] += 0.78 * this._tempDz[i] + 0.22 * sz * inv;
                } else {
                    outArray[dOff] += this._tempDx[i];
                    outArray[dOff + 1] += this._tempDy[i];
                    outArray[dOff + 2] += this._tempDz[i];
                }
            }
        }

        // Apply centered mouthFunnel ("O" rounding) & mouthPucker (kiss protrusion), gated off during side-shifts
        if (bs && this.lipRound && centeredGate > 0.01) {
            const mouthScale = Math.max(0.0, Math.min(1.65, this.regionWeights.mouth)) * centeredGate;
            const funnel = Math.max(0.0, Math.min(1.0, bs.mouthFunnel || 0));
            const pucker = Math.max(0.0, Math.min(1.0, bs.mouthPucker || 0));
            const roundStrength = Math.min(0.95, funnel * 0.75 + pucker * 0.85) * mouthScale;

            if (roundStrength > 0.01) {
                const { mouthHalfW, inwardX, archY, protrudeZ } = this.lipRound;
                const xAmp = mouthHalfW * 0.34 * roundStrength;
                const yAmp = mouthHalfW * 0.16 * funnel * mouthScale;
                const zAmp = this.charEyeSpan * (0.028 * pucker + 0.018 * funnel) * mouthScale;

                for (let i = 0; i < this.n; i++) {
                    const o = i * 3;
                    outArray[o] += inwardX[i] * xAmp;
                    outArray[o + 1] += archY[i] * yAmp;
                    outArray[o + 2] += protrudeZ[i] * zAmp;
                }
            }
        }

        // Apply EAR-fused eyelid blink / wide / squint closure scaled by the Eyes slider
        if (bs) {
            const eyeScale = Math.max(0.0, Math.min(2.0, this.regionWeights.eyes));
            const blinkL = Math.min(1.15, (bs.eyeBlinkLeft || 0) * eyeScale);
            const blinkR = Math.min(1.15, (bs.eyeBlinkRight || 0) * eyeScale);
            const wideL = (bs.eyeWideLeft || 0) * eyeScale;
            const wideR = (bs.eyeWideRight || 0) * eyeScale;
            const squintL = (bs.eyeSquintLeft || 0) * eyeScale;
            const squintR = (bs.eyeSquintRight || 0) * eyeScale;

            for (let i = 0; i < this.n; i++) {
                const yOff = i * 3 + 1;
                const rUp = this.eyeRight.upperWeights[i];
                const rLo = this.eyeRight.lowerWeights[i];
                if (rUp > 0) {
                    outArray[yOff] -= blinkR * this.eyeRight.lidGap * 1.02 * rUp;
                    outArray[yOff] += wideR * this.eyeRight.lidGap * 0.42 * rUp;
                }
                if (rLo > 0) {
                    outArray[yOff] += (blinkR * 0.26 + squintR * 0.40) * this.eyeRight.lidGap * rLo;
                }

                const lUp = this.eyeLeft.upperWeights[i];
                const lLo = this.eyeLeft.lowerWeights[i];
                if (lUp > 0) {
                    outArray[yOff] -= blinkL * this.eyeLeft.lidGap * 1.02 * lUp;
                    outArray[yOff] += wideL * this.eyeLeft.lidGap * 0.42 * lUp;
                }
                if (lLo > 0) {
                    outArray[yOff] += (blinkL * 0.26 + squintL * 0.40) * this.eyeLeft.lidGap * lLo;
                }
            }

            // Guarantee full column-by-column lid sealing at full blink before spherical wrapping
            const sealPairs = (upperIds, lowerIds, blinkVal, lidGap) => {
                if (blinkVal < 0.45) return;
                const seal = smoothstep(0.45, 0.96, blinkVal);
                const overlap = 0.05 * lidGap;
                for (let k = 0; k < upperIds.length; k++) {
                    const uY = upperIds[k] * 3 + 1;
                    const lY = lowerIds[k] * 3 + 1;
                    const maxUpperY = outArray[lY] - overlap;
                    if (outArray[uY] > maxUpperY) {
                        outArray[uY] += (maxUpperY - outArray[uY]) * seal;
                    }
                }
            };
            sealPairs(RIGHT_UPPER_LID, RIGHT_LOWER_LID, blinkR, this.eyeRight.lidGap);
            sealPairs(LEFT_UPPER_LID, LEFT_LOWER_LID, blinkL, this.eyeLeft.lidGap);
        }

        // Arc eyelid & fold vertices smoothly forward (+Z) over each eyeball dome to eliminate poke-through
        const wrapEye = (socket, spec, wrapIds) => {
            if (!socket) return;
            const outO = spec.outerIdx * 3;
            const inO = spec.innerIdx * 3;
            const cx = socket.cx + (0.5 * (outArray[outO] + outArray[inO]) - socket.cornerMidX);
            const cy = socket.cy + (0.5 * (outArray[outO + 1] + outArray[inO + 1]) - socket.cornerMidY);
            const cz = socket.cz + (0.5 * (outArray[outO + 2] + outArray[inO + 2]) - socket.cornerMidZ);
            clampEyelidsOverEyeball(outArray, socket, wrapIds, cx, cy, cz);
        };
        wrapEye(this.rightEyeSocket, EYE_SPECS[0], this.rightEyeWrapIds);
        wrapEye(this.leftEyeSocket, EYE_SPECS[1], this.leftEyeWrapIds);
    }
}
