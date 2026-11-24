import * as THREE from "../lib/three.module.js";

const NOSE_BRIDGE_IDX = 168;
const FOREHEAD_IDX = 10;
const CHIN_IDX = 152;
const LEFT_EYE_OUTER_IDX = 33;
const LEFT_EYE_INNER_IDX = 133;
const RIGHT_EYE_OUTER_IDX = 263;
const RIGHT_EYE_INNER_IDX = 362;

const _mat4 = new THREE.Matrix4();
const _p = new THREE.Vector3();
const _origin = new THREE.Vector3();
const _xAxis = new THREE.Vector3();
const _yTemp = new THREE.Vector3();
const _zAxis = new THREE.Vector3();
const _yAxis = new THREE.Vector3();
const _eyeL = new THREE.Vector3();
const _eyeR = new THREE.Vector3();
const _chin = new THREE.Vector3();

function clamp01(v) {
    return Math.max(0, Math.min(1, v));
}

function smoothstep(edge0, edge1, x) {
    const t = clamp01((x - edge0) / Math.max(edge1 - edge0, 1e-6));
    return t * t * (3 - 2 * t);
}

// Region tags: 2 = Mouth & Jaw (snappy speech), 1 = Brows & Pupils (fast saccades), 0 = Cheeks/Rest (anti-jitter)
const REGION_TAGS = (() => {
    const tags = new Uint8Array(478);
    const group2 = [
        0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 42, 61, 62,
        72, 73, 74, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96,
        146, 148, 149, 150, 152, 175, 176, 178, 179, 180, 181, 183, 184, 185, 191,
        199, 200, 201, 267, 268, 269, 270, 271, 272, 291, 292, 302, 303, 304,
        306, 307, 308, 310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321,
        324, 325, 375, 377, 378, 379, 400, 402, 403, 404, 405, 407, 408, 409, 415,
    ];
    const group1 = [
        46, 52, 53, 55, 63, 65, 66, 70, 105, 107,
        276, 282, 283, 285, 293, 295, 296, 300, 334, 336,
        468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
    ];
    for (const id of group1) tags[id] = 1;
    for (const id of group2) tags[id] = 2;
    return tags;
})();

function swapLeftRightName(name) {
    if (name.endsWith("Left")) return `${name.slice(0, -4)}Right`;
    if (name.endsWith("Right")) return `${name.slice(0, -5)}Left`;
    return name;
}

function computeEyeOpenRatio(landmarks, upperIdx, lowerIdx, outerIdx, innerIdx) {
    const up = landmarks[upperIdx];
    const lo = landmarks[lowerIdx];
    const out = landmarks[outerIdx];
    const inn = landmarks[innerIdx];
    if (!up || !lo || !out || !inn) return null;
    const eyeW = Math.hypot(out.x - inn.x, out.y - inn.y);
    if (eyeW < 1e-4) return null;
    const eyeH = Math.hypot(up.x - lo.x, up.y - lo.y);
    return eyeH / eyeW;
}

function toMetricVec(lm, aspect, outVec) {
    return outVec.set(lm.x - 0.5, -(lm.y - 0.5) * aspect, -lm.z);
}

/**
 * Extracts head pose quaternion, intrinsic upper-skull normalized 478-vertex deltas
 * (compared against the actor's calibrated neutral reference), and EAR-fused blinks.
 */
export class MotionDriver {
    constructor() {
        this.neutralEuler = new THREE.Euler(0, 0, 0, "YXZ");
        this.neutralSkull = null;
        this.neutralFaceAspect = 1.0;
        this.hasCalibration = false;

        this.smoothedQuat = new THREE.Quaternion();
        this.smoothedDeltas = null;
        this.smoothedBlinkL = 0;
        this.smoothedBlinkR = 0;
        this.hasPrev = false;

        this.openEarLeft = 0.28;
        this.openEarRight = 0.28;
    }

    reset() {
        this.neutralEuler.set(0, 0, 0, "YXZ");
        this.neutralSkull = null;
        this.neutralFaceAspect = 1.0;
        this.hasCalibration = false;

        this.smoothedQuat.identity();
        this.smoothedDeltas = null;
        this.smoothedBlinkL = 0;
        this.smoothedBlinkR = 0;
        this.hasPrev = false;

        this.openEarLeft = 0.28;
        this.openEarRight = 0.28;
    }

    calibrateNextFrame() {
        this.hasCalibration = false;
        this.neutralSkull = null;
        this.openEarLeft = 0.26;
        this.openEarRight = 0.26;
    }

    processDetection(detection, { mirrorYaw = false, smoothAlpha = 0.5 } = {}) {
        if (!detection?.landmarks?.length) return null;

        const { landmarks, matrix, blendshapes, width, height } = detection;
        const n = landmarks.length;
        const aspect = height / Math.max(width, 1);

        // 1. Head rotation quaternion from MediaPipe 4x4 pose matrix
        const rawQuat = new THREE.Quaternion();
        if (matrix && matrix.length === 16) {
            _mat4.fromArray(matrix);
            rawQuat.setFromRotationMatrix(_mat4);
        }

        const euler = new THREE.Euler().setFromQuaternion(rawQuat, "YXZ");
        if (mirrorYaw) {
            euler.y = -euler.y;
            euler.z = -euler.z;
        }

        if (!this.hasCalibration) {
            this.neutralEuler.set(euler.x * 0.7, euler.y, euler.z, "YXZ");
        }

        euler.x -= this.neutralEuler.x;
        euler.y -= this.neutralEuler.y;
        euler.z -= this.neutralEuler.z;
        const poseQuat = new THREE.Quaternion().setFromEuler(euler);

        // 2. Intrinsic upper-skull basis built directly from rigid cranial landmarks
        toMetricVec(landmarks[NOSE_BRIDGE_IDX], aspect, _origin);
        toMetricVec(landmarks[LEFT_EYE_OUTER_IDX], aspect, _eyeL);
        toMetricVec(landmarks[LEFT_EYE_INNER_IDX], aspect, _p);
        _eyeL.add(_p).multiplyScalar(0.5);

        toMetricVec(landmarks[RIGHT_EYE_OUTER_IDX], aspect, _eyeR);
        toMetricVec(landmarks[RIGHT_EYE_INNER_IDX], aspect, _p);
        _eyeR.add(_p).multiplyScalar(0.5);

        const interOcular = Math.max(_eyeL.distanceTo(_eyeR), 1e-4);
        _xAxis.subVectors(_eyeR, _eyeL).normalize();

        toMetricVec(landmarks[FOREHEAD_IDX], aspect, _yTemp);
        _yTemp.sub(_origin).normalize();

        _zAxis.crossVectors(_xAxis, _yTemp).normalize();
        _yAxis.crossVectors(_zAxis, _xAxis).normalize();

        // Project all 478 landmarks into the rigid skull frame normalized by inter-ocular width
        const skullCoords = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) {
            toMetricVec(landmarks[i], aspect, _p);
            _p.sub(_origin);
            skullCoords[i * 3] = _p.dot(_xAxis) / interOcular;
            skullCoords[i * 3 + 1] = _p.dot(_yAxis) / interOcular;
            skullCoords[i * 3 + 2] = _p.dot(_zAxis) / interOcular;
        }

        if (!this.hasCalibration || !this.neutralSkull || this.neutralSkull.length !== skullCoords.length) {
            this.neutralSkull = new Float32Array(skullCoords);
            toMetricVec(landmarks[CHIN_IDX], aspect, _chin);
            _chin.sub(_origin);
            const neutralChinDrop = Math.abs(_chin.dot(_yAxis)) / interOcular;
            this.neutralFaceAspect = Math.max(neutralChinDrop, 0.8);
            this.hasCalibration = true;
        }

        const rawDeltas = new Float32Array(n * 3);
        for (let i = 0; i < skullCoords.length; i++) {
            rawDeltas[i] = skullCoords[i] - this.neutralSkull[i];
        }

        // 3. Fuse ARKit blinks with geometric Eye Aspect Ratio (EAR) for reliable blink closure behind glasses
        const rawBs = {};
        if (blendshapes) {
            for (const cat of blendshapes) {
                if (!cat.categoryName) continue;
                const key = mirrorYaw ? swapLeftRightName(cat.categoryName) : cat.categoryName;
                rawBs[key] = cat.score;
            }
        }

        const earR = computeEyeOpenRatio(landmarks, 159, 145, 33, 133);
        const earL = computeEyeOpenRatio(landmarks, 386, 374, 263, 362);
        const mapBlink = (arkitRaw, ear, openRefKey) => {
            let score = clamp01((arkitRaw - 0.12) / 0.50);
            if (ear !== null) {
                if (arkitRaw < 0.20 && ear > this[openRefKey] * 0.85) {
                    this[openRefKey] = this[openRefKey] * 0.92 + ear * 0.08;
                }
                const normEar = ear / Math.max(this[openRefKey], 0.14);
                const earBlink = clamp01((0.74 - normEar) / 0.36);
                score = Math.max(score, earBlink);
            }
            return score;
        };

        const blinkR = mapBlink(rawBs.eyeBlinkRight || 0, mirrorYaw ? earL : earR, mirrorYaw ? "openEarLeft" : "openEarRight");
        const blinkL = mapBlink(rawBs.eyeBlinkLeft || 0, mirrorYaw ? earR : earL, mirrorYaw ? "openEarRight" : "openEarLeft");

        // 4. Velocity-adaptive regional EMA for pose, vertex deltas, and fast blinks
        if (smoothAlpha < 1.0 && this.hasPrev && this.smoothedDeltas?.length === rawDeltas.length) {
            const angStep = 2.0 * Math.acos(Math.min(1.0, Math.abs(this.smoothedQuat.dot(poseQuat))));
            const rotT = smoothstep(0.006, 0.045, angStep);
            const rotAlpha = 0.35 + (0.68 - 0.35) * rotT;
            this.smoothedQuat.slerp(poseQuat, rotAlpha);

            const highAlphaFloor = smoothAlpha > 0.55 ? Math.min(smoothAlpha, 0.90) : 0.0;
            for (let v = 0; v < n; v++) {
                const o = v * 3;
                const step = Math.hypot(
                    rawDeltas[o] - this.smoothedDeltas[o],
                    rawDeltas[o + 1] - this.smoothedDeltas[o + 1],
                    rawDeltas[o + 2] - this.smoothedDeltas[o + 2]
                );
                const t = smoothstep(0.004, 0.022, step);
                const group = v < REGION_TAGS.length ? REGION_TAGS[v] : 0;
                let alpha = group === 2
                    ? 0.48 + (0.88 - 0.48) * t
                    : group === 1
                        ? 0.45 + (0.82 - 0.45) * t
                        : 0.30 + (0.60 - 0.30) * t;
                if (highAlphaFloor > alpha) alpha = highAlphaFloor;

                const invA = 1.0 - alpha;
                this.smoothedDeltas[o] = this.smoothedDeltas[o] * invA + rawDeltas[o] * alpha;
                this.smoothedDeltas[o + 1] = this.smoothedDeltas[o + 1] * invA + rawDeltas[o + 1] * alpha;
                this.smoothedDeltas[o + 2] = this.smoothedDeltas[o + 2] * invA + rawDeltas[o + 2] * alpha;
            }
            const blinkAlpha = Math.max(smoothAlpha, 0.88);
            this.smoothedBlinkL = this.smoothedBlinkL * (1.0 - blinkAlpha) + blinkL * blinkAlpha;
            this.smoothedBlinkR = this.smoothedBlinkR * (1.0 - blinkAlpha) + blinkR * blinkAlpha;
        } else {
            this.smoothedQuat.copy(poseQuat);
            this.smoothedDeltas = rawDeltas;
            this.smoothedBlinkL = blinkL;
            this.smoothedBlinkR = blinkR;
            this.hasPrev = true;
        }

        rawBs.eyeBlinkLeft = this.smoothedBlinkL;
        rawBs.eyeBlinkRight = this.smoothedBlinkR;
        const poseAngleMag = Math.hypot(euler.x, euler.y);

        return {
            quat: [this.smoothedQuat.x, this.smoothedQuat.y, this.smoothedQuat.z, this.smoothedQuat.w],
            deltas: new Float32Array(this.smoothedDeltas),
            neutralAspect: this.neutralFaceAspect,
            poseAngleMag,
            mirror: mirrorYaw,
            bs: rawBs,
        };
    }
}
