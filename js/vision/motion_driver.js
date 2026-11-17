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

        // 4. Smooth pose, vertex deltas, and fast blinks
        if (smoothAlpha < 1.0 && this.hasPrev && this.smoothedDeltas?.length === rawDeltas.length) {
            this.smoothedQuat.slerp(poseQuat, smoothAlpha);
            const deltaAlpha = Math.max(smoothAlpha, 0.55);
            const invD = 1.0 - deltaAlpha;
            for (let i = 0; i < rawDeltas.length; i++) {
                this.smoothedDeltas[i] = this.smoothedDeltas[i] * invD + rawDeltas[i] * deltaAlpha;
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
