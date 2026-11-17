import * as THREE from "../lib/three.module.js";
import { createOpenPoseMaterial } from "./face_shader.js";

// Exact 70-landmark MediaPipe -> OpenPose Face mapping matching OP_Face_0..69 in Yedp_Rig.glb
export const MP_TO_OP_FACE = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, // 0-16 Jaw
    46, 53, 52, 65, 55,       // 17-21 Right Brow (-X)
    285, 295, 282, 283, 276,  // 22-26 Left Brow (+X)
    6, 197, 195, 5,           // 27-30 Nose Bridge
    98, 97, 2, 326, 327,      // 31-35 Nose Bottom
    33, 160, 158, 133, 153, 144,  // 36-41 Right Eye (-X)
    362, 385, 387, 263, 373, 380, // 42-47 Left Eye (+X)
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, // 48-59 Outer Lips
    78, 81, 13, 311, 308, 402, 14, 178, // 60-67 Inner Lips
    468, 473,                 // 68-69 Pupils (-X, +X)
];

const SURFACE_JOINT_NAMES = [
    "M_NOSE_JT",
    "M_L_EYE_JT",
    "M_R_EYE_JT",
    "M_L_EAR_JT",
    "M_R_EAR_JT",
];

const SURFACE_STICK_SPECS = [
    { name: "M_L_EYEBROW", from: "nose", to: "lEye" },
    { name: "M_R_EYEBROW", from: "nose", to: "rEye" },
    { name: "M_L_EARLINE", from: "lEye", to: "lEar" },
    { name: "M_R_EARLINE", from: "rEye", to: "rEar" },
];

const BASE_FACE_DOT_SCALE = 0.34;
const BASE_HEAD_JOINT_SCALE = 0.28;

const _dir = new THREE.Vector3();
const _ref = new THREE.Vector3();
const _u = new THREE.Vector3();
const _v = new THREE.Vector3();

function computeCentroid(posArray, indices, outVec) {
    outVec.set(0, 0, 0);
    if (!indices?.length) return false;
    let count = 0;
    for (let i = 0; i < indices.length; i++) {
        const idx = indices[i] * 3;
        if (idx + 2 >= posArray.length) continue;
        outVec.x += posArray[idx];
        outVec.y += posArray[idx + 1];
        outVec.z += posArray[idx + 2];
        count++;
    }
    if (!count) return false;
    outVec.multiplyScalar(1.0 / count);
    return true;
}

function writeSphere(tpl, cx, cy, cz, scale, outPos, vOffset) {
    const src = tpl.positions;
    const n = Math.floor(src.length / 3);
    for (let i = 0; i < n; i++) {
        const s = i * 3;
        const d = (vOffset + i) * 3;
        outPos[d] = cx + src[s] * scale;
        outPos[d + 1] = cy + src[s + 1] * scale;
        outPos[d + 2] = cz + src[s + 2] * scale;
    }
}

function writeStick(tpl, p0, p1, radiusScale, outPos, vOffset) {
    _dir.subVectors(p1, p0);
    const len = _dir.length();
    if (len < 1e-5) {
        _dir.set(0, 1, 0);
    } else {
        _dir.multiplyScalar(1.0 / len);
    }
    _ref.set(Math.abs(_dir.y) < 0.9 ? 0 : 1, Math.abs(_dir.y) < 0.9 ? 1 : 0, 0);
    _u.crossVectors(_ref, _dir).normalize();
    _v.crossVectors(_dir, _u).normalize();

    const uv = tpl.radialUV;
    const endFlag = tpl.endFlag;
    const n = endFlag.length;
    for (let i = 0; i < n; i++) {
        const base = endFlag[i] ? p1 : p0;
        const ru = uv[i * 2] * radiusScale;
        const rv = uv[i * 2 + 1] * radiusScale;
        const d = (vOffset + i) * 3;
        outPos[d] = base.x + _u.x * ru + _v.x * rv;
        outPos[d + 1] = base.y + _u.y * ru + _v.y * rv;
        outPos[d + 2] = base.z + _u.z * ru + _v.z * rv;
    }
}

function appendTemplateMeta(tpl, vertOffset, positions, colors, indices) {
    const nVerts = tpl.positions ? Math.floor(tpl.positions.length / 3) : tpl.endFlag.length;
    const [r, g, b] = tpl.color;
    for (let i = 0; i < nVerts; i++) {
        colors.push(r, g, b);
        positions.push(0, 0, 0);
    }
    for (let i = 0; i < tpl.indices.length; i++) {
        indices.push(vertOffset + tpl.indices[i]);
    }
    return vertOffset + nVerts;
}

/**
 * Unlit 3D OpenPose Head + Neck + 70-Point Face skeleton extracted from `Yedp_Rig.glb`
 * and driven in real time by MediaPipe reconstruction, SculptRig, and FaceRetargetRig.
 */
export class OpenPoseFaceRig {
    constructor(rigData) {
        this.rigData = rigData;
        this.group = new THREE.Group();
        this.group.visible = false;

        this.poseMode = "head_face";
        this.dotScale = 1.0;
        this.occlude = true;
        this.overlay = false;
        this.guidesVisible = true;

        this.lastPositions = null;
        this.lastShellRig = null;

        this.anchors = {
            nose: new THREE.Vector3(),
            lEye: new THREE.Vector3(),
            rEye: new THREE.Vector3(),
            lEar: new THREE.Vector3(),
            rEar: new THREE.Vector3(),
            neck: new THREE.Vector3(),
        };

        // 1. Surface mesh: 70 OP_Face white dots FIRST, then 5 head spheres + 4 head sticks
        const surfPos = [];
        const surfCol = [];
        const surfIdx = [];
        let surfOff = 0;

        this.faceDotOffsets = new Int32Array(MP_TO_OP_FACE.length);
        for (let k = 0; k < MP_TO_OP_FACE.length; k++) {
            this.faceDotOffsets[k] = surfOff;
            surfOff = appendTemplateMeta(rigData.faceDotTemplate, surfOff, surfPos, surfCol, surfIdx);
        }
        this.faceOnlyIndexCount = surfIdx.length;

        this.surfaceJointEntries = [];
        for (const name of SURFACE_JOINT_NAMES) {
            const tpl = rigData.jointTemplates[name];
            if (!tpl) continue;
            this.surfaceJointEntries.push({ name, offset: surfOff, tpl });
            surfOff = appendTemplateMeta(tpl, surfOff, surfPos, surfCol, surfIdx);
        }

        this.surfaceStickEntries = [];
        for (const spec of SURFACE_STICK_SPECS) {
            const tpl = rigData.stickTemplates[spec.name];
            if (!tpl) continue;
            this.surfaceStickEntries.push({ ...spec, offset: surfOff, tpl });
            surfOff = appendTemplateMeta(tpl, surfOff, surfPos, surfCol, surfIdx);
        }
        this.surfaceTotalIndexCount = surfIdx.length;

        this.surfaceGeo = new THREE.BufferGeometry();
        this.surfacePositions = new Float32Array(surfPos);
        this.surfaceGeo.setAttribute("position", new THREE.BufferAttribute(this.surfacePositions, 3));
        this.surfaceGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(surfCol), 3));
        this.surfaceGeo.setIndex(surfIdx);

        this.surfaceMat = createOpenPoseMaterial();
        this.surfaceMesh = new THREE.Mesh(this.surfaceGeo, this.surfaceMat);
        this.surfaceMesh.frustumCulled = false;

        // 2. Armature stem mesh: Neck joint + M_HEAD stick (rendered behind occlusion core when occluding)
        const armPos = [];
        const armCol = [];
        const armIdx = [];
        let armOff = 0;

        this.neckJointEntry = null;
        if (rigData.jointTemplates.M_NECK_JT) {
            this.neckJointEntry = { offset: armOff, tpl: rigData.jointTemplates.M_NECK_JT };
            armOff = appendTemplateMeta(this.neckJointEntry.tpl, armOff, armPos, armCol, armIdx);
        }
        this.headStickEntry = null;
        if (rigData.stickTemplates.M_HEAD) {
            this.headStickEntry = { offset: armOff, tpl: rigData.stickTemplates.M_HEAD };
            armOff = appendTemplateMeta(this.headStickEntry.tpl, armOff, armPos, armCol, armIdx);
        }

        this.armatureGeo = new THREE.BufferGeometry();
        this.armaturePositions = new Float32Array(armPos);
        this.armatureGeo.setAttribute("position", new THREE.BufferAttribute(this.armaturePositions, 3));
        this.armatureGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(armCol), 3));
        this.armatureGeo.setIndex(armIdx);

        this.armatureMat = createOpenPoseMaterial();
        this.armatureMesh = new THREE.Mesh(this.armatureGeo, this.armatureMat);
        this.armatureMesh.frustumCulled = false;

        this.group.add(this.armatureMesh, this.surfaceMesh);
        this.applyRenderFlags();
    }

    setOptions({ poseMode, dotScale, occlude, guidesVisible, overlay } = {}) {
        let needsGeoUpdate = false;
        if (poseMode !== undefined && poseMode !== this.poseMode) {
            this.poseMode = poseMode;
        }
        if (dotScale !== undefined) {
            const nextScale = Math.max(0.25, Math.min(2.5, Number(dotScale) || 1.0));
            if (nextScale !== this.dotScale) {
                this.dotScale = nextScale;
                needsGeoUpdate = true;
            }
        }
        if (occlude !== undefined) {
            this.occlude = Boolean(occlude);
        }
        if (guidesVisible !== undefined && Boolean(guidesVisible) !== this.guidesVisible) {
            this.guidesVisible = Boolean(guidesVisible);
            needsGeoUpdate = true;
        }
        if (overlay !== undefined) {
            this.overlay = Boolean(overlay);
        }

        this.applyRenderFlags();
        if (needsGeoUpdate && this.lastPositions) {
            this.updateFromPositions(this.lastPositions, this.lastShellRig, this.guidesVisible);
        }
    }

    applyRenderFlags() {
        const showHeadSkeleton = this.poseMode !== "face_only";
        this.surfaceGeo.setDrawRange(0, showHeadSkeleton ? this.surfaceTotalIndexCount : this.faceOnlyIndexCount);
        this.armatureMesh.visible = showHeadSkeleton && this.guidesVisible && this.armaturePositions.length > 0;

        this.armatureMesh.renderOrder = !this.overlay && this.occlude ? -1 : 2;
        this.surfaceMesh.renderOrder = 2;

        const useOffset = this.overlay;
        for (const mat of [this.surfaceMat, this.armatureMat]) {
            mat.polygonOffset = useOffset;
            mat.polygonOffsetFactor = useOffset ? -4 : 0;
            mat.polygonOffsetUnits = useOffset ? -8 : 0;
        }
    }

    updateFromPositions(posArray, headShellRig = null, guidesVisible = this.guidesVisible) {
        if (!posArray || posArray.length < 468 * 3) return;
        this.lastPositions = posArray;
        this.lastShellRig = headShellRig;
        this.guidesVisible = Boolean(guidesVisible);

        // Compute scale ratio relative to Yedp_Rig.glb OP_Face_0..16 span
        const dxSpan = posArray[389 * 3] - posArray[162 * 3];
        const dySpan = posArray[389 * 3 + 1] - posArray[162 * 3 + 1];
        const dzSpan = posArray[389 * 3 + 2] - posArray[162 * 3 + 2];
        const faceSpanX = Math.max(Math.hypot(dxSpan, dySpan, dzSpan), 0.1);
        const headScale = faceSpanX / Math.max(this.rigData.rigFaceSpanX, 1e-4);

        const dotScale = headScale * BASE_FACE_DOT_SCALE * this.dotScale;
        const jointScale = headScale * BASE_HEAD_JOINT_SCALE * this.dotScale;

        // 1. Place 70 OP_Face white spheres
        const dotTpl = this.rigData.faceDotTemplate;
        const hasIris = posArray.length >= 478 * 3;
        for (let k = 0; k < MP_TO_OP_FACE.length; k++) {
            let mpIdx = MP_TO_OP_FACE[k];
            if (mpIdx >= 468 && !hasIris) {
                mpIdx = mpIdx === 468 ? 159 : 386;
            }
            const o = mpIdx * 3;
            writeSphere(dotTpl, posArray[o], posArray[o + 1], posArray[o + 2], dotScale, this.surfacePositions, this.faceDotOffsets[k]);
        }

        // 2. Resolve live Head & Neck OpenPose anchors
        const a = this.anchors;
        a.nose.set(posArray[3], posArray[4], posArray[5]); // Landmark #1 (nose tip)
        if (hasIris) {
            a.rEye.set(posArray[468 * 3], posArray[468 * 3 + 1], posArray[468 * 3 + 2]);
            a.lEye.set(posArray[473 * 3], posArray[473 * 3 + 1], posArray[473 * 3 + 2]);
        } else {
            a.rEye.set(
                0.5 * (posArray[33 * 3] + posArray[133 * 3]),
                0.5 * (posArray[33 * 3 + 1] + posArray[133 * 3 + 1]),
                0.5 * (posArray[33 * 3 + 2] + posArray[133 * 3 + 2])
            );
            a.lEye.set(
                0.5 * (posArray[362 * 3] + posArray[263 * 3]),
                0.5 * (posArray[362 * 3 + 1] + posArray[263 * 3 + 1]),
                0.5 * (posArray[362 * 3 + 2] + posArray[263 * 3 + 2])
            );
        }

        const hasShell = Boolean(this.guidesVisible && headShellRig && posArray.length >= 1258 * 3);
        if (!hasShell || !computeCentroid(posArray, headShellRig.leftEarMerged, a.lEar)) {
            a.lEar.set(posArray[454 * 3], posArray[454 * 3 + 1], posArray[454 * 3 + 2]);
        }
        if (!hasShell || !computeCentroid(posArray, headShellRig.rightEarMerged, a.rEar)) {
            a.rEar.set(posArray[234 * 3], posArray[234 * 3 + 1], posArray[234 * 3 + 2]);
        }
        if (!hasShell || !computeCentroid(posArray, headShellRig.neckLoopMerged, a.neck)) {
            a.neck.set(
                posArray[152 * 3],
                posArray[152 * 3 + 1] - faceSpanX * 0.42,
                posArray[152 * 3 + 2] - faceSpanX * 0.16
            );
        }

        const jointMap = {
            M_NOSE_JT: a.nose,
            M_L_EYE_JT: a.lEye,
            M_R_EYE_JT: a.rEye,
            M_L_EAR_JT: a.lEar,
            M_R_EAR_JT: a.rEar,
        };
        for (const entry of this.surfaceJointEntries) {
            const pt = jointMap[entry.name];
            if (pt) {
                writeSphere(entry.tpl, pt.x, pt.y, pt.z, jointScale, this.surfacePositions, entry.offset);
            }
        }
        for (const entry of this.surfaceStickEntries) {
            writeStick(entry.tpl, a[entry.from], a[entry.to], jointScale, this.surfacePositions, entry.offset);
        }

        this.surfaceGeo.attributes.position.needsUpdate = true;
        this.surfaceGeo.computeBoundingSphere();

        if (this.neckJointEntry) {
            writeSphere(
                this.neckJointEntry.tpl,
                a.neck.x,
                a.neck.y,
                a.neck.z,
                jointScale,
                this.armaturePositions,
                this.neckJointEntry.offset
            );
        }
        if (this.headStickEntry) {
            writeStick(
                this.headStickEntry.tpl,
                a.neck,
                a.nose,
                jointScale,
                this.armaturePositions,
                this.headStickEntry.offset
            );
        }
        if (this.armaturePositions.length > 0) {
            this.armatureGeo.attributes.position.needsUpdate = true;
            this.armatureGeo.computeBoundingSphere();
        }
        this.applyRenderFlags();
    }

    dispose() {
        this.surfaceGeo.dispose();
        this.surfaceMat.dispose();
        this.armatureGeo.dispose();
        this.armatureMat.dispose();
    }
}
