import * as THREE from "../lib/three.module.js";
import {
    createFaceMaterial,
    createGuideDepthMaterial,
    createDepthHullMaterial,
    createPlaceholderTexture,
} from "./face_shader.js";
import { FaceRetargetRig } from "../vision/retarget_rig.js";
import { SculptRig } from "../vision/sculpt_rig.js";
import { MeshSubdivider } from "./mesh_subdiv.js";
import { fetchWeldedGlbMesh, fetchOpenPoseRigGlb } from "./glb_loader.js";
import { HeadShellRig } from "./head_shell_rig.js";
import { OpenPoseFaceRig } from "./openpose_rig.js";
import { EYE_SPECS, EyeballRig } from "./eyeball_rig.js";

const _worldCenter = new THREE.Vector3();
const _dbgColor = new THREE.Color();
const _yAxis = new THREE.Vector3(0, 1, 0);
const _yawQuat = new THREE.Quaternion();
const _actorQuat = new THREE.Quaternion();
const SKIN_SAMPLE_LANDMARKS = [50, 280, 117, 346, 151, 9, 195];
const DEFAULT_SCLERA_HEX = 0xd6d1cb;

let _headShellRigPromise = null;
let _openPoseRigDataPromise = null;
let _canonicalFaceDataPromise = null;

export function getHeadShellRig() {
    if (!_headShellRigPromise) {
        const glbUrl = new URL("../../assets/head_shell.glb", import.meta.url).href;
        _headShellRigPromise = fetchWeldedGlbMesh(glbUrl)
            .then(({ positions, indices }) => new HeadShellRig(positions, indices))
            .catch((err) => {
                console.warn("[YedpFace3D] Optional head_shell.glb not loaded:", err?.message || err);
                return null;
            });
    }
    return _headShellRigPromise;
}

export function getCanonicalFaceData() {
    if (!_canonicalFaceDataPromise) {
        const jsonUrl = new URL("../../assets/canonical_face.json", import.meta.url).href;
        _canonicalFaceDataPromise = fetch(jsonUrl).then((res) => {
            if (!res.ok) {
                throw new Error(`Failed to load canonical_face.json (HTTP ${res.status})`);
            }
            return res.json();
        });
    }
    return _canonicalFaceDataPromise;
}

function getOpenPoseRigData() {
    if (!_openPoseRigDataPromise) {
        const glbUrl = new URL("../../assets/Yedp_Rig.glb", import.meta.url).href;
        _openPoseRigDataPromise = fetchOpenPoseRigGlb(glbUrl).catch((err) => {
            console.warn("[YedpFace3D] Optional Yedp_Rig.glb OpenPose not loaded:", err?.message || err);
            return null;
        });
    }
    return _openPoseRigDataPromise;
}

function sampleMedianSkinColor(img, uvs) {
    try {
        const w = Math.min(img.naturalWidth || img.width || 256, 256);
        const h = Math.min(img.naturalHeight || img.height || 256, 256);
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(img, 0, 0, w, h);

        const rs = [];
        const gs = [];
        const bs = [];
        for (const id of SKIN_SAMPLE_LANDMARKS) {
            if (id * 2 + 1 >= uvs.length) continue;
            const px = Math.max(0, Math.min(w - 1, Math.round(uvs[id * 2] * (w - 1))));
            const py = Math.max(0, Math.min(h - 1, Math.round((1.0 - uvs[id * 2 + 1]) * (h - 1))));
            const d = ctx.getImageData(px, py, 1, 1).data;
            rs.push(d[0]);
            gs.push(d[1]);
            bs.push(d[2]);
        }
        if (!rs.length) return null;
        const mid = Math.floor(rs.length / 2);
        rs.sort((a, b) => a - b);
        gs.sort((a, b) => a - b);
        bs.sort((a, b) => a - b);
        return [rs[mid] / 255, gs[mid] / 255, bs[mid] / 255];
    } catch {
        return null;
    }
}

function medianChannel(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return sorted.length % 2 === 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

export function clampNaturalScleraRgb(r, g, b) {
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const maxDev = Math.max(Math.abs(r - luma), Math.abs(g - luma), Math.abs(b - luma));
    if (maxDev > 0.045) {
        const satScale = 0.045 / maxDev;
        r = luma + (r - luma) * satScale;
        g = luma + (g - luma) * satScale;
        b = luma + (b - luma) * satScale;
    }
    const curLuma = Math.max(0.2126 * r + 0.7152 * g + 0.0722 * b, 1e-4);
    const targetLuma = Math.max(0.45, Math.min(0.84, curLuma));
    const lumaScale = targetLuma / curLuma;
    return [
        Math.max(0.42, Math.min(0.86, r * lumaScale)),
        Math.max(0.42, Math.min(0.86, g * lumaScale)),
        Math.max(0.42, Math.min(0.86, b * lumaScale)),
    ];
}

export function sampleNaturalScleraColor(img, uvs) {
    try {
        if (!img || !uvs || uvs.length < 468 * 2) return null;
        const natW = img.naturalWidth || img.width || 512;
        const natH = img.naturalHeight || img.height || 512;
        const scale = Math.min(1.0, 1024 / Math.max(natW, natH, 1));
        const w = Math.max(1, Math.round(natW * scale));
        const h = Math.max(1, Math.round(natH * scale));
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(img, 0, 0, w, h);

        const toPx = (idx) => [uvs[idx * 2] * (w - 1), (1.0 - uvs[idx * 2 + 1]) * (h - 1)];
        const hasIris = uvs.length >= 478 * 2;
        const samples = [];

        for (let eIdx = 0; eIdx < EYE_SPECS.length; eIdx++) {
            const spec = EYE_SPECS[eIdx];
            const outerPx = toPx(spec.outerIdx);
            const innerPx = toPx(spec.innerIdx);
            const upPx = toPx(spec.upperMidIdx);
            const loPx = toPx(spec.lowerMidIdx);
            const eyeSpanPx = Math.hypot(innerPx[0] - outerPx[0], innerPx[1] - outerPx[1]);
            if (eyeSpanPx < 4) continue;

            const pupilPx = hasIris
                ? toPx(spec.pupilIdx)
                : [0.5 * (outerPx[0] + innerPx[0]), 0.5 * (outerPx[1] + innerPx[1])];
            let rIrisPx = 0.22 * eyeSpanPx;
            if (hasIris) {
                let sumR = 0;
                for (const ri of spec.irisRingIndices) {
                    const rp = toPx(ri);
                    sumR += Math.hypot(rp[0] - pupilPx[0], rp[1] - pupilPx[1]);
                }
                rIrisPx = sumR / spec.irisRingIndices.length;
            }

            const upX = upPx[0] - loPx[0];
            const upY = upPx[1] - loPx[1];
            const corners = [outerPx, innerPx];
            for (let cIdx = 0; cIdx < corners.length; cIdx++) {
                const corner = corners[cIdx];
                const dx = corner[0] - pupilPx[0];
                const dy = corner[1] - pupilPx[1];
                const dist = Math.hypot(dx, dy);
                if (dist <= rIrisPx * 1.05) continue;

                const t = (0.5 * (rIrisPx + dist)) / dist;
                // Slightly nudge above/below horizontal midline to avoid lashes while spanning both eyelids
                const vShift = (eIdx + cIdx) % 2 === 0 ? -0.06 : 0.05;
                const px = Math.max(0, Math.min(w - 1, Math.round(pupilPx[0] + dx * t + upX * vShift)));
                const py = Math.max(0, Math.min(h - 1, Math.round(pupilPx[1] + dy * t + upY * vShift)));
                const d = ctx.getImageData(px, py, 1, 1).data;
                if (d[3] < 128) continue;

                const r = d[0] / 255;
                const g = d[1] / 255;
                const b = d[2] / 255;
                const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                if (luma < 0.22 || luma > 0.85) continue;
                samples.push({ r, g, b, luma });
            }
        }

        if (!samples.length) return null;
        const rMed = medianChannel(samples.map((s) => s.r)) * 0.94;
        const gMed = medianChannel(samples.map((s) => s.g)) * 0.94;
        const bMed = medianChannel(samples.map((s) => s.b)) * 0.94;
        return clampNaturalScleraRgb(rMed, gMed, bMed);
    } catch {
        return null;
    }
}

function loadTextureFromDataUrl(dataUrl, uvs, filter = THREE.LinearFilter, sampleSclera = false) {
    return new Promise((resolve, reject) => {
        if (!dataUrl) {
            resolve({ tex: null, skinColor: null, scleraColor: null });
            return;
        }
        const img = new Image();
        img.onload = () => {
            const skinColor = uvs ? sampleMedianSkinColor(img, uvs) : null;
            const scleraColor = (sampleSclera && uvs) ? sampleNaturalScleraColor(img, uvs) : null;
            const tex = new THREE.Texture(img);
            tex.colorSpace = THREE.SRGBColorSpace;
            tex.flipY = true;
            tex.wrapS = THREE.ClampToEdgeWrapping;
            tex.wrapT = THREE.ClampToEdgeWrapping;
            tex.minFilter = filter;
            tex.magFilter = filter;
            tex.needsUpdate = true;
            resolve({ tex, skinColor, scleraColor });
        };
        img.onerror = () => reject(new Error("Failed to load texture image."));
        img.src = dataUrl;
    });
}

/**
 * Manages the reconstructed 478-vertex Three.js face mesh optionally welded to the
 * 36-vertex-socketed `head_shell.glb`, sculpt rig, skull retargeting, and Phong subdivision.
 */
export class ReconstructedFaceMesh {
    constructor() {
        this.group = new THREE.Group();
        this.fallbackTex = createPlaceholderTexture();
        this.material = createFaceMaterial(this.fallbackTex);
        this.geometry = new THREE.BufferGeometry();
        this.mesh = new THREE.Mesh(this.geometry, this.material);

        // Legacy procedural guide meshes kept only as fallback when head_shell.glb is absent
        this.guidesGroup = new THREE.Group();
        this.guideMat = new THREE.MeshStandardMaterial({
            color: 0x64748b,
            roughness: 0.65,
            metalness: 0.08,
            transparent: true,
            opacity: 0.68,
        });
        this.earMat = new THREE.MeshStandardMaterial({
            color: 0x38bdf8,
            roughness: 0.4,
            metalness: 0.15,
            transparent: true,
            opacity: 0.85,
        });
        this.guideNormalMat = new THREE.MeshNormalMaterial();
        this.guideDepthMat = createGuideDepthMaterial();

        this.neckGeo = new THREE.CylinderGeometry(0.16, 0.19, 0.44, 24);
        this.earGeo = new THREE.TorusGeometry(0.065, 0.018, 10, 24);
        this.neckMesh = new THREE.Mesh(this.neckGeo, this.guideMat);
        this.leftEarMesh = new THREE.Mesh(this.earGeo, this.earMat);
        this.rightEarMesh = new THREE.Mesh(this.earGeo, this.earMat);
        this.leftEarMesh.rotation.y = Math.PI / 2;
        this.rightEarMesh.rotation.y = Math.PI / 2;
        this.guidesGroup.add(this.neckMesh, this.leftEarMesh, this.rightEarMesh);

        this.group.add(this.mesh, this.guidesGroup);

        this.depthHullMaterial = createDepthHullMaterial();
        this.depthHullMesh = new THREE.Mesh(this.geometry, this.depthHullMaterial);
        this.depthHullMesh.visible = false;
        this.depthHullMesh.renderOrder = 0;
        this.depthHullMesh.frustumCulled = false;
        this.eyeballRig = new EyeballRig(this.material);
        this.weightDebugGeo = new THREE.BufferGeometry();
        this.weightDebugMat = new THREE.PointsMaterial({
            size: 6.0,
            sizeAttenuation: false,
            vertexColors: true,
            depthTest: true,
            depthWrite: false,
        });
        this.weightDebugPoints = new THREE.Points(this.weightDebugGeo, this.weightDebugMat);
        this.weightDebugPoints.visible = false;
        this.weightDebugPoints.renderOrder = 5;
        this.weightDebugPoints.frustumCulled = false;
        this.group.add(this.depthHullMesh, this.eyeballRig.group, this.weightDebugPoints);

        this.frontTexture = null;
        this.frontSkinColor = null;
        this.frontScleraColor = null;
        this.bakedTexture = null;
        this.bakedSkinColor = null;
        this.sourceMapTexture = null;
        this.hairMaskTexture = null;
        this.hasTexture = false;
        this.mvTexEnabled = true;
        this.uvDebugMode = "off";

        this.basePositions = null;
        this.baseIndices = null;
        this.baseUvs = null;
        this.canonicalUvs = null;
        this.restPositions = null;
        this.animScratch = null;
        this.mergedPositions = null;
        this.mergedUvs = null;
        this.mergedUvsFront = null;
        this.subdivUvsFront = null;
        this.mergedUvsCanonical = null;
        this.subdivUvsCanonical = null;
        this.mergedFaceMask = null;
        this.mergedIndices = null;
        this.mergedScratch = null;
        this.shellFitState = null;
        this.faceIndicesCount = 0;
        this.totalIndicesCount = 0;

        this.shellRig = null;
        this.shellFitState = null;
        this.openPoseRig = null;
        this.sculptRig = null;
        this.morphRig = null;
        this.subdivider = null;
        this.lastMotionFrame = null;
        this.latestActivePositions = null;
        this.openPoseDirty = false;
        this.landmarkWeights = null;
        this.landmarkWeightsByView = null;
        this.weightDebugStats = null;
        this.weightDebugVisible = false;
        this.weightDebugView = "combined";
        this.weightDebugDirty = false;

        this.sculptState = null;
        this.regionWeights = null;
        this.smoothEnabled = true;
        this.guidesVisible = true;
        this.hasMesh = false;
        this.viewMode = "clay";
        this.material.uniforms.uViewMode.value = 3;
        this.poseMode = "head_face";
        this.poseDotScale = 1.0;
        this.poseOcclude = true;
        this.poseOverlay = false;

        // Start warming cached GLB & canonical UV fetches immediately in browser environments
        if (typeof window !== "undefined" && typeof fetch === "function") {
            getHeadShellRig().then((rig) => {
                if (rig) this.shellRig = rig;
            });
            getOpenPoseRigData().then((data) => {
                if (data) this.attachOpenPoseData(data);
            });
            getCanonicalFaceData()
                .then((data) => {
                    if (data?.uv_front && !this.canonicalUvs) {
                        this.canonicalUvs = new Float32Array(data.uv_front);
                    }
                })
                .catch(() => {});
        }
    }

    attachOpenPoseData(data) {
        if (!data || this.openPoseRig) return;
        this.openPoseRig = new OpenPoseFaceRig(data);
        this.group.add(this.openPoseRig.group);
        this.openPoseDirty = true;
        this.syncPoseVisibility();
    }

    async updateFromReconstruction(meshData, frontImageUrl, overrideShellRig = undefined, overrideOpenPoseData = undefined) {
        this.basePositions = new Float32Array(meshData.positions);
        this.baseIndices = meshData.indices;
        this.baseUvs = new Float32Array(meshData.uv_front);
        this.animScratch = new Float32Array(this.basePositions.length);
        this.landmarkWeights = Array.isArray(meshData.landmark_weights)
            ? new Float32Array(meshData.landmark_weights)
            : null;
        if (meshData.landmark_weights_by_view) {
            const byView = meshData.landmark_weights_by_view;
            this.landmarkWeightsByView = {
                combined: Array.isArray(byView.combined) ? new Float32Array(byView.combined) : this.landmarkWeights,
                front: Array.isArray(byView.front) ? new Float32Array(byView.front) : null,
                three_quarter: Array.isArray(byView.three_quarter) ? new Float32Array(byView.three_quarter) : null,
                three_quarter_2: Array.isArray(byView.three_quarter_2) ? new Float32Array(byView.three_quarter_2) : null,
                side: Array.isArray(byView.side) ? new Float32Array(byView.side) : null,
                side_2: Array.isArray(byView.side_2) ? new Float32Array(byView.side_2) : null,
            };
        } else {
            this.landmarkWeightsByView = null;
        }
        this.weightDebugStats = meshData.weight_debug_stats || null;
        this.weightDebugDirty = true;

        this.sculptRig = new SculptRig(this.basePositions, this.baseIndices);
        this.restPositions = this.sculptRig.evaluate(this.sculptState);
        this.morphRig = new FaceRetargetRig(this.restPositions, this.baseIndices, this.regionWeights);

        if (overrideShellRig !== undefined) {
            this.shellRig = overrideShellRig;
        } else if (!this.shellRig && typeof window !== "undefined" && typeof fetch === "function") {
            this.shellRig = await getHeadShellRig();
        }

        if (overrideOpenPoseData !== undefined) {
            if (overrideOpenPoseData) this.attachOpenPoseData(overrideOpenPoseData);
        } else if (!this.openPoseRig && typeof window !== "undefined" && typeof fetch === "function") {
            const opData = await getOpenPoseRigData();
            if (opData) this.attachOpenPoseData(opData);
        }

        if (!this.canonicalUvs && typeof window !== "undefined" && typeof fetch === "function") {
            try {
                const cfData = await getCanonicalFaceData();
                if (cfData?.uv_front) {
                    this.canonicalUvs = new Float32Array(cfData.uv_front);
                }
            } catch {
                // Canonical UVs only needed when MV Tex baking is active
            }
        }

        if (this.shellRig) {
            const merged = this.shellRig.buildMergedRestMesh(
                this.restPositions,
                this.baseIndices,
                this.baseUvs
            );
            this.mergedPositions = merged.positions;
            this.mergedUvs = merged.uvs;
            this.mergedUvsFront = merged.uvs;
            this.mergedFaceMask = merged.faceMask;
            this.mergedIndices = merged.indices;
            this.mergedScratch = new Float32Array(merged.totalVerts * 3);
            this.shellFitState = merged.fitState;
            this.faceIndicesCount = merged.faceIndicesCount;
            this.totalIndicesCount = merged.indices.length;
            this.material.uniforms.uHasHeadShell.value = 1.0;

            this.subdivider = new MeshSubdivider(
                merged.totalVerts,
                merged.indices,
                merged.uvs,
                0.65,
                merged.faceMask
            );
        } else {
            const nVerts = Math.floor(this.basePositions.length / 3);
            this.mergedPositions = new Float32Array(this.restPositions);
            this.mergedUvs = this.baseUvs;
            this.mergedUvsFront = this.baseUvs;
            this.mergedFaceMask = new Float32Array(nVerts).fill(1.0);
            this.mergedIndices = this.baseIndices;
            this.mergedScratch = new Float32Array(this.basePositions.length);
            this.shellFitState = null;
            this.faceIndicesCount = this.baseIndices.length;
            this.totalIndicesCount = this.baseIndices.length;
            this.material.uniforms.uHasHeadShell.value = 0.0;

            this.subdivider = new MeshSubdivider(
                nVerts,
                this.baseIndices,
                this.baseUvs,
                0.65,
                this.mergedFaceMask
            );
        }
        this.subdivUvsFront = this.subdivider.subdivUvs;
        this.mergedUvsCanonical = null;
        this.subdivUvsCanonical = null;

        this.eyeballRig.configureRest(this.restPositions, this.baseUvs);
        this.lastMotionFrame = null;
        this.bindActiveTopology();
        this.uploadPositions(this.mergedPositions);
        this.geometry.computeBoundingSphere();

        await this.updateFrontTexture(frontImageUrl);
        this.updateGuideAnchors(meshData.anchors);
        this.hasMesh = true;
        this.applyGuideVisibility();
        this.syncPoseVisibility();
    }

    buildMergedUvs(faceUvs) {
        if (!faceUvs) return null;
        if (!this.shellRig) return faceUvs;
        const totalVerts = 478 + this.shellRig.nInterior;
        const faceCount = Math.min(478, Math.floor(faceUvs.length / 2));
        const out = new Float32Array(totalVerts * 2);
        out.set(faceUvs.subarray(0, faceCount * 2));
        for (let i = 0; i < this.shellRig.nInterior; i++) {
            const mp = this.shellRig.nearestSocketMp[i];
            if (mp >= 0 && mp * 2 + 1 < faceUvs.length) {
                out[(478 + i) * 2] = faceUvs[mp * 2];
                out[(478 + i) * 2 + 1] = faceUvs[mp * 2 + 1];
            } else {
                out[(478 + i) * 2] = 0.5;
                out[(478 + i) * 2 + 1] = 0.5;
            }
        }
        return out;
    }

    buildSubdivUvs(mergedUvs) {
        if (!mergedUvs || !this.subdivider) return null;
        const out = new Float32Array(this.subdivider.totalVerts * 2);
        out.set(mergedUvs);
        const edges = this.subdivider.edgePairs;
        for (let e = 0; e < edges.length; e += 3) {
            const a = edges[e];
            const b = edges[e + 1];
            const m = edges[e + 2];
            out[m * 2] = 0.5 * (mergedUvs[a * 2] + mergedUvs[b * 2]);
            out[m * 2 + 1] = 0.5 * (mergedUvs[a * 2 + 1] + mergedUvs[b * 2 + 1]);
        }
        return out;
    }

    isMultiViewTextureActive() {
        return Boolean(this.mvTexEnabled && this.bakedTexture && this.canonicalUvs);
    }

    hasBakedTextures() {
        return Boolean(this.bakedTexture);
    }

    getActiveFaceUvs() {
        return this.isMultiViewTextureActive() ? this.canonicalUvs : this.baseUvs;
    }

    ensureCanonicalMergedUvs() {
        if (!this.canonicalUvs) return;
        if (!this.mergedUvsCanonical) {
            this.mergedUvsCanonical = this.buildMergedUvs(this.canonicalUvs);
        }
        if (this.subdivider && !this.subdivUvsCanonical && this.mergedUvsCanonical) {
            this.subdivUvsCanonical = this.buildSubdivUvs(this.mergedUvsCanonical);
        }
    }

    getActiveDisplayUvs() {
        if (this.isMultiViewTextureActive()) {
            this.ensureCanonicalMergedUvs();
            const arr = this.smoothEnabled ? this.subdivUvsCanonical : this.mergedUvsCanonical;
            if (arr) return arr;
        }
        return this.smoothEnabled ? this.subdivUvsFront : this.mergedUvsFront;
    }

    bindActiveTopology() {
        if (!this.mergedPositions || !this.subdivider) return;
        const activeUvs = this.getActiveDisplayUvs();
        if (this.smoothEnabled) {
            const posCopy = new Float32Array(this.subdivider.subdivPositions.length);
            const uvCopy = new Float32Array(activeUvs || this.subdivider.subdivUvs);
            const maskCopy = new Float32Array(this.subdivider.subdivFaceMask);
            const scleraCopy = new Float32Array(this.subdivider.totalVerts).fill(-1.0);
            this.geometry.setAttribute("position", new THREE.BufferAttribute(posCopy, 3));
            this.geometry.setAttribute("uv", new THREE.BufferAttribute(uvCopy, 2));
            this.geometry.setAttribute("aFaceMask", new THREE.BufferAttribute(maskCopy, 1));
            this.geometry.setAttribute("aScleraBlend", new THREE.BufferAttribute(scleraCopy, 1));
            this.geometry.setIndex(this.subdivider.subdivIndices);
        } else {
            const posCopy = new Float32Array(this.mergedPositions.length);
            const uvCopy = new Float32Array(activeUvs || this.mergedUvs);
            const maskCopy = new Float32Array(this.mergedFaceMask);
            const scleraCopy = new Float32Array(this.mergedFaceMask.length).fill(-1.0);
            this.geometry.setAttribute("position", new THREE.BufferAttribute(posCopy, 3));
            this.geometry.setAttribute("uv", new THREE.BufferAttribute(uvCopy, 2));
            this.geometry.setAttribute("aFaceMask", new THREE.BufferAttribute(maskCopy, 1));
            this.geometry.setAttribute("aScleraBlend", new THREE.BufferAttribute(scleraCopy, 1));
            this.geometry.setIndex(this.mergedIndices);
        }
        this.applyDrawRange();
    }

    applyDrawRange() {
        if (!this.totalIndicesCount) return;
        const mult = this.smoothEnabled ? 4 : 1;
        const count = this.guidesVisible ? this.totalIndicesCount * mult : this.faceIndicesCount * mult;
        this.geometry.setDrawRange(0, count);
    }

    applyGuideVisibility() {
        const isPose = this.viewMode === "pose";
        this.guidesGroup.visible = !isPose && !this.shellRig && this.guidesVisible;
        this.applyDrawRange();
        if (this.openPoseRig && (isPose || this.poseOverlay)) {
            this.openPoseRig.setOptions({ guidesVisible: this.guidesVisible });
        } else {
            this.openPoseDirty = true;
        }
    }

    uploadPositions(mergedOrFacePos) {
        const posAttr = this.geometry.getAttribute("position");
        if (!posAttr) return;
        if (this.smoothEnabled && this.subdivider) {
            const subdivPos = this.subdivider.evaluate(mergedOrFacePos);
            posAttr.array.set(subdivPos);
        } else {
            posAttr.array.set(mergedOrFacePos);
        }
        posAttr.needsUpdate = true;
        this.geometry.computeVertexNormals();
        this.latestActivePositions = mergedOrFacePos;
        this.eyeballRig.updateFromPositions(mergedOrFacePos);

        if (this.openPoseRig && (this.viewMode === "pose" || this.poseOverlay)) {
            if (mergedOrFacePos.length >= 455 * 3) {
                const spanX = Math.max(Math.abs(mergedOrFacePos[454 * 3] - mergedOrFacePos[234 * 3]), 0.1);
                const cx = 0.5 * (mergedOrFacePos[234 * 3] + mergedOrFacePos[454 * 3]);
                const cy = 0.5 * (mergedOrFacePos[10 * 3 + 1] + mergedOrFacePos[152 * 3 + 1]);
                const czTemple = 0.5 * (mergedOrFacePos[234 * 3 + 2] + mergedOrFacePos[454 * 3 + 2]);
                const cz = czTemple - (this.guidesVisible && this.shellRig ? 0.14 : 0.18) * spanX;
                this.depthHullMaterial.uniforms.uHullCenter.value.set(cx, cy, cz);
            }
            this.openPoseRig.updateFromPositions(mergedOrFacePos, this.shellRig, this.guidesVisible);
            this.openPoseDirty = false;
        } else {
            this.openPoseDirty = true;
        }

        if (this.weightDebugVisible && this.viewMode !== "pose") {
            this.syncWeightDebugOverlay(mergedOrFacePos);
        } else {
            this.weightDebugDirty = true;
        }
    }

    setWeightDebugVisible(visible) {
        this.weightDebugVisible = Boolean(visible);
        this.syncPoseVisibility();
    }

    setWeightDebugView(viewKey) {
        const valid = ["combined", "front", "three_quarter", "three_quarter_2", "side", "side_2"];
        const next = valid.includes(viewKey) ? viewKey : "combined";
        if (next === this.weightDebugView) return;
        this.weightDebugView = next;
        this.weightDebugDirty = true;
        this.syncPoseVisibility();
    }

    syncWeightDebugOverlay(positions = this.latestActivePositions) {
        if (!positions || !this.basePositions) return;
        const nPts = Math.min(478, Math.floor(this.basePositions.length / 3), Math.floor(positions.length / 3));
        if (nPts <= 0) return;

        let posAttr = this.weightDebugGeo.getAttribute("position");
        let colAttr = this.weightDebugGeo.getAttribute("color");
        if (!posAttr || posAttr.count !== nPts) {
            posAttr = new THREE.BufferAttribute(new Float32Array(nPts * 3), 3);
            colAttr = new THREE.BufferAttribute(new Float32Array(nPts * 3), 3);
            this.weightDebugGeo.setAttribute("position", posAttr);
            this.weightDebugGeo.setAttribute("color", colAttr);
        }

        posAttr.array.set(positions.subarray(0, nPts * 3));
        posAttr.needsUpdate = true;

        const cols = colAttr.array;
        const wArr = this.landmarkWeightsByView?.[this.weightDebugView] || this.landmarkWeights;
        for (let i = 0; i < nPts; i++) {
            const w = wArr && i < wArr.length ? wArr[i] : 1.0;
            if (w > 0.70) {
                _dbgColor.setHex(0x22c55e); // Green: high reliability
            } else if (w >= 0.30) {
                _dbgColor.setHex(0xfacc15); // Yellow: moderate reliability
            } else if (w >= 0.05) {
                _dbgColor.setHex(0xef4444); // Red: low reliability / boundary
            } else {
                _dbgColor.setHex(0x94a3b8); // Grey: occluded / unreliable
            }
            const o = i * 3;
            cols[o] = _dbgColor.r;
            cols[o + 1] = _dbgColor.g;
            cols[o + 2] = _dbgColor.b;
        }
        colAttr.needsUpdate = true;
        this.weightDebugGeo.computeBoundingSphere();
        this.weightDebugDirty = false;
    }

    uploadFaceFrame(facePos) {
        if (this.shellRig && this.shellFitState && this.mergedScratch) {
            this.shellRig.deformInto(facePos, this.shellFitState, this.mergedScratch);
            this.uploadPositions(this.mergedScratch);
        } else {
            this.uploadPositions(facePos);
        }
    }

    setSmoothEnabled(enabled) {
        const next = Boolean(enabled);
        if (next === this.smoothEnabled) return;
        this.smoothEnabled = next;
        if (!this.hasMesh) return;
        this.bindActiveTopology();
        if (this.lastMotionFrame) {
            this.applyMotionFrame(this.lastMotionFrame);
        } else {
            this.uploadFaceFrame(this.restPositions);
        }
    }

    setSculptState(sculptState) {
        const next = sculptState ? { ...sculptState } : null;
        if (this.hasMesh && this.sculptRig && this.sculptState && next) {
            const keysA = Object.keys(this.sculptState);
            const keysB = Object.keys(next);
            if (keysA.length === keysB.length && keysB.every((k) => Number(this.sculptState[k]) === Number(next[k]))) {
                return;
            }
        }
        this.sculptState = next;
        if (!this.hasMesh || !this.sculptRig) return;
        this.restPositions = this.sculptRig.evaluate(this.sculptState);
        this.eyeballRig.configureRest(this.restPositions, this.getActiveFaceUvs());
        if (this.morphRig) {
            this.morphRig.updateRestPositions(this.restPositions);
        } else {
            this.morphRig = new FaceRetargetRig(this.restPositions, this.baseIndices, this.regionWeights);
        }
        if (this.lastMotionFrame) {
            this.applyMotionFrame(this.lastMotionFrame);
        } else {
            this.uploadFaceFrame(this.restPositions);
        }
    }

    setIrisMaskScale(scale) {
        const val = Number(scale);
        if (Number.isFinite(val)) {
            this.material.uniforms.uIrisMaskScale.value = val;
        }
    }

    setRegionWeights(weights) {
        this.regionWeights = { ...weights };
        this.morphRig?.setRegionWeights(this.regionWeights);
        if (this.hasMesh && this.lastMotionFrame) {
            this.applyMotionFrame(this.lastMotionFrame);
        }
    }

    syncActiveTextureAndUvs() {
        const useBaked = this.isMultiViewTextureActive();
        const activeFaceUvs = this.getActiveFaceUvs();
        if (this.restPositions && activeFaceUvs) {
            this.eyeballRig.configureRest(this.restPositions, activeFaceUvs);
        }

        const uvAttr = this.geometry.getAttribute("uv");
        const displayUvs = this.getActiveDisplayUvs();
        if (uvAttr && displayUvs && uvAttr.array.length === displayUvs.length) {
            uvAttr.array.set(displayUvs);
            uvAttr.needsUpdate = true;
        }

        const u = this.material.uniforms;
        if (this.frontTexture && this.frontScleraColor) {
            const sc = this.frontScleraColor;
            u.uScleraColor.value.setRGB(sc[0], sc[1], sc[2], THREE.SRGBColorSpace);
        } else {
            u.uScleraColor.value.setHex(DEFAULT_SCLERA_HEX);
        }

        if (!useBaked) {
            u.uFrontTex.value = this.frontTexture || this.fallbackTex;
            u.uHasTexture.value = this.hasTexture ? 1.0 : 0.0;
            const c = this.frontTexture && this.frontSkinColor
                ? this.frontSkinColor
                : [216 / 255, 180 / 255, 160 / 255];
            u.uSkinColor.value.setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);
            return;
        }

        let activeTex = this.bakedTexture;
        let skin = this.bakedSkinColor || this.frontSkinColor || [216 / 255, 180 / 255, 160 / 255];
        let hasActivePortraitTex = this.hasTexture || Boolean(this.bakedTexture);
        if (this.uvDebugMode === "source" && this.sourceMapTexture) {
            activeTex = this.sourceMapTexture;
            skin = [55 / 255, 65 / 255, 81 / 255];
            hasActivePortraitTex = false;
        } else if (this.uvDebugMode === "hair" && this.hairMaskTexture) {
            activeTex = this.hairMaskTexture;
            skin = [55 / 255, 65 / 255, 81 / 255];
            hasActivePortraitTex = false;
        }
        u.uFrontTex.value = activeTex;
        u.uHasTexture.value = hasActivePortraitTex ? 1.0 : 0.0;
        u.uSkinColor.value.setRGB(skin[0], skin[1], skin[2], THREE.SRGBColorSpace);
    }

    async updateFrontTexture(frontImageUrl) {
        const { tex, skinColor, scleraColor } = await loadTextureFromDataUrl(
            frontImageUrl,
            this.baseUvs,
            THREE.LinearFilter,
            true
        );
        this.disposeTextures();
        this.frontTexture = tex;
        this.frontSkinColor = skinColor;
        this.frontScleraColor = scleraColor;
        this.hasTexture = Boolean(tex);
        this.syncActiveTextureAndUvs();
    }

    async setBakedTextures({ bakedAlbedoUrl, sourceMapUrl, hairMaskUrl }, overrideCanonicalUvs = null) {
        if (overrideCanonicalUvs) {
            this.canonicalUvs = overrideCanonicalUvs instanceof Float32Array
                ? overrideCanonicalUvs
                : new Float32Array(overrideCanonicalUvs);
            this.mergedUvsCanonical = null;
            this.subdivUvsCanonical = null;
        } else if (!this.canonicalUvs && typeof window !== "undefined" && typeof fetch === "function") {
            const cfData = await getCanonicalFaceData();
            if (cfData?.uv_front) {
                this.canonicalUvs = new Float32Array(cfData.uv_front);
            }
        }

        const [bakedRes, sourceRes, hairRes] = await Promise.all([
            loadTextureFromDataUrl(bakedAlbedoUrl, this.canonicalUvs, THREE.LinearFilter),
            loadTextureFromDataUrl(sourceMapUrl, null, THREE.NearestFilter),
            loadTextureFromDataUrl(hairMaskUrl, null, THREE.NearestFilter),
        ]);

        this.disposeBakedTextures();
        this.bakedTexture = bakedRes.tex;
        this.bakedSkinColor = bakedRes.skinColor;
        this.sourceMapTexture = sourceRes.tex;
        this.hairMaskTexture = hairRes.tex;
        this.mergedUvsCanonical = null;
        this.subdivUvsCanonical = null;
        if (this.hasMesh) {
            this.syncActiveTextureAndUvs();
        }
    }

    clearBakedTextures() {
        this.disposeBakedTextures();
        if (this.hasMesh) {
            this.syncActiveTextureAndUvs();
        }
    }

    setMvTexEnabled(enabled) {
        const next = Boolean(enabled);
        if (next === this.mvTexEnabled) return;
        this.mvTexEnabled = next;
        if (!next) {
            this.uvDebugMode = "off";
        }
        if (this.hasMesh) {
            this.syncActiveTextureAndUvs();
        }
    }

    setUvDebugMode(mode) {
        const valid = ["off", "source", "hair"];
        const next = valid.includes(mode) ? mode : "off";
        if (next === this.uvDebugMode) return;
        this.uvDebugMode = next;
        if (this.hasMesh) {
            this.syncActiveTextureAndUvs();
        }
    }

    updateGuideAnchors(anchors) {
        if (!anchors) return;
        const [lx, ly, lz] = anchors.left_temple;
        const [rx, ry, rz] = anchors.right_temple;
        const [cx, cy] = anchors.chin;

        this.leftEarMesh.position.set(lx, ly, lz);
        this.rightEarMesh.position.set(rx, ry, rz);
        const centerZ = 0.5 * (lz + rz) - 0.04;
        this.neckMesh.position.set(cx, cy - 0.16, centerZ);
    }

    applyMotionFrame(motionFrame) {
        if (!this.hasMesh || !this.morphRig || !motionFrame) return;
        this.lastMotionFrame = motionFrame;

        if (motionFrame.quat) {
            const [qx, qy, qz, qw] = motionFrame.quat;
            this.group.quaternion.set(qx, qy, qz, qw);
        }

        this.morphRig.evaluate(motionFrame, this.animScratch);
        this.uploadFaceFrame(this.animScratch);
    }

    resetPose() {
        this.lastMotionFrame = null;
        this.group.rotation.set(0, 0, 0);
        this.group.quaternion.identity();
        if (!this.hasMesh || !this.restPositions) return;
        this.uploadFaceFrame(this.restPositions);
    }

    setTurntableYaw(yawRad, baseQuat = null) {
        _yawQuat.setFromAxisAngle(_yAxis, yawRad);
        if (baseQuat && baseQuat.length === 4) {
            _actorQuat.set(baseQuat[0], baseQuat[1], baseQuat[2], baseQuat[3]);
            this.group.quaternion.copy(_yawQuat).multiply(_actorQuat);
        } else {
            this.group.quaternion.copy(_yawQuat);
        }
    }

    updateDepthBounds(camera, dynamicMode = false) {
        if (!camera || !this.geometry.boundingSphere) return;
        this.group.updateMatrixWorld(true);

        let near;
        let far;
        if (dynamicMode) {
            const posAttr = this.geometry.getAttribute("position");
            if (!posAttr) return;
            const arr = posAttr.array;
            const mv = new THREE.Matrix4().multiplyMatrices(camera.matrixWorldInverse, this.mesh.matrixWorld);
            const pt = new THREE.Vector3();
            let minZ = Infinity;
            let maxZ = -Infinity;
            // Face vertices always occupy indices 0..477 in both base and Phong-subdivided buffers
            const vLimit = this.guidesVisible ? posAttr.count : Math.min(posAttr.count, 478);
            for (let i = 0; i < vLimit; i++) {
                pt.set(arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]).applyMatrix4(mv);
                const vz = -pt.z;
                if (vz < minZ) minZ = vz;
                if (vz > maxZ) maxZ = vz;
            }
            near = Math.max(0.05, minZ - 0.005);
            far = Math.max(near + 0.02, maxZ + 0.005);
        } else {
            _worldCenter.copy(this.geometry.boundingSphere.center).applyMatrix4(this.group.matrixWorld);
            const dist = camera.position.distanceTo(_worldCenter);
            const r = Math.max(this.geometry.boundingSphere.radius, 0.25);
            near = Math.max(0.05, dist - r * 1.05);
            far = Math.max(near + 0.2, dist + r * 1.05);
        }

        this.material.uniforms.uDepthNear.value = near;
        this.material.uniforms.uDepthFar.value = far;
        this.guideDepthMat.uniforms.uDepthNear.value = near;
        this.guideDepthMat.uniforms.uDepthFar.value = far;
        return { near, far };
    }

    applyRenderState(rs) {
        if (!rs) return;
        const u = this.material.uniforms;
        u.uDepthInvert.value = rs.depthInvert ? 1.0 : 0.0;
        u.uNormalFlip.value.set(rs.flipX ? -1.0 : 1.0, rs.flipY ? -1.0 : 1.0, rs.flipZ ? -1.0 : 1.0);
        u.uAlphaMode.value = rs.maskMode === "face" ? 1.0 : 0.0;

        const az = THREE.MathUtils.degToRad(rs.clayKeyAz ?? 30);
        const el = THREE.MathUtils.degToRad(rs.clayKeyEl ?? 35);
        u.uClayKeyDir.value.set(
            Math.cos(el) * Math.sin(az),
            Math.sin(el),
            Math.cos(el) * Math.cos(az)
        ).normalize();
        u.uClayKeyIntensity.value = Number(rs.clayKeyPower ?? 0.85);
        u.uClayFillIntensity.value = Number(rs.clayFill ?? 0.35);

        this.guideDepthMat.uniforms.uDepthInvert.value = u.uDepthInvert.value;

        this.poseMode = rs.poseMode || "head_face";
        this.poseDotScale = Number(rs.poseDotScale ?? 1.0);
        this.poseOcclude = rs.poseOcclude !== false;
        this.poseOverlay = Boolean(rs.poseOverlay);
        this.syncPoseVisibility();
    }

    syncPoseVisibility() {
        const isPose = this.viewMode === "pose";
        this.mesh.visible = !isPose;
        this.eyeballRig.group.visible = !isPose && this.hasMesh;
        this.guidesGroup.visible = !isPose && !this.shellRig && this.guidesVisible;
        this.depthHullMesh.visible = isPose && this.hasMesh && this.poseOcclude;

        const showWeightDebug = !isPose && this.hasMesh && this.weightDebugVisible;
        this.weightDebugPoints.visible = showWeightDebug;
        if (showWeightDebug && this.weightDebugDirty && this.latestActivePositions) {
            this.syncWeightDebugOverlay(this.latestActivePositions);
        }

        if (this.openPoseRig) {
            const needPose = this.hasMesh && (isPose || this.poseOverlay);
            this.openPoseRig.group.visible = needPose;
            if (needPose && this.openPoseDirty && this.latestActivePositions) {
                if (this.latestActivePositions.length >= 455 * 3) {
                    const p = this.latestActivePositions;
                    const spanX = Math.max(Math.abs(p[454 * 3] - p[234 * 3]), 0.1);
                    const cx = 0.5 * (p[234 * 3] + p[454 * 3]);
                    const cy = 0.5 * (p[10 * 3 + 1] + p[152 * 3 + 1]);
                    const czTemple = 0.5 * (p[234 * 3 + 2] + p[454 * 3 + 2]);
                    const cz = czTemple - (this.guidesVisible && this.shellRig ? 0.14 : 0.18) * spanX;
                    this.depthHullMaterial.uniforms.uHullCenter.value.set(cx, cy, cz);
                }
                this.openPoseRig.updateFromPositions(this.latestActivePositions, this.shellRig, this.guidesVisible);
                this.openPoseDirty = false;
            }
            if (needPose) {
                this.openPoseRig.setOptions({
                    poseMode: this.poseMode,
                    dotScale: this.poseDotScale,
                    occlude: this.poseOcclude,
                    guidesVisible: this.guidesVisible,
                    overlay: !isPose && this.poseOverlay,
                });
            }
        }
    }

    getWorldBoundingBox() {
        this.group.updateMatrixWorld(true);
        const box = new THREE.Box3();
        const posAttr = this.geometry.getAttribute("position");
        if (this.hasMesh && posAttr) {
            if (this.guidesVisible) {
                this.geometry.computeBoundingBox();
                if (this.geometry.boundingBox) {
                    box.copy(this.geometry.boundingBox).applyMatrix4(this.mesh.matrixWorld);
                    return box;
                }
            } else {
                const pt = new THREE.Vector3();
                const nFace = Math.min(posAttr.count, 478);
                for (let i = 0; i < nFace; i++) {
                    pt.fromBufferAttribute(posAttr, i).applyMatrix4(this.mesh.matrixWorld);
                    box.expandByPoint(pt);
                }
                if (!box.isEmpty()) return box;
            }
        }
        box.setFromObject(this.group);
        return box;
    }

    updateLights({ keyAzimuth, keyElevation, keyIntensity, keyColor, rimAzimuth, rimElevation, rimIntensity, rimColor, ambientIntensity }) {
        const toDir = (azDeg, elDeg) => {
            const az = THREE.MathUtils.degToRad(azDeg);
            const el = THREE.MathUtils.degToRad(elDeg);
            return new THREE.Vector3(
                Math.cos(el) * Math.sin(az),
                Math.sin(el),
                Math.cos(el) * Math.cos(az)
            ).normalize();
        };

        this.material.uniforms.uKeyDir.value.copy(toDir(keyAzimuth, keyElevation));
        this.material.uniforms.uKeyIntensity.value = keyIntensity;
        this.material.uniforms.uKeyColor.value.set(keyColor);

        this.material.uniforms.uRimDir.value.copy(toDir(rimAzimuth, rimElevation));
        this.material.uniforms.uRimIntensity.value = rimIntensity;
        this.material.uniforms.uRimColor.value.set(rimColor);

        this.material.uniforms.uAmbientIntensity.value = ambientIntensity;
    }

    setViewMode(mode) {
        this.viewMode = mode;
        const isWire = mode === "wireframe";
        const modeInt = mode === "normals" ? 1 : mode === "depth" ? 2 : mode === "clay" ? 3 : 0;

        this.material.uniforms.uViewMode.value = modeInt;
        this.material.wireframe = isWire;

        if (mode === "normals") {
            this.neckMesh.material = this.guideNormalMat;
            this.leftEarMesh.material = this.guideNormalMat;
            this.rightEarMesh.material = this.guideNormalMat;
        } else if (mode === "depth") {
            this.neckMesh.material = this.guideDepthMat;
            this.leftEarMesh.material = this.guideDepthMat;
            this.rightEarMesh.material = this.guideDepthMat;
        } else {
            this.neckMesh.material = this.guideMat;
            this.leftEarMesh.material = this.earMat;
            this.rightEarMesh.material = this.earMat;
            this.guideMat.wireframe = isWire;
            this.earMat.wireframe = isWire;
        }
        this.syncPoseVisibility();
    }

    setGuidesVisible(visible) {
        this.guidesVisible = Boolean(visible);
        this.applyGuideVisibility();
    }

    disposeBakedTextures() {
        if (this.bakedTexture) {
            this.bakedTexture.dispose();
            this.bakedTexture = null;
        }
        if (this.sourceMapTexture) {
            this.sourceMapTexture.dispose();
            this.sourceMapTexture = null;
        }
        if (this.hairMaskTexture) {
            this.hairMaskTexture.dispose();
            this.hairMaskTexture = null;
        }
        this.bakedSkinColor = null;
    }

    disposeTextures() {
        this.disposeBakedTextures();
        if (this.frontTexture) {
            this.frontTexture.dispose();
            this.frontTexture = null;
        }
        this.frontSkinColor = null;
        this.frontScleraColor = null;
        this.hasTexture = false;
        this.material.uniforms.uFrontTex.value = this.fallbackTex;
        this.material.uniforms.uHasTexture.value = 0.0;
        this.material.uniforms.uScleraColor.value.setHex(DEFAULT_SCLERA_HEX);
        this.material.uniforms.uSkinColor.value.setRGB(
            216 / 255,
            180 / 255,
            160 / 255,
            THREE.SRGBColorSpace
        );
    }

    clear(resetParams = false) {
        this.disposeTextures();
        this.geometry.deleteAttribute("position");
        this.geometry.deleteAttribute("uv");
        this.geometry.deleteAttribute("aFaceMask");
        this.geometry.deleteAttribute("aScleraBlend");
        this.geometry.deleteAttribute("normal");
        this.geometry.setIndex([]);
        this.geometry.setDrawRange(0, Infinity);
        this.geometry.boundingSphere = null;
        this.geometry.boundingBox = null;

        this.basePositions = null;
        this.baseIndices = null;
        this.baseUvs = null;
        this.restPositions = null;
        this.animScratch = null;
        this.mergedPositions = null;
        this.mergedUvs = null;
        this.mergedUvsFront = null;
        this.subdivUvsFront = null;
        this.mergedUvsCanonical = null;
        this.subdivUvsCanonical = null;
        this.mergedFaceMask = null;
        this.mergedIndices = null;
        this.mergedScratch = null;
        this.shellFitState = null;
        this.faceIndicesCount = 0;
        this.totalIndicesCount = 0;

        this.sculptRig = null;
        this.morphRig = null;
        this.subdivider = null;
        this.lastMotionFrame = null;
        this.latestActivePositions = null;
        this.openPoseDirty = false;
        this.weightDebugGeo.deleteAttribute("position");
        this.weightDebugGeo.deleteAttribute("color");
        this.weightDebugGeo.boundingSphere = null;
        this.landmarkWeights = null;
        this.landmarkWeightsByView = null;
        this.weightDebugStats = null;
        this.weightDebugDirty = false;
        if (resetParams) {
            this.sculptState = null;
            this.regionWeights = null;
            this.weightDebugVisible = false;
            this.weightDebugView = "combined";
            this.mvTexEnabled = true;
            this.uvDebugMode = "off";
        }
        this.group.rotation.set(0, 0, 0);
        this.group.quaternion.identity();
        this.hasMesh = false;
        this.eyeballRig.clear();
        this.syncPoseVisibility();
    }

    dispose() {
        this.disposeTextures();
        this.fallbackTex.dispose();
        this.geometry.dispose();
        this.material.dispose();
        this.depthHullMaterial.dispose();
        this.weightDebugGeo.dispose();
        this.weightDebugMat.dispose();
        this.eyeballRig.dispose();
        this.openPoseRig?.dispose();
        this.neckGeo.dispose();
        this.earGeo.dispose();
        this.guideMat.dispose();
        this.earMat.dispose();
        this.guideNormalMat.dispose();
        this.guideDepthMat.dispose();
    }
}
