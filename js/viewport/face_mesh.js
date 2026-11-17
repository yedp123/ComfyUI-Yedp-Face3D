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

const _worldCenter = new THREE.Vector3();
const SKIN_SAMPLE_LANDMARKS = [50, 280, 117, 346, 151, 9, 195];

let _headShellRigPromise = null;
let _openPoseRigDataPromise = null;

function getHeadShellRig() {
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

function loadTextureFromDataUrl(dataUrl, uvs) {
    return new Promise((resolve, reject) => {
        if (!dataUrl) {
            resolve({ tex: null, skinColor: null });
            return;
        }
        const img = new Image();
        img.onload = () => {
            const skinColor = uvs ? sampleMedianSkinColor(img, uvs) : null;
            const tex = new THREE.Texture(img);
            tex.colorSpace = THREE.SRGBColorSpace;
            tex.flipY = true;
            tex.wrapS = THREE.ClampToEdgeWrapping;
            tex.wrapT = THREE.ClampToEdgeWrapping;
            tex.minFilter = THREE.LinearFilter;
            tex.magFilter = THREE.LinearFilter;
            tex.needsUpdate = true;
            resolve({ tex, skinColor });
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
        this.group.add(this.depthHullMesh);

        this.frontTexture = null;

        this.basePositions = null;
        this.baseIndices = null;
        this.baseUvs = null;
        this.restPositions = null;
        this.animScratch = null;
        this.mergedPositions = null;
        this.mergedUvs = null;
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

        this.sculptState = null;
        this.regionWeights = null;
        this.smoothEnabled = true;
        this.guidesVisible = true;
        this.hasMesh = false;
        this.viewMode = "textured";
        this.poseMode = "head_face";
        this.poseDotScale = 1.0;
        this.poseOcclude = true;
        this.poseOverlay = false;

        // Start warming cached GLB fetches immediately in browser environments
        if (typeof window !== "undefined" && typeof fetch === "function") {
            getHeadShellRig().then((rig) => {
                if (rig) this.shellRig = rig;
            });
            getOpenPoseRigData().then((data) => {
                if (data) this.attachOpenPoseData(data);
            });
        }
    }

    attachOpenPoseData(data) {
        if (!data || this.openPoseRig) return;
        this.openPoseRig = new OpenPoseFaceRig(data);
        this.group.add(this.openPoseRig.group);
        if (this.hasMesh) {
            const activePos = this.lastMotionFrame
                ? (this.shellRig ? this.mergedScratch : this.animScratch)
                : this.mergedPositions;
            this.openPoseRig.updateFromPositions(activePos || this.mergedPositions, this.shellRig, this.guidesVisible);
        }
        this.syncPoseVisibility();
    }

    async updateFromReconstruction(meshData, frontImageUrl, overrideShellRig = undefined, overrideOpenPoseData = undefined) {
        this.basePositions = new Float32Array(meshData.positions);
        this.baseIndices = meshData.indices;
        this.baseUvs = new Float32Array(meshData.uv_front);
        this.animScratch = new Float32Array(this.basePositions.length);

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

        if (this.shellRig) {
            const merged = this.shellRig.buildMergedRestMesh(
                this.restPositions,
                this.baseIndices,
                this.baseUvs
            );
            this.mergedPositions = merged.positions;
            this.mergedUvs = merged.uvs;
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

    bindActiveTopology() {
        if (!this.mergedPositions || !this.subdivider) return;
        if (this.smoothEnabled) {
            const posCopy = new Float32Array(this.subdivider.subdivPositions.length);
            const uvCopy = new Float32Array(this.subdivider.subdivUvs);
            const maskCopy = new Float32Array(this.subdivider.subdivFaceMask);
            this.geometry.setAttribute("position", new THREE.BufferAttribute(posCopy, 3));
            this.geometry.setAttribute("uv", new THREE.BufferAttribute(uvCopy, 2));
            this.geometry.setAttribute("aFaceMask", new THREE.BufferAttribute(maskCopy, 1));
            this.geometry.setIndex(this.subdivider.subdivIndices);
        } else {
            const posCopy = new Float32Array(this.mergedPositions.length);
            const uvCopy = new Float32Array(this.mergedUvs);
            const maskCopy = new Float32Array(this.mergedFaceMask);
            this.geometry.setAttribute("position", new THREE.BufferAttribute(posCopy, 3));
            this.geometry.setAttribute("uv", new THREE.BufferAttribute(uvCopy, 2));
            this.geometry.setAttribute("aFaceMask", new THREE.BufferAttribute(maskCopy, 1));
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
        this.openPoseRig?.setOptions({ guidesVisible: this.guidesVisible });
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

        if (mergedOrFacePos.length >= 455 * 3) {
            const spanX = Math.max(Math.abs(mergedOrFacePos[454 * 3] - mergedOrFacePos[234 * 3]), 0.1);
            const cx = 0.5 * (mergedOrFacePos[234 * 3] + mergedOrFacePos[454 * 3]);
            const cy = 0.5 * (mergedOrFacePos[10 * 3 + 1] + mergedOrFacePos[152 * 3 + 1]);
            const czTemple = 0.5 * (mergedOrFacePos[234 * 3 + 2] + mergedOrFacePos[454 * 3 + 2]);
            const cz = czTemple - (this.guidesVisible && this.shellRig ? 0.14 : 0.18) * spanX;
            this.depthHullMaterial.uniforms.uHullCenter.value.set(cx, cy, cz);
        }
        this.openPoseRig?.updateFromPositions(mergedOrFacePos, this.shellRig, this.guidesVisible);
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
        this.sculptState = { ...sculptState };
        if (!this.hasMesh || !this.sculptRig) return;
        this.restPositions = this.sculptRig.evaluate(this.sculptState);
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

    setRegionWeights(weights) {
        this.regionWeights = { ...weights };
        this.morphRig?.setRegionWeights(this.regionWeights);
        if (this.hasMesh && this.lastMotionFrame) {
            this.applyMotionFrame(this.lastMotionFrame);
        }
    }

    async updateFrontTexture(frontImageUrl) {
        const { tex, skinColor } = await loadTextureFromDataUrl(frontImageUrl, this.baseUvs);
        this.disposeTextures();
        this.frontTexture = tex;
        this.material.uniforms.uFrontTex.value = tex || this.fallbackTex;
        if (skinColor) {
            this.material.uniforms.uSkinColor.value.setRGB(
                skinColor[0],
                skinColor[1],
                skinColor[2],
                THREE.SRGBColorSpace
            );
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

    setTurntableYaw(yawRad) {
        this.group.quaternion.identity();
        this.group.rotation.set(0, yawRad, 0);
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
        this.guidesGroup.visible = !isPose && !this.shellRig && this.guidesVisible;
        this.depthHullMesh.visible = isPose && this.hasMesh && this.poseOcclude;

        if (this.openPoseRig) {
            this.openPoseRig.group.visible = this.hasMesh && (isPose || this.poseOverlay);
            this.openPoseRig.setOptions({
                poseMode: this.poseMode,
                dotScale: this.poseDotScale,
                occlude: this.poseOcclude,
                guidesVisible: this.guidesVisible,
                overlay: !isPose && this.poseOverlay,
            });
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

    disposeTextures() {
        if (this.frontTexture) {
            this.frontTexture.dispose();
            this.frontTexture = null;
        }
        this.material.uniforms.uFrontTex.value = this.fallbackTex;
    }

    clear(resetParams = false) {
        this.disposeTextures();
        this.geometry.deleteAttribute("position");
        this.geometry.deleteAttribute("uv");
        this.geometry.deleteAttribute("aFaceMask");
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
        if (resetParams) {
            this.sculptState = null;
            this.regionWeights = null;
        }
        this.group.rotation.set(0, 0, 0);
        this.group.quaternion.identity();
        this.hasMesh = false;
        this.syncPoseVisibility();
    }

    dispose() {
        this.disposeTextures();
        this.fallbackTex.dispose();
        this.geometry.dispose();
        this.material.dispose();
        this.depthHullMaterial.dispose();
        this.openPoseRig?.dispose();
        this.neckGeo.dispose();
        this.earGeo.dispose();
        this.guideMat.dispose();
        this.earMat.dispose();
        this.guideNormalMat.dispose();
        this.guideDepthMat.dispose();
    }
}
