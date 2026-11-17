import * as THREE from "../lib/three.module.js";
import {
    createFaceMaterial,
    createGuideDepthMaterial,
    createPlaceholderTexture,
} from "./face_shader.js";
import { FaceRetargetRig } from "../vision/retarget_rig.js";
import { SculptRig } from "../vision/sculpt_rig.js";
import { MeshSubdivider } from "./mesh_subdiv.js";

const _worldCenter = new THREE.Vector3();

function loadTextureFromDataUrl(dataUrl) {
    return new Promise((resolve, reject) => {
        if (!dataUrl) {
            resolve(null);
            return;
        }
        const img = new Image();
        img.onload = () => {
            const tex = new THREE.Texture(img);
            tex.colorSpace = THREE.SRGBColorSpace;
            tex.flipY = true;
            tex.wrapS = THREE.ClampToEdgeWrapping;
            tex.wrapT = THREE.ClampToEdgeWrapping;
            tex.minFilter = THREE.LinearFilter;
            tex.magFilter = THREE.LinearFilter;
            tex.needsUpdate = true;
            resolve(tex);
        };
        img.onerror = () => reject(new Error("Failed to load texture image."));
        img.src = dataUrl;
    });
}

/**
 * Manages the reconstructed 478-vertex Three.js face mesh, sculpt push/pull rig,
 * intrinsic skull retargeting, optional 1-level Phong subdivision, view modes, and guides.
 */
export class ReconstructedFaceMesh {
    constructor() {
        this.group = new THREE.Group();
        this.fallbackTex = createPlaceholderTexture();
        this.material = createFaceMaterial(this.fallbackTex);
        this.geometry = new THREE.BufferGeometry();
        this.mesh = new THREE.Mesh(this.geometry, this.material);

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
        this.frontTexture = null;

        this.basePositions = null;
        this.baseIndices = null;
        this.baseUvs = null;
        this.restPositions = null;
        this.animScratch = null;
        this.sculptRig = null;
        this.morphRig = null;
        this.subdivider = null;
        this.lastMotionFrame = null;

        this.sculptState = null;
        this.regionWeights = null;
        this.smoothEnabled = true;
        this.hasMesh = false;
        this.viewMode = "textured";
    }

    async updateFromReconstruction(meshData, frontImageUrl) {
        this.basePositions = new Float32Array(meshData.positions);
        this.baseIndices = meshData.indices;
        this.baseUvs = new Float32Array(meshData.uv_front);
        this.animScratch = new Float32Array(this.basePositions.length);

        this.sculptRig = new SculptRig(this.basePositions, this.baseIndices);
        this.restPositions = this.sculptRig.evaluate(this.sculptState);
        this.morphRig = new FaceRetargetRig(this.restPositions, this.baseIndices, this.regionWeights);
        this.subdivider = new MeshSubdivider(
            Math.floor(this.basePositions.length / 3),
            this.baseIndices,
            this.baseUvs,
            0.65
        );
        this.lastMotionFrame = null;

        this.bindActiveTopology();
        this.uploadPositions(this.restPositions);
        this.geometry.computeBoundingSphere();

        await this.updateFrontTexture(frontImageUrl);
        this.updateGuideAnchors(meshData.anchors);
        this.hasMesh = true;
    }

    bindActiveTopology() {
        if (!this.basePositions || !this.subdivider) return;
        if (this.smoothEnabled) {
            const posCopy = new Float32Array(this.subdivider.subdivPositions.length);
            const uvCopy = new Float32Array(this.subdivider.subdivUvs);
            this.geometry.setAttribute("position", new THREE.BufferAttribute(posCopy, 3));
            this.geometry.setAttribute("uv", new THREE.BufferAttribute(uvCopy, 2));
            this.geometry.setIndex(this.subdivider.subdivIndices);
        } else {
            const posCopy = new Float32Array(this.basePositions.length);
            const uvCopy = new Float32Array(this.baseUvs);
            this.geometry.setAttribute("position", new THREE.BufferAttribute(posCopy, 3));
            this.geometry.setAttribute("uv", new THREE.BufferAttribute(uvCopy, 2));
            this.geometry.setIndex(this.baseIndices);
        }
    }

    uploadPositions(basePos) {
        const posAttr = this.geometry.getAttribute("position");
        if (!posAttr) return;
        if (this.smoothEnabled && this.subdivider) {
            const subdivPos = this.subdivider.evaluate(basePos);
            posAttr.array.set(subdivPos);
        } else {
            posAttr.array.set(basePos);
        }
        posAttr.needsUpdate = true;
        this.geometry.computeVertexNormals();
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
            this.uploadPositions(this.restPositions);
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
            this.uploadPositions(this.restPositions);
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
        const frontTex = await loadTextureFromDataUrl(frontImageUrl);
        this.disposeTextures();
        this.frontTexture = frontTex;
        this.material.uniforms.uFrontTex.value = frontTex || this.fallbackTex;
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
        this.uploadPositions(this.animScratch);
    }

    resetPose() {
        this.lastMotionFrame = null;
        this.group.rotation.set(0, 0, 0);
        this.group.quaternion.identity();
        if (!this.hasMesh || !this.restPositions) return;
        this.uploadPositions(this.restPositions);
    }

    setTurntableYaw(yawRad) {
        this.group.quaternion.identity();
        this.group.rotation.set(0, yawRad, 0);
    }

    updateDepthBounds(camera) {
        if (!camera || !this.geometry.boundingSphere) return;
        this.group.getWorldPosition(_worldCenter);
        const dist = camera.position.distanceTo(_worldCenter);
        const r = Math.max(this.geometry.boundingSphere.radius, 0.25);
        const near = Math.max(0.05, dist - r * 1.05);
        const far = dist + r * 1.15;

        this.material.uniforms.uDepthNear.value = near;
        this.material.uniforms.uDepthFar.value = far;
        this.guideDepthMat.uniforms.uDepthNear.value = near;
        this.guideDepthMat.uniforms.uDepthFar.value = far;
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
        const modeInt = mode === "normals" ? 1 : mode === "depth" ? 2 : 0;

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
    }

    setGuidesVisible(visible) {
        this.guidesGroup.visible = Boolean(visible);
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
        this.geometry.deleteAttribute("normal");
        this.geometry.setIndex([]);
        this.geometry.boundingSphere = null;
        this.geometry.boundingBox = null;

        this.basePositions = null;
        this.baseIndices = null;
        this.baseUvs = null;
        this.restPositions = null;
        this.animScratch = null;
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
    }

    dispose() {
        this.disposeTextures();
        this.fallbackTex.dispose();
        this.geometry.dispose();
        this.material.dispose();
        this.neckGeo.dispose();
        this.earGeo.dispose();
        this.guideMat.dispose();
        this.earMat.dispose();
        this.guideNormalMat.dispose();
        this.guideDepthMat.dispose();
    }
}
