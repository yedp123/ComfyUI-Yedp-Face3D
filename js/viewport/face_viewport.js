import { api } from "../../../../scripts/api.js";
import * as THREE from "../lib/three.module.js";
import { OrbitControls } from "../lib/OrbitControls.js";
import { createPlaceholderHead } from "./placeholder_head.js";
import { ReconstructedFaceMesh } from "./face_mesh.js";
import {
    detectFaceForVideo,
    detectOptionalFaceFromDataUrl,
    extractFaceFromDataUrl,
    getTesselationTriangles,
    resetImageLandmarker,
} from "../vision/mediapipe_runner.js";
import { MotionDriver } from "../vision/motion_driver.js";

const DEFAULT_CAMERA_POS = [0, 0.05, 1.85];
const DEFAULT_TARGET = [0, -0.02, 0];
const MAX_RECORDED_FRAMES = 600;

/**
 * Interactive Three.js viewport with multi-view face mesh, right-hand accordion
 * controls (Sculpt, Weights, Lights, Mocap, Render), view-mode switcher,
 * 180-deg turntable sweep, and live webcam / driving video mocap.
 */
export class FaceViewport {
    constructor(ui) {
        this.ui = ui;
        this.disposed = false;
        this.rafId = null;
        this.resizeObserver = null;
        this.viewMode = "textured";
        this.guidesVisible = true;
        this.onGuidesChanged = null;
        this.depthWeight = 1.0;
        this.turntableFrames = 24;
        this.viewportWidth = 480;
        this.viewportHeight = 380;
        this.outputWidth = 512;
        this.outputHeight = 512;

        this.refUrls = { front: null, three_quarter: null, side: null };
        this.extractedViews = { front: null, three_quarter: null, side: null };
        this.loadToken = 0;
        this.reconstructToken = 0;

        // Motion state (Turntable / Webcam / Recorded Clip / Driving Video)
        this.motionDriver = new MotionDriver();
        this.turntablePlaying = false;
        this.turntableCursor = 0;
        this.lastTickMs = 0;

        this.webcamActive = false;
        this.webcamStream = null;
        this.isRecording = false;
        this.lastWebcamDetectMs = 0;
        this.recordedClip = [];
        this.clipPlaying = false;
        this.clipCursor = 0;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(36, 480 / 380, 0.05, 50);
        this.camera.position.set(...DEFAULT_CAMERA_POS);

        this.renderer = this.createRenderer(ui.canvas);
        if (!this.renderer) return;

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minDistance = 0.65;
        this.controls.maxDistance = 6.0;
        this.controls.target.set(...DEFAULT_TARGET);
        this.controls.update();

        this.setupLights();
        this.setupGrid();

        this.placeholderHead = createPlaceholderHead();
        this.reconstructedFace = new ReconstructedFaceMesh();
        this.reconstructedFace.group.visible = false;
        this.reconstructedFace.updateLights(this.ui.lightPanel.getState());
        this.reconstructedFace.setRegionWeights(this.ui.weightPanel.getState());
        this.reconstructedFace.setSculptState(this.ui.sculptPanel.getState());
        this.scene.add(this.placeholderHead.group, this.reconstructedFace.group);

        this.bindHudEvents();
        this.observeResize();
        this.syncSizeFromContainer();
        this.updateRenderUI();
        this.startRenderLoop();
    }

    createRenderer(canvas) {
        try {
            const renderer = new THREE.WebGLRenderer({
                canvas,
                antialias: true,
                alpha: true,
                preserveDrawingBuffer: false,
            });
            renderer.outputColorSpace = THREE.SRGBColorSpace;
            renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
            return renderer;
        } catch (err) {
            const msg = document.createElement("div");
            msg.textContent = `WebGL init failed: ${err?.message || err}`;
            Object.assign(msg.style, {
                position: "absolute",
                inset: "0",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#fca5a5",
                fontSize: "12px",
                padding: "16px",
                textAlign: "center",
            });
            this.ui.stage.appendChild(msg);
            return null;
        }
    }

    setupLights() {
        const ambient = new THREE.HemisphereLight(0xe2e8f0, 0x1e293b, 0.75);
        const keyLight = new THREE.DirectionalLight(0xfff1e0, 1.55);
        keyLight.position.set(1.8, 1.6, 2.4);
        const rimLight = new THREE.DirectionalLight(0x7dd3fc, 0.9);
        rimLight.position.set(-2.0, 1.1, -1.8);
        this.scene.add(ambient, keyLight, rimLight);
    }

    setupGrid() {
        this.grid = new THREE.GridHelper(3.2, 16, 0x334155, 0x1e293b);
        this.grid.position.y = -0.75;
        this.scene.add(this.grid);
    }

    bindHudEvents() {
        this.ui.resetBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.resetCamera();
        });

        for (const [modeId, btn] of Object.entries(this.ui.modeBar.modeButtons)) {
            btn.addEventListener("click", (e) => {
                e.stopPropagation();
                this.setViewMode(modeId);
            });
        }

        this.ui.modeBar.smoothBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            const next = !this.reconstructedFace.smoothEnabled;
            this.reconstructedFace.setSmoothEnabled(next);
            this.ui.modeBar.smoothBtn.style.borderColor = next ? "#38bdf8" : "rgba(255, 255, 255, 0.16)";
            this.ui.modeBar.smoothBtn.style.color = next ? "#38bdf8" : "#f8fafc";
        });

        // MOCAP controls
        this.ui.mocapPanel.webcamBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.toggleWebcam();
        });
        this.ui.mocapPanel.recordBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.toggleRecording();
        });
        this.ui.mocapPanel.calibrateBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.motionDriver.calibrateNextFrame();
        });
        this.ui.mocapPanel.playClipBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.toggleClipPlayback();
        });
        this.ui.mocapPanel.clearClipBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.clearMotionClip();
        });

        // RENDER controls
        this.ui.renderPanel.turntableBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.toggleTurntable();
        });
        this.ui.renderPanel.guidesBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            const next = !this.guidesVisible;
            this.setGuidesVisible(next);
            this.onGuidesChanged?.(next);
        });
    }

    setViewMode(mode) {
        this.viewMode = mode;
        this.ui.modeBar.setActiveMode(mode);
        this.placeholderHead?.setWireframe(mode === "wireframe");
        this.reconstructedFace?.setViewMode(mode);
    }

    updateLights(lightState) {
        this.reconstructedFace?.updateLights(lightState);
    }

    updateWeights(weightState) {
        this.reconstructedFace?.setRegionWeights(weightState);
    }

    updateSculpt(sculptState) {
        this.reconstructedFace?.setSculptState(sculptState);
    }

    toggleTurntable() {
        if (this.turntablePlaying) {
            this.turntablePlaying = false;
            this.reconstructedFace.resetPose();
            this.placeholderHead.group.rotation.set(0, 0, 0);
            this.updateRenderUI();
            return;
        }
        this.stopWebcam();
        this.clipPlaying = false;
        this.turntableCursor = 0;
        this.turntablePlaying = true;
        this.updateMotionUI();
        this.updateRenderUI();
    }

    async toggleWebcam() {
        if (this.webcamActive) {
            this.stopWebcam();
            return;
        }
        this.turntablePlaying = false;
        this.clipPlaying = false;
        this.ui.rightSidebar?.setSectionOpen("mocap", true);

        try {
            this.webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
                audio: false,
            });
            if (this.disposed) {
                for (const track of this.webcamStream.getTracks()) track.stop();
                return;
            }
            this.ui.webcamVideo.srcObject = this.webcamStream;
            await this.ui.webcamVideo.play();
            this.ui.webcamOverlay.show();
            this.webcamActive = true;
            this.motionDriver.reset();
            this.updateMotionUI();
            this.updateRenderUI();
        } catch (err) {
            this.setStatus(`Webcam error: ${err.message}`, true);
            this.stopWebcam();
        }
    }

    stopWebcam() {
        this.isRecording = false;
        this.webcamActive = false;
        if (this.webcamStream) {
            for (const track of this.webcamStream.getTracks()) track.stop();
            this.webcamStream = null;
        }
        if (this.ui.webcamVideo) {
            this.ui.webcamVideo.pause();
            this.ui.webcamVideo.srcObject = null;
        }
        this.ui.webcamOverlay?.hide();
        this.reconstructedFace?.resetPose();
        this.updateMotionUI();
    }

    toggleRecording() {
        if (!this.webcamActive) return;
        if (this.isRecording) {
            this.isRecording = false;
            this.updateMotionUI();
            return;
        }
        this.recordedClip = [];
        this.motionDriver.calibrateNextFrame();
        this.isRecording = true;
        this.updateMotionUI();
    }

    toggleClipPlayback() {
        if (!this.recordedClip.length) return;
        if (this.clipPlaying) {
            this.clipPlaying = false;
            this.reconstructedFace.resetPose();
            this.updateMotionUI();
            return;
        }
        this.stopWebcam();
        this.turntablePlaying = false;
        this.clipCursor = 0;
        this.clipPlaying = true;
        this.updateMotionUI();
        this.updateRenderUI();
    }

    clearMotionClip() {
        this.clipPlaying = false;
        this.isRecording = false;
        this.recordedClip = [];
        this.motionDriver.reset();
        this.reconstructedFace?.resetPose();
        this.updateMotionUI();
    }

    updateMotionUI() {
        const mp = this.ui.mocapPanel;
        mp.webcamBtn.style.borderColor = this.webcamActive ? "#22c55e" : "rgba(255,255,255,0.16)";
        mp.webcamBtn.style.color = this.webcamActive ? "#4ade80" : "#f8fafc";
        mp.recordBtn.style.display = this.webcamActive ? "inline-block" : "none";
        mp.calibrateBtn.style.display = this.webcamActive ? "inline-block" : "none";
        mp.recordBtn.style.borderColor = this.isRecording ? "#ef4444" : "rgba(255,255,255,0.16)";
        mp.recordBtn.style.color = this.isRecording ? "#ef4444" : "#f8fafc";
        mp.recordBtn.textContent = this.isRecording ? "Stop Rec" : "Rec";

        const hasClip = this.recordedClip.length > 0;
        mp.playClipBtn.style.display = hasClip ? "inline-block" : "none";
        mp.clearClipBtn.style.display = hasClip ? "inline-block" : "none";
        mp.playClipBtn.style.borderColor = this.clipPlaying ? "#38bdf8" : "rgba(255,255,255,0.16)";
        mp.playClipBtn.textContent = this.clipPlaying ? "Stop Clip" : `Play (${this.recordedClip.length}f)`;

        if (this.isRecording) {
            mp.motionStatus.textContent = `REC ${this.recordedClip.length}f`;
            mp.motionStatus.style.color = "#f87171";
        } else if (this.webcamActive) {
            mp.motionStatus.textContent = "LIVE tracking active";
            mp.motionStatus.style.color = "#4ade80";
        } else if (this.clipPlaying) {
            mp.motionStatus.textContent = `Playing clip (${this.recordedClip.length}f)`;
            mp.motionStatus.style.color = "#38bdf8";
        } else {
            mp.motionStatus.textContent = hasClip ? `Clip ready: ${this.recordedClip.length}f` : "No active mocap";
            mp.motionStatus.style.color = "#94a3b8";
        }
    }

    updateRenderUI() {
        const rp = this.ui.renderPanel;
        rp.turntableBtn.style.borderColor = this.turntablePlaying ? "#38bdf8" : "rgba(255,255,255,0.16)";
        rp.turntableBtn.style.color = this.turntablePlaying ? "#38bdf8" : "#f8fafc";
        rp.setOutputSpec(this.outputWidth, this.outputHeight, this.turntableFrames);
        if (!this.turntablePlaying) {
            rp.renderStatus.textContent = "Idle · Ready to queue";
            rp.renderStatus.style.color = "#94a3b8";
        }
    }

    setStatus(text, isError = false) {
        if (!this.ui.badge) return;
        this.ui.badge.textContent = text;
        this.ui.badge.style.color = isError ? "#fca5a5" : "#cbd5e1";
        this.ui.badge.style.borderColor = isError
            ? "rgba(248, 113, 113, 0.5)"
            : "rgba(255, 255, 255, 0.12)";
    }

    async loadSingleReference(viewKey, dataUrl) {
        const token = ++this.loadToken;
        this.refUrls[viewKey] = dataUrl;
        this.ui.refStack.setPreview(viewKey, dataUrl, false);
        this.setStatus(`Extracting ${viewKey} (CPU)...`);

        try {
            const extracted = await extractFaceFromDataUrl(dataUrl);
            if (this.disposed || token !== this.loadToken) return;
            this.extractedViews[viewKey] = extracted;
            this.updateRefButtonsUI();

            if (this.extractedViews.front) {
                await this.runReconstruction();
            } else {
                this.setStatus("Load front photo to build mesh");
            }
        } catch (err) {
            if (this.disposed || token !== this.loadToken) return;
            this.refUrls[viewKey] = null;
            this.extractedViews[viewKey] = null;
            this.ui.refStack.setPreview(viewKey, null, false);
            this.setStatus(`Error: ${err.message}`, true);
        }
    }

    async loadReferenceImages({
        front,
        three_quarter = null,
        side = null,
        depth_weight = null,
        driving_batch_id = null,
        driving_frame_count = 0,
    }) {
        if (this.disposed) return;
        const token = ++this.loadToken;

        if (depth_weight !== null && depth_weight !== undefined) {
            this.depthWeight = Number(depth_weight);
        }

        this.refUrls = { front, three_quarter, side };
        this.ui.refStack.setPreview("front", front, false);
        this.ui.refStack.setPreview("three_quarter", three_quarter, false);
        this.ui.refStack.setPreview("side", side, false);
        this.setStatus("Extracting landmarks (CPU)...");

        try {
            const [frontData, tqData, sideData] = await Promise.all([
                front ? extractFaceFromDataUrl(front) : Promise.resolve(null),
                three_quarter ? extractFaceFromDataUrl(three_quarter) : Promise.resolve(null),
                side ? extractFaceFromDataUrl(side) : Promise.resolve(null),
            ]);

            if (this.disposed || token !== this.loadToken) return;
            if (!frontData) {
                this.setStatus("No face in front_image", true);
                return;
            }

            this.extractedViews = { front: frontData, three_quarter: tqData, side: sideData };
            await this.runReconstruction();

            if (driving_batch_id && driving_frame_count > 0 && token === this.loadToken) {
                await this.loadDrivingBatch(driving_batch_id, driving_frame_count, token);
            }
        } catch (err) {
            if (this.disposed || token !== this.loadToken) return;
            console.error("[YedpFace3D] Landmark extraction failed:", err);
            this.setStatus(`Error: ${err.message}`, true);
        }
    }

    async loadDrivingBatch(batchId, count, token = this.loadToken) {
        try {
            this.setStatus(`Extracting driving video (0/${count})...`);
            const res = await api.fetchApi(`/yedp_face3d/driving_batch/${batchId}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            const frames = data.frames || [];
            if (!frames.length || this.disposed || token !== this.loadToken) return;

            this.motionDriver.reset();
            const clip = [];
            for (let i = 0; i < frames.length; i++) {
                if (this.disposed || token !== this.loadToken) return;
                const det = await detectOptionalFaceFromDataUrl(frames[i]);
                if (this.disposed || token !== this.loadToken) return;
                if (det) {
                    const m = this.motionDriver.processDetection(det, { mirrorYaw: false, smoothAlpha: 0.55 });
                    if (m) clip.push(m);
                } else if (clip.length > 0) {
                    clip.push(clip[clip.length - 1]);
                }
                if ((i + 1) % 4 === 0 || i + 1 === frames.length) {
                    this.setStatus(`Extracting driving video (${i + 1}/${frames.length})...`);
                }
            }

            if (clip.length > 0 && token === this.loadToken) {
                this.recordedClip = clip;
                this.turntablePlaying = false;
                this.clipCursor = 0;
                this.clipPlaying = true;
                this.updateMotionUI();
                this.updateRenderUI();
                this.setStatus(`Driving Video Ready (${clip.length}f)`);
            }
        } catch (err) {
            if (this.disposed || token !== this.loadToken) return;
            console.error("[YedpFace3D] Driving batch failed:", err);
            this.setStatus(`Video error: ${err.message}`, true);
        }
    }

    async runReconstruction() {
        if (this.disposed || !this.extractedViews.front) return;
        const token = ++this.reconstructToken;

        try {
            const triangles = await getTesselationTriangles();
            const response = await api.fetchApi("/yedp_face3d/reconstruct", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    front: this.extractedViews.front,
                    three_quarter: this.extractedViews.three_quarter,
                    side: this.extractedViews.side,
                    triangles,
                    depth_weight: this.depthWeight,
                }),
            });

            if (!response.ok) {
                const errBody = await response.json().catch(() => ({}));
                throw new Error(errBody.error || `HTTP ${response.status}`);
            }

            const meshData = await response.json();
            if (this.disposed || token !== this.reconstructToken) return;

            this.reconstructedFace.setSculptState(this.ui.sculptPanel.getState());
            this.reconstructedFace.setRegionWeights(this.ui.weightPanel.getState());
            await this.reconstructedFace.updateFromReconstruction(meshData, this.refUrls.front);
            if (this.disposed || token !== this.reconstructToken) return;

            this.reconstructedFace.setGuidesVisible(this.guidesVisible);
            this.reconstructedFace.setViewMode(this.viewMode);

            this.placeholderHead.group.visible = false;
            this.reconstructedFace.group.visible = true;
            this.updateRefButtonsUI();

            const views = ["Front"];
            if (meshData.has_three_quarter) views.push("3/4 depth");
            if (meshData.has_side) views.push("Side depth");
            this.setStatus(`3D Face Ready (${views.join(" + ")})`);
        } catch (err) {
            if (this.disposed || token !== this.reconstructToken) return;
            console.error("[YedpFace3D] Reconstruction failed:", err);
            this.setStatus(`Reconstruct error: ${err.message}`, true);
        }
    }

    updateRefButtonsUI() {
        this.ui.refStack.setSlotActive("front", Boolean(this.extractedViews.front));
        this.ui.refStack.setSlotActive("three_quarter", Boolean(this.extractedViews.three_quarter));
        this.ui.refStack.setSlotActive("side", Boolean(this.extractedViews.side));
    }

    removeReference(viewKey) {
        this.loadToken++;
        this.reconstructToken++;
        this.refUrls[viewKey] = null;
        this.extractedViews[viewKey] = null;
        this.ui.refStack.setPreview(viewKey, null, false);
        this.updateRefButtonsUI();

        if (!this.extractedViews.front) {
            this.turntablePlaying = false;
            this.clipPlaying = false;
            this.reconstructedFace.clear(false);
            this.reconstructedFace.group.visible = false;
            this.placeholderHead.group.visible = true;
            this.placeholderHead.group.rotation.set(0, 0, 0);
            this.updateMotionUI();
            this.updateRenderUI();
            this.setStatus(
                this.extractedViews.three_quarter || this.extractedViews.side
                    ? "Load front photo to build mesh"
                    : "Yedp Face3D · Placeholder"
            );
            return;
        }
        this.runReconstruction();
    }

    clearAllReferences() {
        this.loadToken++;
        this.reconstructToken++;
        this.stopWebcam();
        this.turntablePlaying = false;
        this.turntableCursor = 0;
        this.clearMotionClip();
        resetImageLandmarker();

        this.refUrls = { front: null, three_quarter: null, side: null };
        this.extractedViews = { front: null, three_quarter: null, side: null };
        this.ui.refStack.clearAll();

        this.ui.sculptPanel.reset(false);
        this.ui.weightPanel.reset(false);
        this.ui.lightPanel.reset(false);

        this.reconstructedFace.clear(true);
        this.reconstructedFace.updateLights(this.ui.lightPanel.getState());
        this.reconstructedFace.setRegionWeights(this.ui.weightPanel.getState());
        this.reconstructedFace.setSculptState(this.ui.sculptPanel.getState());
        this.reconstructedFace.group.visible = false;

        this.placeholderHead.group.visible = true;
        this.placeholderHead.group.rotation.set(0, 0, 0);
        this.updateRefButtonsUI();
        this.updateMotionUI();
        this.updateRenderUI();
        this.setStatus("Yedp Face3D · Placeholder");
    }

    setDepthWeight(weight) {
        const next = Number(weight);
        if (!Number.isFinite(next) || next === this.depthWeight) return;
        this.depthWeight = next;
        if (this.extractedViews.front) {
            this.runReconstruction();
        }
    }

    setTurntableFrames(frames) {
        this.turntableFrames = Math.max(1, Number(frames) || 24);
        this.updateRenderUI();
    }

    resetCamera() {
        if (!this.camera || !this.controls) return;
        this.camera.position.set(...DEFAULT_CAMERA_POS);
        this.controls.target.set(...DEFAULT_TARGET);
        this.controls.update();
    }

    setGuidesVisible(visible) {
        this.guidesVisible = Boolean(visible);
        this.placeholderHead?.setGuidesVisible(this.guidesVisible);
        this.reconstructedFace?.setGuidesVisible(this.guidesVisible);
        this.ui.renderPanel?.setGuidesActive(this.guidesVisible);
    }

    setOutputResolution(width, height) {
        this.outputWidth = Math.max(1, Number(width) || 512);
        this.outputHeight = Math.max(1, Number(height) || 512);
        this.updateReadout();
        this.updateRenderUI();
    }

    observeResize() {
        this.resizeObserver = new ResizeObserver((entries) => {
            if (this.disposed) return;
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                this.applyResize(width, height);
            }
        });
        this.resizeObserver.observe(this.ui.stage);
    }

    syncSizeFromContainer() {
        const w = this.ui.stage.clientWidth;
        const h = this.ui.stage.clientHeight;
        if (w > 0 && h > 0) {
            this.applyResize(w, h);
        } else {
            this.applyResize(this.viewportWidth, this.viewportHeight);
        }
    }

    applyResize(width, height) {
        if (!this.renderer || !this.camera) return;
        const w = Math.floor(width);
        const h = Math.floor(height);
        if (w < 2 || h < 2) return;

        this.viewportWidth = w;
        this.viewportHeight = h;
        // Pass updateStyle=false so canvas CSS stays 100% and avoids ResizeObserver loops.
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.updateReadout();
    }

    updateReadout() {
        this.ui.sizeReadout.textContent =
            `Viewport: ${this.viewportWidth}x${this.viewportHeight} | Output: ${this.outputWidth}x${this.outputHeight}`;
    }

    async stepMotion(nowMs) {
        const dt = this.lastTickMs ? (nowMs - this.lastTickMs) / 1000 : 0.016;
        this.lastTickMs = nowMs;

        if (this.turntablePlaying) {
            const fps = 24;
            this.turntableCursor = (this.turntableCursor + dt * fps) % this.turntableFrames;
            const t = this.turntableFrames > 1
                ? this.turntableCursor / (this.turntableFrames - 1)
                : 0.5;
            const yawDeg = -90 + 180 * t;
            const yawRad = THREE.MathUtils.degToRad(yawDeg);
            if (this.reconstructedFace.hasMesh) {
                this.reconstructedFace.setTurntableYaw(yawRad);
            } else {
                this.placeholderHead.group.rotation.set(0, yawRad, 0);
            }
            const frameIdx = Math.floor(this.turntableCursor) + 1;
            this.ui.renderPanel.renderStatus.textContent = `Turntable F${frameIdx}/${this.turntableFrames} (${Math.round(yawDeg)}°)`;
            this.ui.renderPanel.renderStatus.style.color = "#38bdf8";
            return;
        }

        if (this.clipPlaying && this.recordedClip.length > 0 && this.reconstructedFace.hasMesh) {
            const fps = 24;
            this.clipCursor = (this.clipCursor + dt * fps) % this.recordedClip.length;
            const idx = Math.floor(this.clipCursor);
            this.reconstructedFace.applyMotionFrame(this.recordedClip[idx]);
            this.ui.mocapPanel.motionStatus.textContent = `F${idx + 1}/${this.recordedClip.length}`;
            this.ui.mocapPanel.motionStatus.style.color = "#38bdf8";
            return;
        }

        if (this.webcamActive && this.ui.webcamVideo.readyState >= 2 && nowMs - this.lastWebcamDetectMs > 38) {
            this.lastWebcamDetectMs = nowMs;
            try {
                const det = await detectFaceForVideo(this.ui.webcamVideo, nowMs);
                if (!this.disposed && this.webcamActive) {
                    const frame = det
                        ? this.motionDriver.processDetection(det, { mirrorYaw: true, smoothAlpha: 0.45 })
                        : null;
                    this.ui.webcamOverlay.drawFrame(this.ui.webcamVideo, det, frame?.bs);
                    if (frame) {
                        this.reconstructedFace.applyMotionFrame(frame);
                        if (this.isRecording && this.recordedClip.length < MAX_RECORDED_FRAMES) {
                            this.recordedClip.push(frame);
                            this.ui.mocapPanel.motionStatus.textContent = `REC ${this.recordedClip.length}f`;
                        }
                    }
                }
            } catch {
                // Ignore transient video frame decode errors
            }
        }
    }

    startRenderLoop() {
        const tick = (nowMs) => {
            if (this.disposed) return;
            this.rafId = requestAnimationFrame(tick);
            this.stepMotion(nowMs);
            this.controls.update();
            if (this.viewMode === "depth" && this.reconstructedFace.hasMesh) {
                this.reconstructedFace.updateDepthBounds(this.camera);
            }
            this.renderer.render(this.scene, this.camera);
        };
        this.rafId = requestAnimationFrame(tick);
    }

    dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.loadToken++;
        this.reconstructToken++;
        this.stopWebcam();

        if (this.rafId !== null) {
            cancelAnimationFrame(this.rafId);
            this.rafId = null;
        }
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }
        if (this.placeholderHead) {
            this.placeholderHead.dispose();
            this.placeholderHead = null;
        }
        if (this.reconstructedFace) {
            this.reconstructedFace.dispose();
            this.reconstructedFace = null;
        }
        if (this.grid) {
            this.grid.geometry?.dispose();
            this.grid.material?.dispose();
            this.grid = null;
        }
        if (this.controls) {
            this.controls.dispose();
            this.controls = null;
        }
        if (this.renderer) {
            this.renderer.dispose();
            this.renderer = null;
        }
    }
}
