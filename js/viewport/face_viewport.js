import { api } from "../../../../scripts/api.js";
import * as THREE from "../lib/three.module.js";
import { OrbitControls } from "../lib/OrbitControls.js";
import { getCanonicalFaceData, getHeadShellRig, ReconstructedFaceMesh } from "./face_mesh.js";
import { renderAndUploadPasses } from "./offscreen_renderer.js";
import { stopViewportEvents } from "./ui_icons.js";
import {
    detectFaceForVideo,
    detectOptionalFaceFromDataUrl,
    extractFaceFromDataUrl,
    getTesselationTriangles,
    resetImageLandmarker,
} from "../vision/mediapipe_runner.js";
import { MotionDriver } from "../vision/motion_driver.js";
import {
    buildRefViewUrl,
    createRefSlotMap,
    getCachedLandmarks,
    getRefCacheKey,
    normalizePersistedRefs,
    packCameraPose,
    packSnapshotPose,
    REF_VIEW_KEYS,
    setCachedLandmarks,
    unpackCameraPose,
    unpackSnapshotPose,
    uploadReferenceImage,
} from "./state_serializer.js";

const DEFAULT_CAMERA_POS = [0, 0.05, 1.85];
const DEFAULT_TARGET = [0, -0.02, 0];
const MAX_RECORDED_FRAMES = 600;

function cloneMotionFrame(frame) {
    if (!frame) return null;
    return {
        quat: frame.quat ? [...frame.quat] : null,
        deltas: frame.deltas ? new Float32Array(frame.deltas) : null,
        neutralAspect: frame.neutralAspect,
        mirror: frame.mirror,
        poseAngleMag: frame.poseAngleMag,
        bs: frame.bs ? { ...frame.bs } : null,
    };
}

/**
 * Interactive Three.js viewport supporting Ortho/Persp cameras, Fit Head,
 * multi-view face + welded head shell reconstruction, sculpting, mocap, and
 * synchronized Phase 4 offscreen multi-pass rendering.
 */
export class FaceViewport {
    constructor(ui) {
        this.ui = ui;
        this.disposed = false;
        this.rafId = null;
        this.resizeObserver = null;
        this.viewMode = "clay";
        this.isDefaultHead = false;
        this.guidesVisible = true;
        this.enableReliabilityWeighting = true;
        this.enableAnatomicalWeighting = true;
        this.enableRegularization = true;
        this.weightDebugEnabled = false;
        this.weightDebugView = "combined";
        this.weightDebugBtn = null;
        this.weightDebugViewBtn = null;
        this.relWeightBtn = null;
        this.anatWeightBtn = null;
        this.regBtn = null;
        this.regRebuildTimer = null;
        this.lastRegState = { baseAnchor: 0.25, dispSmooth: 0.45, maxDisp: 0.18, symmetry: 0.0 };
        this.lastMocapWeightState = { eyes: 1.15, brows: 1.0, mouth: 1.0, mouthWidth: 0.55, jaw: 0.85, cheeks: 0.45 };
        this.onGuidesChanged = null;
        this.depthWeight = 1.0;
        this.turntableFrames = 24;
        this.viewportWidth = 480;
        this.viewportHeight = 380;
        this.outputWidth = 512;
        this.outputHeight = 512;

        this.refUrls = createRefSlotMap(null);
        this.persistedRefs = createRefSlotMap(null);
        this.extractedViews = createRefSlotMap(null);
        this.lastPythonSyncedRefs = null;
        this.loadToken = 0;
        this.resetToken = 0;
        this.viewTokens = createRefSlotMap(0);
        this.reconstructToken = 0;
        this.pendingRestoredCamera = null;

        // Motion state (Turntable / Webcam / Recorded Clip / Driving Video)
        this.motionDriver = new MotionDriver();
        this.turntablePlaying = false;
        this.turntableCursor = 0;
        this.lastTickMs = 0;

        this.webcamActive = false;
        this.webcamBusy = false;
        this.webcamStream = null;
        this.isRecording = false;
        this.lastWebcamDetectMs = 0;
        this.lastWebcamFrame = null;
        this.isExporting = false;

        this.recordedClip = [];
        this.clipSource = null;
        this.clipPlaying = false;
        this.clipCursor = 0;

        this.snapshotPose = null;
        this.snapshotActive = false;
        this.snapCountdownTimer = null;
        this.snapCountdownRemaining = 0;

        this.scene = new THREE.Scene();
        this.renderState = this.ui.renderPanel.getState();

        const initAspect = 480 / 380;
        this.perspCamera = new THREE.PerspectiveCamera(this.renderState.fov || 36, initAspect, 0.05, 50);
        this.perspCamera.position.set(...DEFAULT_CAMERA_POS);

        const initDist = new THREE.Vector3(...DEFAULT_CAMERA_POS).distanceTo(new THREE.Vector3(...DEFAULT_TARGET));
        this.orthoHalfH = initDist * Math.tan(THREE.MathUtils.degToRad((this.renderState.fov || 36) * 0.5));
        this.orthoCamera = new THREE.OrthographicCamera(
            -this.orthoHalfH * initAspect,
            this.orthoHalfH * initAspect,
            this.orthoHalfH,
            -this.orthoHalfH,
            0.05,
            50
        );
        this.orthoCamera.position.set(...DEFAULT_CAMERA_POS);
        this.camera = this.renderState.projection === "ortho" ? this.orthoCamera : this.perspCamera;

        this.renderer = this.createRenderer(ui.canvas);
        if (!this.renderer) return;

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minDistance = 0.45;
        this.controls.maxDistance = 6.0;
        this.controls.minZoom = 0.25;
        this.controls.maxZoom = 8.0;
        this.controls.target.set(...DEFAULT_TARGET);
        this.controls.update();

        this.setupLights();
        this.setupGrid();

        this.reconstructedFace = new ReconstructedFaceMesh();
        this.reconstructedFace.group.visible = false;
        this.reconstructedFace.updateLights(this.ui.lightPanel.getState());
        this.reconstructedFace.setRegionWeights(this.ui.weightPanel.getState());
        this.reconstructedFace.setSculptState(this.ui.sculptPanel.getState());
        this.reconstructedFace.applyRenderState(this.renderState);
        this.scene.add(this.reconstructedFace.group);

        this.ui.refStack.setClearHandler?.((viewKey) => this.clearSingleReference(viewKey));
        this.bindHudEvents();
        this.setViewMode("clay");
        this.observeResize();
        this.syncSizeFromContainer();
        this.updateRenderUI();
        this.startRenderLoop();
        this.setStatus("Loading Canonical Head...");
        this.loadDefaultCanonicalHead();
    }

    async loadDefaultCanonicalHead() {
        if (this.disposed) return;
        const initResetToken = this.resetToken;
        const initReconstructToken = this.reconstructToken;

        try {
            const [shellRig, canonicalData] = await Promise.all([
                getHeadShellRig(),
                getCanonicalFaceData(),
            ]);
            if (
                this.disposed
                || initResetToken !== this.resetToken
                || initReconstructToken !== this.reconstructToken
                || this.extractedViews.front
            ) {
                return;
            }

            this.reconstructedFace.setSculptState(this.ui.sculptPanel.getState());
            this.reconstructedFace.setRegionWeights(this.ui.weightPanel.getState());
            await this.reconstructedFace.updateFromReconstruction(canonicalData, null, shellRig);
            if (
                this.disposed
                || initResetToken !== this.resetToken
                || initReconstructToken !== this.reconstructToken
                || this.extractedViews.front
            ) {
                return;
            }

            this.isDefaultHead = true;
            this.reconstructedFace.setGuidesVisible(this.guidesVisible);
            this.reconstructedFace.setViewMode(this.viewMode);
            this.reconstructedFace.setWeightDebugView(this.weightDebugView);
            this.reconstructedFace.setWeightDebugVisible(this.weightDebugEnabled);
            this.syncWeightDebugButtons();
            this.reconstructedFace.applyRenderState(this.renderState);
            this.reconstructedFace.group.visible = true;

            if (this.snapshotActive && this.snapshotPose?.motionFrame) {
                this.reconstructedFace.applyMotionFrame(this.snapshotPose.motionFrame);
            }
            if (this.pendingRestoredCamera) {
                const pendingCam = this.pendingRestoredCamera;
                this.pendingRestoredCamera = null;
                this.applyCameraPoseState(pendingCam);
            } else {
                this.fitHeadToView();
            }

            this.setStatus("Default Canonical Head (Load Front Photo)");
            console.log("[YedpFace3D] Loaded default canonical clay head (478 verts + head_shell.glb, instant offline)");
        } catch (err) {
            if (!this.disposed) {
                console.warn("[YedpFace3D] Failed to load default canonical head:", err);
            }
        }
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
            this.setSmoothEnabled(!this.reconstructedFace.smoothEnabled);
        });

        // MOCAP controls
        this.ui.mocapPanel.webcamBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.toggleWebcam();
        });
        this.ui.mocapPanel.calibrateBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.motionDriver.calibrateNextFrame();
        });
        this.ui.mocapPanel.recordBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.toggleRecording();
        });
        this.ui.mocapPanel.snapBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.captureSnapshot();
        });
        this.ui.mocapPanel.snapTimedBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.startSnapCountdown(3);
        });
        this.ui.mocapPanel.snapToggleBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.toggleSnapshot();
        });
        this.ui.mocapPanel.snapAngleBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.restoreSnapshotCamera();
        });
        this.ui.mocapPanel.clearSnapBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.clearSnapshot();
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

        this.initWeightDebugControls();
    }

    initWeightDebugControls() {
        if (!this.ui?.hudLayer) return;
        const bar = document.createElement("div");
        Object.assign(bar.style, {
            position: "absolute",
            top: "10px",
            right: "10px",
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-end",
            gap: "4px",
            zIndex: "10",
            pointerEvents: "auto",
        });
        stopViewportEvents(bar);

        const makeToggle = (label, title) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.textContent = label;
            btn.title = title;
            Object.assign(btn.style, {
                fontSize: "10px",
                fontWeight: "600",
                color: "#cbd5e1",
                backgroundColor: "rgba(15, 18, 25, 0.82)",
                border: "1px solid rgba(255, 255, 255, 0.16)",
                borderRadius: "5px",
                padding: "3px 8px",
                cursor: "pointer",
                backdropFilter: "blur(4px)",
            });
            stopViewportEvents(btn);
            return btn;
        };

        this.weightDebugBtn = makeToggle(
            "Weight Debug",
            "Show per-landmark Phase 1/2 reliability points (Green >0.7, Yellow 0.3-0.7, Red 0.05-0.3, Grey <0.05)"
        );
        this.weightDebugViewBtn = makeToggle(
            "All",
            "Cycle Weight Debug view (All | Front | 3Q A | 3Q B | Side A | Side B)"
        );
        this.relWeightBtn = makeToggle(
            "Rel. Weight: ON",
            "Toggle Phase 1 landmark reliability & visibility weighting (side-by-side comparison)"
        );
        this.anatWeightBtn = makeToggle(
            "Anat+Huber: ON",
            "Toggle Phase 2 anatomical group importance & Huber outlier damping"
        );
        this.regBtn = makeToggle(
            "Reg: ON",
            "Toggle Phase 3 displacement regularization (Base Anchor, Disp Smooth, Max Disp, Symmetry)"
        );

        const debugRow = document.createElement("div");
        Object.assign(debugRow.style, {
            display: "flex",
            alignItems: "center",
            gap: "4px",
        });
        debugRow.append(this.weightDebugBtn, this.weightDebugViewBtn);

        const phase23Row = document.createElement("div");
        Object.assign(phase23Row.style, {
            display: "flex",
            alignItems: "center",
            gap: "4px",
        });
        phase23Row.append(this.anatWeightBtn, this.regBtn);

        this.weightDebugBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.setWeightDebugEnabled(!this.weightDebugEnabled);
        });
        this.weightDebugViewBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.cycleWeightDebugView();
        });
        this.relWeightBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.setEnableReliabilityWeighting(!this.enableReliabilityWeighting);
        });
        this.anatWeightBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.setEnableAnatomicalWeighting(!this.enableAnatomicalWeighting);
        });
        this.regBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.setEnableRegularization(!this.enableRegularization);
        });

        bar.append(debugRow, this.relWeightBtn, phase23Row);
        this.ui.hudLayer.appendChild(bar);
        this.syncWeightDebugButtons();
    }

    getAvailableWeightDebugViews() {
        const byView = this.reconstructedFace?.landmarkWeightsByView;
        const views = ["combined", "front"];
        if (byView?.three_quarter || this.extractedViews.three_quarter) views.push("three_quarter");
        if (byView?.three_quarter_2 || this.extractedViews.three_quarter_2) views.push("three_quarter_2");
        if (byView?.side || this.extractedViews.side) views.push("side");
        if (byView?.side_2 || this.extractedViews.side_2) views.push("side_2");
        return views;
    }

    cycleWeightDebugView() {
        const avail = this.getAvailableWeightDebugViews();
        const curIdx = avail.indexOf(this.weightDebugView);
        const nextView = avail[(curIdx + 1) % avail.length];
        this.setWeightDebugView(nextView);
    }

    setWeightDebugView(viewKey) {
        const avail = this.getAvailableWeightDebugViews();
        this.weightDebugView = avail.includes(viewKey) ? viewKey : "combined";
        this.reconstructedFace?.setWeightDebugView(this.weightDebugView);
        this.syncWeightDebugButtons();
    }

    syncWeightDebugButtons() {
        const viewLabels = {
            combined: "All",
            front: "Front",
            three_quarter: "3Q A",
            three_quarter_2: "3Q B",
            side: "Side A",
            side_2: "Side B",
        };
        if (this.weightDebugBtn) {
            const active = this.weightDebugEnabled;
            this.weightDebugBtn.style.borderColor = active ? "#38bdf8" : "rgba(255, 255, 255, 0.16)";
            this.weightDebugBtn.style.color = active ? "#38bdf8" : "#cbd5e1";
        }
        if (this.weightDebugViewBtn) {
            this.weightDebugViewBtn.style.display = this.weightDebugEnabled ? "inline-block" : "none";
            this.weightDebugViewBtn.textContent = viewLabels[this.weightDebugView] || "All";
            this.weightDebugViewBtn.style.borderColor = this.weightDebugView === "combined" ? "rgba(255, 255, 255, 0.22)" : "#38bdf8";
            this.weightDebugViewBtn.style.color = this.weightDebugView === "combined" ? "#e2e8f0" : "#38bdf8";
        }
        if (this.relWeightBtn) {
            const on = this.enableReliabilityWeighting;
            this.relWeightBtn.textContent = on ? "Rel. Weight: ON" : "Rel. Weight: OFF";
            this.relWeightBtn.style.borderColor = on ? "#22c55e" : "#f59e0b";
            this.relWeightBtn.style.color = on ? "#4ade80" : "#fbbf24";
        }
        if (this.anatWeightBtn) {
            const relOn = this.enableReliabilityWeighting;
            const on = relOn && this.enableAnatomicalWeighting;
            this.anatWeightBtn.textContent = on ? "Anat+Huber: ON" : "Anat+Huber: OFF";
            this.anatWeightBtn.style.borderColor = on ? "#22c55e" : "rgba(255, 255, 255, 0.16)";
            this.anatWeightBtn.style.color = on ? "#4ade80" : (relOn ? "#fbbf24" : "#64748b");
            this.anatWeightBtn.style.opacity = relOn ? "1.0" : "0.6";
        }
        if (this.regBtn) {
            const on = this.enableRegularization;
            this.regBtn.textContent = on ? "Reg: ON" : "Reg: OFF";
            this.regBtn.style.borderColor = on ? "#22c55e" : "#f59e0b";
            this.regBtn.style.color = on ? "#4ade80" : "#fbbf24";
        }
    }

    setWeightDebugEnabled(enabled) {
        this.weightDebugEnabled = Boolean(enabled);
        this.reconstructedFace?.setWeightDebugView(this.weightDebugView);
        this.reconstructedFace?.setWeightDebugVisible(this.weightDebugEnabled);
        this.syncWeightDebugButtons();
    }

    setEnableReliabilityWeighting(enabled) {
        const next = Boolean(enabled);
        if (next === this.enableReliabilityWeighting) return;
        this.enableReliabilityWeighting = next;
        this.syncWeightDebugButtons();
        if (this.extractedViews.front) {
            this.runReconstruction();
        }
    }

    setEnableAnatomicalWeighting(enabled) {
        const next = Boolean(enabled);
        if (next === this.enableAnatomicalWeighting) return;
        this.enableAnatomicalWeighting = next;
        this.syncWeightDebugButtons();
        if (this.extractedViews.front) {
            this.runReconstruction();
        }
    }

    setEnableRegularization(enabled) {
        const next = Boolean(enabled);
        if (next === this.enableRegularization) return;
        this.enableRegularization = next;
        this.syncWeightDebugButtons();
        if (this.extractedViews.front) {
            this.runReconstruction();
        }
    }

    scheduleReconstruction() {
        if (this.regRebuildTimer !== null) {
            clearTimeout(this.regRebuildTimer);
        }
        this.regRebuildTimer = setTimeout(() => {
            this.regRebuildTimer = null;
            if (!this.disposed && this.extractedViews.front) {
                this.runReconstruction();
            }
        }, 80);
    }

    setViewMode(mode) {
        this.viewMode = mode;
        this.ui.modeBar.setActiveMode(mode);
        this.reconstructedFace?.setViewMode(mode);
    }

    setSmoothEnabled(enabled) {
        const next = Boolean(enabled);
        this.reconstructedFace?.setSmoothEnabled(next);
        if (this.ui.modeBar?.smoothBtn) {
            this.ui.modeBar.smoothBtn.style.borderColor = next ? "#38bdf8" : "rgba(255, 255, 255, 0.16)";
            this.ui.modeBar.smoothBtn.style.color = next ? "#38bdf8" : "#f8fafc";
        }
    }

    captureCameraPose() {
        if (!this.controls || !this.camera) return null;
        return {
            projection: this.renderState?.projection || "ortho",
            orthoHalfH: this.orthoHalfH,
            orthoZoom: this.orthoCamera.zoom,
            fov: this.perspCamera.fov,
            position: this.camera.position.clone(),
            quaternion: this.camera.quaternion.clone(),
            target: this.controls.target.clone(),
        };
    }

    applyCameraPoseState(cp) {
        if (!cp || !this.controls) return;
        const targetProj = cp.projection || this.renderState?.projection || "ortho";
        const targetCam = targetProj === "ortho" ? this.orthoCamera : this.perspCamera;
        if (this.camera !== targetCam) {
            this.camera = targetCam;
            this.controls.object = targetCam;
        }
        if (this.ui.renderPanel && this.renderState?.projection !== targetProj) {
            this.ui.renderPanel.setState({ projection: targetProj }, false);
            this.renderState = this.ui.renderPanel.getState();
            this.ui.renderPanel.syncProjectionUI?.();
        }
        this.orthoHalfH = cp.orthoHalfH;
        this.orthoCamera.zoom = cp.orthoZoom;
        this.perspCamera.fov = cp.fov;
        this.perspCamera.updateProjectionMatrix();
        this.controls.target.copy(cp.target);
        this.camera.position.copy(cp.position);
        this.camera.quaternion.copy(cp.quaternion);
        this.applyResize(this.viewportWidth, this.viewportHeight);
        this.controls.update();
    }

    updateLights(lightState) {
        this.reconstructedFace?.updateLights(lightState);
    }

    syncWeightTrackingFromPanel(weightState = this.ui.weightPanel?.getState()) {
        if (!weightState) return;
        for (const k of ["eyes", "brows", "mouth", "mouthWidth", "jaw", "cheeks"]) {
            if (weightState[k] !== undefined) this.lastMocapWeightState[k] = Number(weightState[k]);
        }
        for (const k of ["baseAnchor", "dispSmooth", "maxDisp", "symmetry"]) {
            if (weightState[k] !== undefined) this.lastRegState[k] = Number(weightState[k]);
        }
    }

    updateWeights(weightState) {
        if (!weightState) return;
        let mocapChanged = false;
        for (const k of ["eyes", "brows", "mouth", "mouthWidth", "jaw", "cheeks"]) {
            const v = Number(weightState[k]);
            if (Number.isFinite(v) && v !== this.lastMocapWeightState[k]) {
                this.lastMocapWeightState[k] = v;
                mocapChanged = true;
            }
        }
        if (mocapChanged) {
            this.reconstructedFace?.setRegionWeights(weightState);
        }

        let regChanged = false;
        for (const k of ["baseAnchor", "dispSmooth", "maxDisp", "symmetry"]) {
            const v = Number(weightState[k]);
            if (Number.isFinite(v) && v !== this.lastRegState[k]) {
                this.lastRegState[k] = v;
                regChanged = true;
            }
        }
        if (regChanged && this.extractedViews.front) {
            this.scheduleReconstruction();
        }
    }

    updateSculpt(sculptState) {
        this.reconstructedFace?.setSculptState(sculptState);
    }

    updateRenderSettings(state, changedKey = null) {
        const prevProj = this.renderState?.projection || "ortho";
        this.renderState = { ...state };
        this.reconstructedFace?.applyRenderState(this.renderState);

        if (changedKey === "fov") {
            this.perspCamera.fov = Number(state.fov || 36);
            this.perspCamera.updateProjectionMatrix();
        } else if (changedKey === "elevation") {
            const offset = this.camera.position.clone().sub(this.controls.target);
            const r = Math.max(offset.length(), 0.45);
            const az = Math.atan2(offset.x, offset.z);
            const el = THREE.MathUtils.degToRad(state.elevation || 0);
            this.camera.position.set(
                this.controls.target.x + r * Math.cos(el) * Math.sin(az),
                this.controls.target.y + r * Math.sin(el),
                this.controls.target.z + r * Math.cos(el) * Math.cos(az)
            );
            this.controls.update();
        } else if (changedKey === "projection" || changedKey === "reset") {
            if (state.projection !== prevProj || changedKey === "reset") {
                this.switchCameraProjection(state.projection);
            }
            if (changedKey === "reset") {
                this.resetCamera();
            }
        }
        if (changedKey === "exportMode" || changedKey === "reset") {
            this.updateRenderUI();
            this.updateReadout();
        }
    }

    switchCameraProjection(projection) {
        if (!this.controls) return;
        const nextIsOrtho = projection === "ortho";
        const oldCam = this.camera;
        const newCam = nextIsOrtho ? this.orthoCamera : this.perspCamera;
        if (oldCam === newCam) return;

        const target = this.controls.target;
        const offset = oldCam.position.clone().sub(target);
        const dist = Math.max(offset.length(), 0.45);
        const dir = dist > 1e-5 ? offset.normalize() : new THREE.Vector3(0, 0, 1);

        if (nextIsOrtho) {
            this.orthoHalfH = dist * Math.tan(THREE.MathUtils.degToRad((this.perspCamera.fov || 36) * 0.5));
            this.orthoCamera.zoom = 1.0;
            this.orthoCamera.position.copy(oldCam.position);
        } else {
            const visHalfH = this.orthoHalfH / Math.max(this.orthoCamera.zoom, 1e-4);
            this.perspCamera.fov = Number(this.renderState.fov || 36);
            const newDist = Math.max(visHalfH / Math.tan(THREE.MathUtils.degToRad(this.perspCamera.fov * 0.5)), 0.45);
            this.perspCamera.position.copy(target).addScaledVector(dir, newDist);
            this.perspCamera.updateProjectionMatrix();
        }

        this.camera = newCam;
        this.controls.object = newCam;
        this.applyResize(this.viewportWidth, this.viewportHeight);
        this.controls.update();
    }

    fitHeadToView(stateOverride = null) {
        if (!this.controls || !this.reconstructedFace?.hasMesh) return;
        const rs = stateOverride || this.ui.renderPanel.getState();
        this.renderState = { ...rs };

        const box = this.reconstructedFace.getWorldBoundingBox();
        if (box.isEmpty()) return;

        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const margin = 1.0 + Math.max(0.0, Number(rs.fitMargin ?? 0.12));
        const outAspect = this.outputWidth / Math.max(this.outputHeight, 1);
        const viewAspect = this.viewportWidth / Math.max(this.viewportHeight, 1);
        const limitingAspect = Math.min(outAspect, viewAspect);
        const reqHalfH = 0.5 * Math.max(size.y, size.x / Math.max(limitingAspect, 0.25)) * margin;
        const elRad = THREE.MathUtils.degToRad(rs.elevation || 0);

        this.controls.target.copy(center);
        if (this.camera.isOrthographicCamera) {
            this.orthoHalfH = reqHalfH;
            this.orthoCamera.zoom = 1.0;
            const dist = Math.max(size.length() * 2.5, 1.85);
            this.orthoCamera.position.set(
                center.x,
                center.y + Math.sin(elRad) * dist,
                center.z + Math.cos(elRad) * dist
            );
            this.applyResize(this.viewportWidth, this.viewportHeight);
        } else {
            this.perspCamera.fov = Number(rs.fov || 36);
            const fovRad = THREE.MathUtils.degToRad(this.perspCamera.fov * 0.5);
            const dist = Math.max(reqHalfH / Math.tan(fovRad), size.z * 0.8 + 0.6);
            this.perspCamera.position.set(
                center.x,
                center.y + Math.sin(elRad) * dist,
                center.z + Math.cos(elRad) * dist
            );
            this.perspCamera.updateProjectionMatrix();
        }
        this.controls.update();
    }

    toggleTurntable() {
        if (this.turntablePlaying) {
            this.turntablePlaying = false;
            if (this.snapshotActive && this.snapshotPose) {
                this.reconstructedFace.applyMotionFrame(this.snapshotPose.motionFrame);
            } else {
                this.reconstructedFace.resetPose();
            }
            this.updateRenderUI();
            return;
        }
        this.stopWebcam({ preserveSnapshot: true });
        this.clipPlaying = false;
        if (this.snapshotActive && this.snapshotPose && this.reconstructedFace.hasMesh) {
            this.reconstructedFace.applyMotionFrame(this.snapshotPose.motionFrame);
        }
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
        this.snapshotActive = false;
        this.lastWebcamFrame = null;
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
            this.updateReadout();
        } catch (err) {
            this.setStatus(`Webcam error: ${err.message}`, true);
            this.stopWebcam();
        }
    }

    cancelSnapCountdown() {
        if (this.snapCountdownTimer !== null) {
            clearInterval(this.snapCountdownTimer);
            this.snapCountdownTimer = null;
        }
        this.snapCountdownRemaining = 0;
    }

    stopWebcam({ preserveSnapshot = false } = {}) {
        this.cancelSnapCountdown();
        this.isRecording = false;
        this.webcamActive = false;
        this.webcamBusy = false;
        this.lastWebcamFrame = null;
        if (this.webcamStream) {
            for (const track of this.webcamStream.getTracks()) track.stop();
            this.webcamStream = null;
        }
        if (this.ui.webcamVideo) {
            this.ui.webcamVideo.pause();
            this.ui.webcamVideo.srcObject = null;
        }
        this.ui.webcamOverlay?.hide();
        if (preserveSnapshot && this.snapshotActive && this.snapshotPose && this.reconstructedFace?.hasMesh) {
            this.reconstructedFace.applyMotionFrame(this.snapshotPose.motionFrame);
        } else {
            if (!preserveSnapshot) this.snapshotActive = false;
            this.reconstructedFace?.resetPose();
        }
        this.updateMotionUI();
        this.updateRenderUI();
        this.updateReadout();
    }

    captureSnapshot() {
        this.cancelSnapCountdown();
        const srcFrame = this.lastWebcamFrame;
        if (!srcFrame) {
            this.setStatus("Wait for face detection before snapping", true);
            return false;
        }
        this.snapshotPose = {
            motionFrame: cloneMotionFrame(srcFrame),
            cameraPose: this.captureCameraPose(),
        };
        this.snapshotActive = true;
        this.turntablePlaying = false;
        this.clipPlaying = false;
        this.stopWebcam({ preserveSnapshot: true });

        if (this.ui.renderPanel) {
            this.ui.renderPanel.setState({ exportMode: "single" }, false);
            this.renderState = this.ui.renderPanel.getState();
        }
        if (this.reconstructedFace?.hasMesh) {
            this.reconstructedFace.applyMotionFrame(this.snapshotPose.motionFrame);
        }
        this.updateMotionUI();
        this.updateRenderUI();
        this.updateReadout();
        this.setStatus("Snapshot saved (Single mode ready)");
        return true;
    }

    startSnapCountdown(seconds = 3) {
        if (!this.webcamActive) return;
        if (this.snapCountdownTimer !== null) {
            this.cancelSnapCountdown();
            this.updateMotionUI();
            return;
        }
        this.snapCountdownRemaining = Math.max(1, Math.floor(seconds));
        this.updateMotionUI();
        this.snapCountdownTimer = setInterval(() => {
            if (this.disposed || !this.webcamActive) {
                this.cancelSnapCountdown();
                return;
            }
            this.snapCountdownRemaining -= 1;
            if (this.snapCountdownRemaining <= 0) {
                this.captureSnapshot();
            } else {
                this.updateMotionUI();
            }
        }, 1000);
    }

    toggleSnapshot() {
        if (!this.snapshotPose) return;
        this.snapshotActive = !this.snapshotActive;
        if (this.snapshotActive) {
            this.stopWebcam({ preserveSnapshot: true });
            this.turntablePlaying = false;
            this.clipPlaying = false;
            if (this.reconstructedFace?.hasMesh) {
                this.reconstructedFace.applyMotionFrame(this.snapshotPose.motionFrame);
            }
        } else {
            this.reconstructedFace?.resetPose();
        }
        this.updateMotionUI();
        this.updateRenderUI();
        this.updateReadout();
    }

    restoreSnapshotCamera() {
        const cp = this.snapshotPose?.cameraPose;
        if (!cp || !this.controls) return;
        this.applyCameraPoseState(cp);

        if (!this.snapshotActive && this.reconstructedFace?.hasMesh) {
            this.snapshotActive = true;
            this.reconstructedFace.applyMotionFrame(this.snapshotPose.motionFrame);
        }
        this.updateMotionUI();
        this.updateRenderUI();
        this.updateReadout();
    }

    clearSnapshot() {
        this.cancelSnapCountdown();
        const wasActive = this.snapshotActive;
        this.snapshotPose = null;
        this.snapshotActive = false;
        if (wasActive) {
            this.reconstructedFace?.resetPose();
        }
        this.updateMotionUI();
        this.updateRenderUI();
        this.updateReadout();
    }

    toggleRecording() {
        if (!this.webcamActive) return;
        if (this.isRecording) {
            this.isRecording = false;
            this.updateMotionUI();
            if (this.recordedClip.length > 0 && this.ui.renderPanel) {
                this.ui.renderPanel.setState({ exportMode: "clip" }, true);
            }
            return;
        }
        this.recordedClip = [];
        this.clipSource = "webcam";
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
        this.snapshotActive = false;
        this.turntablePlaying = false;
        this.clipCursor = 0;
        this.clipPlaying = true;
        this.updateMotionUI();
        this.updateRenderUI();
        this.updateReadout();
    }

    clearMotionClip() {
        this.clipPlaying = false;
        this.isRecording = false;
        this.recordedClip = [];
        this.clipSource = null;
        this.motionDriver.reset();
        this.reconstructedFace?.resetPose();
        if (this.renderState?.exportMode === "clip" && this.ui.renderPanel) {
            this.ui.renderPanel.setState({ exportMode: "turntable" }, false);
            this.renderState = this.ui.renderPanel.getState();
        }
        this.updateMotionUI();
        this.updateRenderUI();
        this.updateReadout();
    }

    updateMotionUI() {
        const mp = this.ui.mocapPanel;
        mp.webcamBtn.style.borderColor = this.webcamActive ? "#22c55e" : "rgba(255,255,255,0.16)";
        mp.webcamBtn.style.color = this.webcamActive ? "#4ade80" : "#f8fafc";
        mp.calibrateBtn.style.display = this.webcamActive ? "inline-block" : "none";
        mp.captureRow.style.display = this.webcamActive ? "flex" : "none";
        mp.recordBtn.style.borderColor = this.isRecording ? "#ef4444" : "rgba(255,255,255,0.16)";
        mp.recordBtn.style.color = this.isRecording ? "#ef4444" : "#f8fafc";
        mp.recordBtn.textContent = this.isRecording ? "Stop Rec" : "Rec";
        mp.snapTimedBtn.textContent = this.snapCountdownRemaining > 0 ? `${this.snapCountdownRemaining}s` : "3s";
        mp.snapTimedBtn.style.borderColor = this.snapCountdownRemaining > 0 ? "#f59e0b" : "rgba(255,255,255,0.16)";
        mp.snapTimedBtn.style.color = this.snapCountdownRemaining > 0 ? "#fbbf24" : "#f8fafc";

        const hasSnap = Boolean(this.snapshotPose);
        mp.snapshotRow.style.display = hasSnap ? "flex" : "none";
        mp.snapToggleBtn.textContent = this.snapshotActive ? "Snap: ON" : "Reset Pose";
        mp.snapToggleBtn.style.borderColor = this.snapshotActive ? "#38bdf8" : "rgba(255,255,255,0.16)";
        mp.snapToggleBtn.style.color = this.snapshotActive ? "#38bdf8" : "#cbd5e1";

        const hasClip = this.recordedClip.length > 0;
        mp.playbackRow.style.display = hasClip ? "flex" : "none";
        mp.playClipBtn.style.borderColor = this.clipPlaying ? "#38bdf8" : "rgba(255,255,255,0.16)";
        mp.playClipBtn.textContent = this.clipPlaying ? "Stop Clip" : `Play (${this.recordedClip.length}f)`;

        if (this.snapCountdownRemaining > 0) {
            mp.motionStatus.textContent = `Hold pose · Snap in ${this.snapCountdownRemaining}s...`;
            mp.motionStatus.style.color = "#fbbf24";
        } else if (this.isRecording) {
            mp.motionStatus.textContent = `REC ${this.recordedClip.length}f`;
            mp.motionStatus.style.color = "#f87171";
        } else if (this.webcamActive) {
            mp.motionStatus.textContent = "LIVE tracking active";
            mp.motionStatus.style.color = "#4ade80";
        } else if (this.clipPlaying) {
            mp.motionStatus.textContent = `Playing clip (${this.recordedClip.length}f)`;
            mp.motionStatus.style.color = "#38bdf8";
        } else if (this.snapshotActive && hasSnap) {
            mp.motionStatus.textContent = "Snapshot active · Single ready";
            mp.motionStatus.style.color = "#38bdf8";
        } else if (hasSnap) {
            mp.motionStatus.textContent = "Snapshot saved (mesh neutral)";
            mp.motionStatus.style.color = "#94a3b8";
        } else {
            mp.motionStatus.textContent = hasClip ? `Clip ready: ${this.recordedClip.length}f` : "No active mocap";
            mp.motionStatus.style.color = "#94a3b8";
        }
    }

    getExportPlan() {
        const mode = this.renderState?.exportMode || "turntable";
        const snapTag = this.snapshotActive && this.snapshotPose ? "Snap·" : "";
        if (mode === "single") {
            return { mode: "single", label: `${snapTag}Single`, frames: 1 };
        }
        if (mode === "clip") {
            const count = this.recordedClip.length;
            return { mode: "clip", label: count > 0 ? "Clip" : "No Clip", frames: count };
        }
        return { mode: "turntable", label: `${snapTag}Turntable`, frames: this.turntableFrames };
    }

    updateRenderUI() {
        const rp = this.ui.renderPanel;
        const plan = this.getExportPlan();
        rp.turntableBtn.style.borderColor = this.turntablePlaying ? "#38bdf8" : "rgba(255,255,255,0.16)";
        rp.turntableBtn.style.color = this.turntablePlaying ? "#38bdf8" : "#f8fafc";
        rp.turntableBtn.textContent = this.turntablePlaying ? "Stop Turntable" : "Play Turntable";
        rp.setOutputSpec(this.outputWidth, this.outputHeight, plan.frames, plan.label);
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

    async loadSingleReference(viewKey, dataUrl, file = null) {
        const resetToken = this.resetToken;
        const slotToken = ++this.viewTokens[viewKey];
        this.loadToken++;
        this.lastPythonSyncedRefs = null;
        this.refUrls[viewKey] = dataUrl;
        this.persistedRefs[viewKey] = null;
        this.ui.refStack.setPreview(viewKey, dataUrl, false);
        this.setStatus(`Extracting ${viewKey} (CPU)...`);

        const uploadPromise = file
            ? uploadReferenceImage(file, viewKey, api).catch((err) => {
                console.warn(`[YedpFace3D] Failed to persist ${viewKey} ref:`, err);
                return null;
            })
            : Promise.resolve(null);

        try {
            const [extracted, refMeta] = await Promise.all([
                extractFaceFromDataUrl(dataUrl),
                uploadPromise,
            ]);
            if (this.disposed || resetToken !== this.resetToken || slotToken !== this.viewTokens[viewKey]) return;
            this.extractedViews[viewKey] = extracted;
            if (refMeta) {
                this.persistedRefs[viewKey] = refMeta;
                setCachedLandmarks(getRefCacheKey(refMeta), extracted);
            }
            this.updateRefButtonsUI();

            if (this.extractedViews.front) {
                await this.runReconstruction();
            } else {
                this.setStatus("Load front photo to build mesh");
            }
        } catch (err) {
            if (this.disposed || resetToken !== this.resetToken || slotToken !== this.viewTokens[viewKey]) return;
            this.refUrls[viewKey] = null;
            this.persistedRefs[viewKey] = null;
            this.extractedViews[viewKey] = null;
            this.ui.refStack.setPreview(viewKey, null, false);
            this.updateRefButtonsUI();
            this.setStatus(`Error: ${err.message}`, true);
        }
    }

    async clearSingleReference(viewKey) {
        if (this.disposed || !REF_VIEW_KEYS.includes(viewKey)) return;
        const resetToken = this.resetToken;
        this.viewTokens[viewKey] = (this.viewTokens[viewKey] || 0) + 1;
        this.loadToken++;
        this.reconstructToken++;
        this.lastPythonSyncedRefs = null;
        this.refUrls[viewKey] = null;
        this.persistedRefs[viewKey] = null;
        this.extractedViews[viewKey] = null;
        this.ui.refStack.setPreview(viewKey, null, false);
        this.updateRefButtonsUI();

        if (this.weightDebugView === viewKey) {
            this.weightDebugView = "combined";
            this.reconstructedFace?.setWeightDebugView("combined");
        }

        if (viewKey === "front" || !this.extractedViews.front) {
            this.isDefaultHead = false;
            this.reconstructedFace?.clear(false);
            if (this.reconstructedFace) this.reconstructedFace.group.visible = false;
            if (this.viewMode === "textured") {
                this.setViewMode("clay");
            }
            await this.loadDefaultCanonicalHead();
            return;
        }

        if (resetToken === this.resetToken) {
            await this.runReconstruction();
        }
    }

    async loadReferenceImages({
        front,
        three_quarter = null,
        three_quarter_2 = null,
        side = null,
        side_2 = null,
        depth_weight = null,
        driving_batch_id = null,
        driving_frame_count = 0,
    }) {
        if (this.disposed) return;
        const resetToken = ++this.resetToken;
        const token = ++this.loadToken;
        this.persistedRefs = createRefSlotMap(null);

        if (depth_weight !== null && depth_weight !== undefined) {
            this.depthWeight = Number(depth_weight);
        }

        const incoming = {
            front: front || null,
            three_quarter,
            three_quarter_2,
            side,
            side_2,
        };
        this.refUrls = { ...incoming };
        for (const key of REF_VIEW_KEYS) {
            this.viewTokens[key] = (this.viewTokens[key] || 0) + 1;
            this.ui.refStack.setPreview(key, incoming[key], false);
        }
        this.setStatus("Extracting landmarks (CPU)...");

        try {
            const extractedList = await Promise.all(
                REF_VIEW_KEYS.map((key) =>
                    incoming[key] ? extractFaceFromDataUrl(incoming[key]) : Promise.resolve(null)
                )
            );

            if (this.disposed || resetToken !== this.resetToken || token !== this.loadToken) return;
            const frontData = extractedList[0];
            if (!frontData) {
                this.setStatus("No face in front_image", true);
                return;
            }

            const nextViews = createRefSlotMap(null);
            REF_VIEW_KEYS.forEach((key, idx) => {
                nextViews[key] = extractedList[idx];
            });
            this.extractedViews = nextViews;
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
                if (i % 2 === 1) {
                    await new Promise((resolve) => setTimeout(resolve, 0));
                }
            }

            if (clip.length > 0 && token === this.loadToken) {
                const hadDriving = this.clipSource === "driving";
                this.recordedClip = clip;
                this.clipSource = "driving";
                this.turntablePlaying = false;
                this.clipCursor = 0;
                this.clipPlaying = true;
                if (!hadDriving && this.ui.renderPanel) {
                    this.ui.renderPanel.setState({ exportMode: "clip" }, false);
                    this.renderState = this.ui.renderPanel.getState();
                }
                this.updateMotionUI();
                this.updateRenderUI();
                this.updateReadout();
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
            const wasDefaultHead = this.isDefaultHead;
            const hadMeshBefore = this.reconstructedFace.hasMesh && !wasDefaultHead;
            const triangles = await getTesselationTriangles();
            const wState = this.ui.weightPanel?.getState() || {};
            this.syncWeightTrackingFromPanel(wState);
            const response = await api.fetchApi("/yedp_face3d/reconstruct", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    front: this.extractedViews.front,
                    three_quarter: this.extractedViews.three_quarter,
                    three_quarter_2: this.extractedViews.three_quarter_2,
                    side: this.extractedViews.side,
                    side_2: this.extractedViews.side_2,
                    triangles,
                    depth_weight: this.depthWeight,
                    enable_reliability_weighting: this.enableReliabilityWeighting,
                    enable_anatomical_weighting: this.enableAnatomicalWeighting,
                    enable_regularization: this.enableRegularization,
                    base_geometry_anchoring: Number(wState.baseAnchor ?? 0.25),
                    surface_deformation_smoothness: Number(wState.dispSmooth ?? 0.45),
                    max_local_displacement: Number(wState.maxDisp ?? 0.18),
                    symmetry_strength: Number(wState.symmetry ?? 0.0),
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

            const availDebugViews = this.getAvailableWeightDebugViews();
            if (!availDebugViews.includes(this.weightDebugView)) {
                this.weightDebugView = "combined";
            }
            this.isDefaultHead = false;
            if (wasDefaultHead && this.viewMode === "clay") {
                this.viewMode = "textured";
                this.ui.modeBar.setActiveMode("textured");
            }
            this.reconstructedFace.setGuidesVisible(this.guidesVisible);
            this.reconstructedFace.setViewMode(this.viewMode);
            this.reconstructedFace.setWeightDebugView(this.weightDebugView);
            this.reconstructedFace.setWeightDebugVisible(this.weightDebugEnabled);
            this.syncWeightDebugButtons();
            this.reconstructedFace.applyRenderState(this.renderState);

            this.reconstructedFace.group.visible = true;
            this.updateRefButtonsUI();

            if (this.snapshotActive && this.snapshotPose?.motionFrame) {
                this.reconstructedFace.applyMotionFrame(this.snapshotPose.motionFrame);
            }

            if (this.pendingRestoredCamera) {
                const pendingCam = this.pendingRestoredCamera;
                this.pendingRestoredCamera = null;
                this.applyCameraPoseState(pendingCam);
            } else if (!hadMeshBefore && !wasDefaultHead) {
                this.fitHeadToView();
            }

            const views = ["Front"];
            if (meshData.has_three_quarter) views.push("3/4 A");
            if (meshData.has_three_quarter_2) views.push("3/4 B");
            if (meshData.has_side) views.push("Side A");
            if (meshData.has_side_2) views.push("Side B");
            this.setStatus(
                views.length <= 3
                    ? `3D Face Ready (${views.join(" + ")})`
                    : `3D Face Ready (${views.length} views · ${views.join(" + ")})`
            );
        } catch (err) {
            if (this.disposed || token !== this.reconstructToken) return;
            console.error("[YedpFace3D] Reconstruction failed:", err);
            this.setStatus(`Reconstruct error: ${err.message}`, true);
        }
    }

    async executeRenderJob(detail) {
        if (this.disposed || !detail?.job_id) return;
        const {
            job_id,
            front = null,
            three_quarter = null,
            three_quarter_2 = null,
            side = null,
            side_2 = null,
            depth_weight = null,
            driving_batch_id = null,
            driving_frame_count = 0,
            render_width = this.outputWidth,
            render_height = this.outputHeight,
            turntable_frames = this.turntableFrames,
            anatomical_guides = this.guidesVisible,
        } = detail;

        try {
            this.setGuidesVisible(anatomical_guides);
            this.setOutputResolution(render_width, render_height);
            this.setTurntableFrames(turntable_frames);

            if (!driving_batch_id && this.clipSource === "driving") {
                this.clearMotionClip();
            }

            const exportMode = this.ui.renderPanel.getState().exportMode || "turntable";
            const needDrivingDecode = Boolean(
                driving_batch_id
                && driving_frame_count > 0
                && (exportMode === "clip" || this.clipSource !== "driving")
            );

            if (front) {
                const sameRefs = this.lastPythonSyncedRefs
                    && this.lastPythonSyncedRefs.front === front
                    && this.lastPythonSyncedRefs.three_quarter === three_quarter
                    && this.lastPythonSyncedRefs.three_quarter_2 === three_quarter_2
                    && this.lastPythonSyncedRefs.side === side
                    && this.lastPythonSyncedRefs.side_2 === side_2
                    && Number(this.lastPythonSyncedRefs.depth_weight) === Number(depth_weight)
                    && this.reconstructedFace.hasMesh;

                if (!sameRefs) {
                    await this.loadReferenceImages({
                        front,
                        three_quarter,
                        three_quarter_2,
                        side,
                        side_2,
                        depth_weight,
                        driving_batch_id: needDrivingDecode ? driving_batch_id : null,
                        driving_frame_count: needDrivingDecode ? driving_frame_count : 0,
                    });
                    this.lastPythonSyncedRefs = {
                        front,
                        three_quarter,
                        three_quarter_2,
                        side,
                        side_2,
                        depth_weight,
                    };
                } else if (needDrivingDecode) {
                    await this.loadDrivingBatch(driving_batch_id, driving_frame_count, this.loadToken);
                }
            } else if (needDrivingDecode) {
                await this.loadDrivingBatch(driving_batch_id, driving_frame_count, this.loadToken);
            }

            if (this.disposed) return;

            this.renderState = this.ui.renderPanel.getState();
            this.setStatus(`Rendering ${render_width}x${render_height}...`);
            this.isExporting = true;
            this.reconstructedFace?.setWeightDebugVisible(false);
            try {
                await renderAndUploadPasses({
                    jobId: job_id,
                    renderer: this.renderer,
                    scene: this.scene,
                    liveCamera: this.camera,
                    controlsTarget: this.controls.target,
                    grid: this.grid,
                    reconstructedFace: this.reconstructedFace,
                    placeholderHead: null,
                    motionClip: this.recordedClip,
                    snapshotFrame: this.snapshotActive ? this.snapshotPose?.motionFrame : null,
                    turntableFrames: turntable_frames,
                    renderState: this.renderState,
                    width: render_width,
                    height: render_height,
                    onProgress: (done, total) => {
                        this.ui.renderPanel.renderStatus.textContent = `Exported ${done}/${total}f`;
                        this.ui.renderPanel.renderStatus.style.color = "#38bdf8";
                    },
                });
            } finally {
                this.isExporting = false;
                this.reconstructedFace?.setWeightDebugVisible(this.weightDebugEnabled);
            }

            this.updateRenderUI();
            this.setStatus(`Export Complete (${render_width}x${render_height})`);
        } catch (err) {
            console.error("[YedpFace3D] Offscreen render job failed:", err);
            this.setStatus(`Render error: ${err.message}`, true);
            await api.fetchApi(`/yedp_face3d/render_error/${job_id}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ error: err?.message || String(err) }),
            }).catch(() => {});
        }
    }

    updateRefButtonsUI() {
        for (const key of REF_VIEW_KEYS) {
            this.ui.refStack.setSlotActive(key, Boolean(this.extractedViews[key]));
        }
    }

    clearAllReferences() {
        this.resetToken += 1;
        this.loadToken++;
        this.reconstructToken++;
        this.pendingRestoredCamera = null;
        this.lastPythonSyncedRefs = null;
        this.stopWebcam();
        this.turntablePlaying = false;
        this.turntableCursor = 0;
        this.clearMotionClip();
        this.clearSnapshot();
        resetImageLandmarker();

        this.refUrls = createRefSlotMap(null);
        this.persistedRefs = createRefSlotMap(null);
        this.extractedViews = createRefSlotMap(null);
        for (const key of REF_VIEW_KEYS) {
            this.viewTokens[key] = (this.viewTokens[key] || 0) + 1;
        }
        this.ui.refStack.clearAll();

        this.ui.sculptPanel.reset(false);
        this.ui.weightPanel.reset(false);
        this.ui.lightPanel.reset(false);

        this.reconstructedFace.clear(true);
        if (this.regRebuildTimer !== null) {
            clearTimeout(this.regRebuildTimer);
            this.regRebuildTimer = null;
        }
        this.weightDebugEnabled = false;
        this.weightDebugView = "combined";
        this.enableReliabilityWeighting = true;
        this.enableAnatomicalWeighting = true;
        this.enableRegularization = true;
        this.syncWeightTrackingFromPanel(this.ui.weightPanel.getState());
        this.syncWeightDebugButtons();
        this.reconstructedFace.updateLights(this.ui.lightPanel.getState());
        this.reconstructedFace.setRegionWeights(this.ui.weightPanel.getState());
        this.reconstructedFace.setSculptState(this.ui.sculptPanel.getState());
        this.reconstructedFace.applyRenderState(this.renderState);
        this.reconstructedFace.group.visible = false;

        this.isDefaultHead = false;
        this.setViewMode("clay");
        this.resetCamera();
        this.updateRefButtonsUI();
        this.updateMotionUI();
        this.updateRenderUI();
        this.loadDefaultCanonicalHead();
    }

    exportSerializableState() {
        return {
            v: 1,
            sculpt: this.ui.sculptPanel?.getState(),
            weights: this.ui.weightPanel?.getState(),
            lights: this.ui.lightPanel?.getState(),
            render: this.ui.renderPanel?.getState(),
            viewport: {
                viewMode: this.viewMode,
                smoothEnabled: Boolean(this.reconstructedFace?.smoothEnabled ?? true),
                guidesVisible: this.guidesVisible,
            },
            refs: normalizePersistedRefs(this.persistedRefs),
            camera: packCameraPose(this.captureCameraPose()),
            snapshot: packSnapshotPose(this.snapshotPose),
            snapshotActive: Boolean(this.snapshotActive),
        };
    }

    async restoreSerializableState(saved) {
        if (this.disposed || !saved || typeof saved !== "object") return;
        const resetToken = ++this.resetToken;
        this.loadToken += 1;
        this.reconstructToken += 1;
        this.lastPythonSyncedRefs = null;
        this.stopWebcam();
        this.cancelSnapCountdown();
        this.turntablePlaying = false;
        this.clipPlaying = false;

        if (saved.sculpt && this.ui.sculptPanel?.setState) {
            this.ui.sculptPanel.setState(saved.sculpt, false);
        }
        if (saved.weights && this.ui.weightPanel?.setState) {
            this.ui.weightPanel.setState(saved.weights, false);
            this.syncWeightTrackingFromPanel(this.ui.weightPanel.getState());
        }
        if (saved.lights && this.ui.lightPanel?.setState) {
            this.ui.lightPanel.setState(saved.lights, false);
        }
        if (saved.render && this.ui.renderPanel?.setState) {
            this.ui.renderPanel.setState(saved.render, false);
            this.renderState = this.ui.renderPanel.getState();
            this.ui.renderPanel.syncProjectionUI?.();
        }

        const vp = saved.viewport || {};
        if (vp.viewMode) {
            this.setViewMode(vp.viewMode);
        }
        if (typeof vp.smoothEnabled === "boolean") {
            this.setSmoothEnabled(vp.smoothEnabled);
        }
        if (typeof vp.guidesVisible === "boolean") {
            this.setGuidesVisible(vp.guidesVisible);
            this.onGuidesChanged?.(this.guidesVisible);
        }

        this.reconstructedFace?.updateLights(this.ui.lightPanel.getState());
        this.reconstructedFace?.setRegionWeights(this.ui.weightPanel.getState());
        this.reconstructedFace?.setSculptState(this.ui.sculptPanel.getState());
        this.reconstructedFace?.applyRenderState(this.renderState);

        const camPose = unpackCameraPose(saved.camera, THREE);
        if (camPose) {
            this.applyCameraPoseState(camPose);
        }

        this.snapshotPose = unpackSnapshotPose(saved.snapshot, THREE);
        this.snapshotActive = Boolean(saved.snapshotActive && this.snapshotPose);

        const refs = normalizePersistedRefs(saved.refs);
        const hasFront = Boolean(refs.front?.filename);
        this.pendingRestoredCamera = hasFront ? camPose : null;

        this.persistedRefs = { ...refs };

        const urls = {};
        let hasAnyUrl = false;
        for (const key of REF_VIEW_KEYS) {
            const url = buildRefViewUrl(refs[key], api);
            urls[key] = url;
            if (url) hasAnyUrl = true;
            this.refUrls[key] = url;
            this.ui.refStack.setPreview(key, url, false);
        }

        if (!hasAnyUrl) {
            if (!vp.viewMode) {
                this.setViewMode("clay");
            }
            this.updateRefButtonsUI();
            this.updateMotionUI();
            this.updateRenderUI();
            this.updateReadout();
            await this.loadDefaultCanonicalHead();
            return;
        }

        this.setStatus("Restoring saved 3D face...");
        try {
            const restored = await Promise.all(
                REF_VIEW_KEYS.map(async (key) => {
                    const url = urls[key];
                    if (!url) return null;
                    const cacheKey = getRefCacheKey(refs[key]);
                    const cached = await getCachedLandmarks(cacheKey);
                    if (cached) return cached;
                    const extracted = await extractFaceFromDataUrl(url);
                    if (extracted && cacheKey) {
                        setCachedLandmarks(cacheKey, extracted);
                    }
                    return extracted;
                })
            );
            if (this.disposed || resetToken !== this.resetToken) return;

            const nextViews = createRefSlotMap(null);
            REF_VIEW_KEYS.forEach((key, idx) => {
                nextViews[key] = restored[idx];
            });
            this.extractedViews = nextViews;
            this.updateRefButtonsUI();

            if (this.extractedViews.front) {
                await this.runReconstruction();
                if (this.disposed || resetToken !== this.resetToken) return;
                if (this.snapshotActive && this.snapshotPose?.motionFrame && this.reconstructedFace?.hasMesh) {
                    this.reconstructedFace.applyMotionFrame(this.snapshotPose.motionFrame);
                }
            } else {
                this.pendingRestoredCamera = null;
                await this.loadDefaultCanonicalHead();
            }
            this.updateMotionUI();
            this.updateRenderUI();
            this.updateReadout();
        } catch (err) {
            if (this.disposed || resetToken !== this.resetToken) return;
            this.pendingRestoredCamera = null;
            console.warn("[YedpFace3D] State restore failed:", err);
            this.setStatus(`Restore partial: ${err?.message || err}`, true);
        }
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
        this.updateReadout();
    }

    resetCamera() {
        if (!this.camera || !this.controls) return;
        if (this.reconstructedFace?.hasMesh) {
            this.fitHeadToView();
            return;
        }
        this.controls.target.set(...DEFAULT_TARGET);
        const dist = new THREE.Vector3(...DEFAULT_CAMERA_POS).distanceTo(new THREE.Vector3(...DEFAULT_TARGET));
        this.orthoHalfH = dist * Math.tan(THREE.MathUtils.degToRad((this.renderState.fov || 36) * 0.5));
        this.orthoCamera.zoom = 1.0;
        this.perspCamera.position.set(...DEFAULT_CAMERA_POS);
        this.orthoCamera.position.set(...DEFAULT_CAMERA_POS);
        this.applyResize(this.viewportWidth, this.viewportHeight);
        this.controls.update();
    }

    setGuidesVisible(visible) {
        this.guidesVisible = Boolean(visible);
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
        const aspect = w / h;

        // Pass updateStyle=false so canvas CSS stays 100% and avoids ResizeObserver loops.
        this.renderer.setSize(w, h, false);
        this.perspCamera.aspect = aspect;
        this.perspCamera.updateProjectionMatrix();

        this.orthoCamera.left = -this.orthoHalfH * aspect;
        this.orthoCamera.right = this.orthoHalfH * aspect;
        this.orthoCamera.top = this.orthoHalfH;
        this.orthoCamera.bottom = -this.orthoHalfH;
        this.orthoCamera.updateProjectionMatrix();

        this.updateReadout();
    }

    updateReadout() {
        const plan = this.getExportPlan();
        this.ui.sizeReadout.textContent =
            `Viewport: ${this.viewportWidth}x${this.viewportHeight} | Export: ${this.outputWidth}x${this.outputHeight} · ${plan.frames}f (${plan.label})`;
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
            const startDeg = Number(this.renderState.startAngle ?? -90);
            const endDeg = Number(this.renderState.endAngle ?? 90);
            const yawDeg = startDeg + (endDeg - startDeg) * t;
            const yawRad = THREE.MathUtils.degToRad(yawDeg);
            if (this.reconstructedFace.hasMesh) {
                this.reconstructedFace.setTurntableYaw(yawRad);
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

        if (this.webcamActive && !this.webcamBusy && this.ui.webcamVideo.readyState >= 2 && nowMs - this.lastWebcamDetectMs > 38) {
            this.lastWebcamDetectMs = nowMs;
            this.webcamBusy = true;
            try {
                const det = await detectFaceForVideo(this.ui.webcamVideo, nowMs);
                if (!this.disposed && this.webcamActive) {
                    const frame = det
                        ? this.motionDriver.processDetection(det, { mirrorYaw: true, smoothAlpha: 0.45 })
                        : null;
                    this.ui.webcamOverlay.drawFrame(this.ui.webcamVideo, det, frame?.bs);
                    if (frame) {
                        this.lastWebcamFrame = frame;
                        this.reconstructedFace.applyMotionFrame(frame);
                        if (this.isRecording && this.recordedClip.length < MAX_RECORDED_FRAMES) {
                            this.recordedClip.push(frame);
                            this.ui.mocapPanel.motionStatus.textContent = `REC ${this.recordedClip.length}f`;
                        }
                    }
                }
            } catch {
                // Ignore transient video frame decode errors
            } finally {
                this.webcamBusy = false;
            }
        }
    }

    startRenderLoop() {
        const tick = (nowMs) => {
            if (this.disposed) return;
            this.rafId = requestAnimationFrame(tick);
            if (this.isExporting) return;
            this.stepMotion(nowMs);
            this.controls.update();
            if (this.viewMode === "depth" && this.reconstructedFace.hasMesh) {
                this.reconstructedFace.updateDepthBounds(this.camera, this.renderState.depthMode === "dynamic");
            }
            this.renderer.render(this.scene, this.camera);
        };
        this.rafId = requestAnimationFrame(tick);
    }

    dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.resetToken += 1;
        this.loadToken++;
        this.reconstructToken++;
        if (this.regRebuildTimer !== null) {
            clearTimeout(this.regRebuildTimer);
            this.regRebuildTimer = null;
        }
        this.stopWebcam();

        if (this.rafId !== null) {
            cancelAnimationFrame(this.rafId);
            this.rafId = null;
        }
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
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
