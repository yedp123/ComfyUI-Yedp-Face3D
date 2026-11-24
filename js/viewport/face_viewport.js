import { api } from "../../../../scripts/api.js";
import * as THREE from "../lib/three.module.js";
import { OrbitControls } from "../lib/OrbitControls.js";
import { getCanonicalFaceData, getHeadShellRig, ReconstructedFaceMesh } from "./face_mesh.js";
import { renderAndUploadPasses } from "./offscreen_renderer.js";
import { bakeFaceTexturePngBytes, exportFaceGlbBuffer } from "./glb_exporter.js";
import { stopViewportEvents } from "./ui_icons.js";
import {
    detectFaceForVideo,
    extractFaceFromDataUrl,
    getTesselationTriangles,
    resetImageLandmarker,
    segmentImageDataUrl,
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

function normalizeSilhStrength(raw) {
    if (raw === null || raw === undefined || raw === "") return 0.75;
    const num = Number(raw);
    return Number.isFinite(num) ? Math.max(0, Math.min(1, num)) : 0.75;
}

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
        this._defaultHeadPromise = null;
        this.guidesVisible = true;
        this.enableReliabilityWeighting = true;
        this.enableAnatomicalWeighting = true;
        this.enableRegularization = true;
        this.mvTexEnabled = true;
        this.uvDebugMode = "off";
        this.weightDebugEnabled = false;
        this.weightDebugView = "combined";
        this.weightDebugBtn = null;
        this.weightDebugViewBtn = null;
        this.relWeightBtn = null;
        this.anatWeightBtn = null;
        this.regBtn = null;
        this.mvTexBtn = null;
        this.uvDebugBtn = null;
        this.regRebuildTimer = null;
        this.lastBakeResult = null;
        this.bakeToken = 0;
        this.lastColorMatch = true;
        this.lastSeamBlend = true;
        this.lastSilhFit = true;
        this.lastSilhStrength = 0.75;
        this.lastRegState = { baseAnchor: 0.25, dispSmooth: 0.45, maxDisp: 0.18, symmetry: 0.0 };
        this.lastMocapWeightState = { eyes: 1.15, brows: 1.0, mouth: 1.0, mouthWidth: 0.55, jaw: 0.85, cheeks: 0.45 };
        this.lastIrisMaskScale = 1.08;
        this.onGuidesChanged = null;
        this.depthWeight = 1.0;
        this.frameCount = 24;
        this.viewportWidth = 480;
        this.viewportHeight = 380;
        this.outputWidth = 512;
        this.outputHeight = 512;

        this.refUrls = createRefSlotMap(null);
        this.persistedRefs = createRefSlotMap(null);
        this.extractedViews = createRefSlotMap(null);
        this.loadToken = 0;
        this.resetToken = 0;
        this.viewTokens = createRefSlotMap(0);
        this.reconstructToken = 0;
        this.videoLoadToken = 0;
        this.pendingRestoredCamera = null;

        // Motion state (Turntable / Webcam / Recorded Clip / Local Video)
        this.motionDriver = new MotionDriver();
        this.turntablePlaying = false;
        this.turntableCursor = 0;
        this.lastTickMs = 0;

        this.webcamActive = false;
        this.webcamBusy = false;
        this.webcamStream = null;
        this.isRecording = false;
        this.recStartMs = 0;
        this.lastWebcamDetectMs = 0;
        this.lastWebcamFrame = null;
        this.isExporting = false;

        this.recordedClip = [];
        this.clipFps = 24;
        this.clipSource = null;
        this.clipPlaying = false;
        this.clipCursor = 0;

        this.snapshotPose = null;
        this.snapshotActive = false;
        this.snapCountdownTimer = null;
        this.snapCountdownRemaining = 0;

        this.scene = new THREE.Scene();
        this.renderState = this.ui.renderPanel.getState();
        this.renderFov = Number(this.renderState.fov || 36);
        this.debugDrawerOpen = false;

        const initAspect = 480 / 380;
        this.perspCamera = new THREE.PerspectiveCamera(this.renderFov, initAspect, 0.05, 50);
        this.perspCamera.position.set(...DEFAULT_CAMERA_POS);

        const initDist = new THREE.Vector3(...DEFAULT_CAMERA_POS).distanceTo(new THREE.Vector3(...DEFAULT_TARGET));
        this.orthoHalfH = initDist * Math.tan(THREE.MathUtils.degToRad(this.renderFov * 0.5));
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
        this.reconstructedFace.setIrisMaskScale(this.ui.weightPanel.getState().irisMaskScale ?? 1.08);
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

    loadDefaultCanonicalHead() {
        const p = this._performLoadDefaultCanonicalHead().finally(() => {
            if (this._defaultHeadPromise === p) {
                this._defaultHeadPromise = null;
            }
        });
        this._defaultHeadPromise = p;
        return p;
    }

    async _performLoadDefaultCanonicalHead() {
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
        const mp = this.ui.mocapPanel;
        mp.webcamBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.toggleWebcam();
        });
        mp.loadVideoBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            mp.videoInput.click();
        });
        mp.videoInput.addEventListener("change", () => {
            const file = mp.videoInput.files?.[0];
            mp.videoInput.value = "";
            if (file) this.loadMocapVideoFile(file);
        });
        mp.panel.addEventListener("dragover", (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
        mp.panel.addEventListener("drop", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const file = e.dataTransfer?.files?.[0];
            if (file && (!file.type || file.type.startsWith("video/"))) {
                this.loadMocapVideoFile(file);
            }
        });
        mp.calibrateBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.motionDriver.calibrateNextFrame();
        });
        mp.recordBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.toggleRecording();
        });
        mp.snapBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.captureSnapshot();
        });
        mp.snapTimedBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.startSnapCountdown(3);
        });
        mp.snapToggleBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.toggleSnapshot();
        });
        mp.snapAngleBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.restoreSnapshotCamera();
        });
        mp.clearSnapBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.clearSnapshot();
        });
        mp.playClipBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.toggleClipPlayback();
        });
        mp.snapClipBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.captureClipSnapshot();
        });
        mp.setNeutralBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.recalibrateClipAtFrame(Math.floor(this.clipCursor));
        });
        mp.clearClipBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.clearMotionClip();
        });
        mp.clipScrubber.addEventListener("input", () => {
            this.scrubClipTo(Number(mp.clipScrubber.value));
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
        this.mvTexBtn = makeToggle(
            "MV Tex: ON",
            "Toggle Multi-View UV Texture Baking (ON = Canonical UV Bake, OFF = Original Front Projection)"
        );
        this.uvDebugBtn = makeToggle(
            "UV Debug: Off",
            "Cycle UV Debug overlay in Textured mode (Off | Source Map | Hair Mask)"
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

        const mvTexRow = document.createElement("div");
        Object.assign(mvTexRow.style, {
            display: "flex",
            alignItems: "center",
            gap: "4px",
        });
        mvTexRow.append(this.mvTexBtn, this.uvDebugBtn);

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
        this.mvTexBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.setMvTexEnabled(!this.mvTexEnabled);
        });
        this.uvDebugBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.cycleUvDebugMode();
        });

        this.debugToggleBtn = makeToggle("Debug ▾", "Toggle reconstruction & UV debug controls");
        const debugDrawer = document.createElement("div");
        Object.assign(debugDrawer.style, {
            display: "none",
            flexDirection: "column",
            alignItems: "flex-end",
            gap: "4px",
        });
        this.debugDrawer = debugDrawer;
        this.debugToggleBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this.debugDrawerOpen = !this.debugDrawerOpen;
            debugDrawer.style.display = this.debugDrawerOpen ? "flex" : "none";
            this.debugToggleBtn.textContent = this.debugDrawerOpen ? "Debug ▴" : "Debug ▾";
            this.debugToggleBtn.style.borderColor = this.debugDrawerOpen ? "#38bdf8" : "rgba(255, 255, 255, 0.16)";
            this.debugToggleBtn.style.color = this.debugDrawerOpen ? "#38bdf8" : "#cbd5e1";
        });

        debugDrawer.append(debugRow, this.relWeightBtn, phase23Row, mvTexRow);
        bar.append(this.debugToggleBtn, debugDrawer);
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
        const hasMultiView = this.getLoadedViewCount() > 1;
        const bakedReady = Boolean(this.reconstructedFace?.hasBakedTextures?.());
        if (this.mvTexBtn) {
            const on = this.mvTexEnabled;
            this.mvTexBtn.textContent = on ? "MV Tex: ON" : "MV Tex: OFF";
            this.mvTexBtn.style.borderColor = on ? "#22c55e" : "#f59e0b";
            this.mvTexBtn.style.color = on ? "#4ade80" : "#fbbf24";
        }
        if (this.uvDebugBtn) {
            const uvLabels = { off: "Off", source: "Source Map", hair: "Hair Mask" };
            const canDebug = this.mvTexEnabled && hasMultiView && bakedReady;
            const activeUv = canDebug ? this.uvDebugMode : "off";
            this.uvDebugBtn.textContent = `UV Debug: ${uvLabels[activeUv] || "Off"}`;
            this.uvDebugBtn.disabled = !canDebug;
            this.uvDebugBtn.style.opacity = canDebug ? "1.0" : "0.5";
            this.uvDebugBtn.style.cursor = canDebug ? "pointer" : "default";
            this.uvDebugBtn.style.borderColor = activeUv === "off" ? "rgba(255, 255, 255, 0.16)" : "#38bdf8";
            this.uvDebugBtn.style.color = activeUv === "off" ? "#cbd5e1" : "#38bdf8";
        }
    }

    getLoadedViewCount() {
        let count = 0;
        for (const key of REF_VIEW_KEYS) {
            if (this.extractedViews[key]?.landmarks) count++;
        }
        return count;
    }

    stripSegMaskForReconstruct(extracted, includeSegMask = false) {
        if (!extracted) return null;
        const out = {
            landmarks: extracted.landmarks,
            matrix: extracted.matrix || null,
            width: extracted.width,
            height: extracted.height,
        };
        if (includeSegMask) {
            out.seg_mask_png = extracted.segMaskPng || null;
        }
        return out;
    }

    getBakeSignature() {
        const parts = REF_VIEW_KEYS.map((k) => {
            const ext = this.extractedViews[k];
            if (!ext?.landmarks) return "";
            const refKey = this.persistedRefs[k]?.filename || `url${this.refUrls[k]?.length || 0}`;
            const segFlag = ext.segMaskPng ? `s${ext.segMaskPng.length}` : "noseg";
            return `${k}@${this.viewTokens[k] || 0}:${refKey}:${segFlag}`;
        });
        const cmFlag = this.ui.weightPanel?.getState()?.enableColorMatch !== false ? 1 : 0;
        const sbFlag = this.ui.weightPanel?.getState()?.enableSeamBlend !== false ? 1 : 0;
        return `${this.resetToken}|cm=${cmFlag}|sb=${sbFlag}|${parts.join("|")}`;
    }

    async fetchUvBakeData() {
        if (this.disposed || !this.extractedViews.front || this.getLoadedViewCount() <= 1) {
            this.lastBakeResult = null;
            return null;
        }
        const sig = this.getBakeSignature();
        if (this.lastBakeResult?.signature === sig) {
            return this.lastBakeResult.data;
        }
        const token = ++this.bakeToken;
        const triangles = await getTesselationTriangles();
        const views = {};
        for (const key of REF_VIEW_KEYS) {
            const ext = this.extractedViews[key];
            if (!ext?.landmarks) continue;
            const refMeta = this.persistedRefs[key];
            const entry = {
                landmarks: ext.landmarks,
                matrix: ext.matrix || null,
                width: ext.width,
                height: ext.height,
                seg_mask_png: ext.segMaskPng || null,
            };
            if (refMeta?.filename) {
                entry.image_ref = refMeta;
            }
            if (!refMeta?.filename && this.refUrls[key]) {
                entry.image_data_url = this.refUrls[key];
            }
            views[key] = entry;
        }
        const response = await api.fetchApi("/yedp_face3d/bake_uv", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                views,
                triangles,
                atlas_size: 1024,
                debug_size: 512,
                enable_color_match: this.ui.weightPanel?.getState()?.enableColorMatch !== false,
                enable_seam_blend: this.ui.weightPanel?.getState()?.enableSeamBlend !== false,
            }),
        });
        if (!response.ok) {
            const errBody = await response.json().catch(() => ({}));
            throw new Error(errBody.error || `UV bake HTTP ${response.status}`);
        }
        const bakeJson = await response.json();
        if (this.disposed || token !== this.bakeToken) return null;
        const bakeData = {
            bakedAlbedoUrl: bakeJson.baked_albedo,
            sourceMapUrl: bakeJson.debug_source_map,
            hairMaskUrl: bakeJson.debug_hair_mask,
            stats: bakeJson.stats || null,
        };
        this.lastBakeResult = { signature: sig, data: bakeData };
        return bakeData;
    }

    async runUvBake({ applyToMesh = true } = {}) {
        const bakeData = await this.fetchUvBakeData();
        if (!bakeData || this.disposed) return null;
        if (applyToMesh && this.reconstructedFace?.hasMesh) {
            await this.reconstructedFace.setBakedTextures(bakeData);
            this.reconstructedFace.setMvTexEnabled(this.mvTexEnabled);
            this.reconstructedFace.setUvDebugMode(this.mvTexEnabled ? this.uvDebugMode : "off");
            this.syncWeightDebugButtons();
        }
        return bakeData;
    }

    async setMvTexEnabled(enabled) {
        const next = Boolean(enabled);
        if (next === this.mvTexEnabled) return;
        this.mvTexEnabled = next;
        if (!next) {
            this.uvDebugMode = "off";
        }
        this.reconstructedFace?.setMvTexEnabled(next);
        this.reconstructedFace?.setUvDebugMode(next ? this.uvDebugMode : "off");
        this.syncWeightDebugButtons();

        if (next && this.extractedViews.front && this.getLoadedViewCount() > 1 && !this.reconstructedFace?.hasBakedTextures?.()) {
            const prevStatus = this.ui?.badge?.textContent || "";
            this.setStatus("Baking Multi-View UV Texture...");
            try {
                await this.runUvBake({ applyToMesh: true });
                if (!this.disposed && prevStatus) {
                    this.setStatus(prevStatus);
                }
            } catch (err) {
                console.warn("[YedpFace3D] UV bake failed; keeping front texture:", err);
                if (!this.disposed && prevStatus) {
                    this.setStatus(prevStatus);
                }
            }
        }
    }

    cycleUvDebugMode() {
        if (!this.mvTexEnabled || this.getLoadedViewCount() <= 1 || !this.reconstructedFace?.hasBakedTextures?.()) {
            return;
        }
        const order = ["off", "source", "hair"];
        const curIdx = order.indexOf(this.uvDebugMode);
        const next = order[(curIdx + 1) % order.length];
        this.setUvDebugMode(next);
    }

    setUvDebugMode(mode) {
        const valid = ["off", "source", "hair"];
        this.uvDebugMode = valid.includes(mode) ? mode : "off";
        if (this.uvDebugMode !== "off" && this.viewMode !== "textured") {
            this.setViewMode("textured");
        }
        this.reconstructedFace?.setUvDebugMode(this.mvTexEnabled ? this.uvDebugMode : "off");
        this.syncWeightDebugButtons();
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
            fov: this.renderFov,
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
        this.renderFov = Number(cp.fov || 36);
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
        this.lastColorMatch = weightState.enableColorMatch !== false;
        this.lastSeamBlend = weightState.enableSeamBlend !== false;
        this.lastSilhFit = weightState.enableSilhouetteFit !== false;
        this.lastSilhStrength = normalizeSilhStrength(weightState.silhouetteStrength);
        const irisMaskScale = Number(weightState.irisMaskScale ?? 1.08);
        this.lastIrisMaskScale = Number.isFinite(irisMaskScale) ? irisMaskScale : 1.08;
    }

    updateWeights(weightState) {
        if (!weightState) return;
        const irisMaskScale = Number(weightState.irisMaskScale ?? 1.08);
        if (Number.isFinite(irisMaskScale) && irisMaskScale !== this.lastIrisMaskScale) {
            this.lastIrisMaskScale = irisMaskScale;
            this.reconstructedFace?.setIrisMaskScale(irisMaskScale);
        }
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

        const silhFit = weightState.enableSilhouetteFit !== false;
        const silhStrength = normalizeSilhStrength(weightState.silhouetteStrength);
        const silhFitChanged = silhFit !== this.lastSilhFit;
        const silhStrengthChanged = silhStrength !== this.lastSilhStrength;
        if (silhFitChanged || silhStrengthChanged) {
            this.lastSilhFit = silhFit;
            this.lastSilhStrength = silhStrength;
        }

        const colorMatch = weightState.enableColorMatch !== false;
        const seamBlend = weightState.enableSeamBlend !== false;
        const uvBakeOptChanged = colorMatch !== this.lastColorMatch || seamBlend !== this.lastSeamBlend;
        if (uvBakeOptChanged) {
            this.lastColorMatch = colorMatch;
            this.lastSeamBlend = seamBlend;
            this.lastBakeResult = null;
        }

        if ((regChanged || silhStrengthChanged || silhFitChanged) && this.extractedViews.front) {
            if (silhFitChanged && !regChanged && !silhStrengthChanged) {
                if (this.regRebuildTimer !== null) {
                    clearTimeout(this.regRebuildTimer);
                    this.regRebuildTimer = null;
                }
                this.runReconstruction();
            } else {
                this.scheduleReconstruction();
            }
        } else if (
            uvBakeOptChanged
            && this.extractedViews.front
            && this.mvTexEnabled
            && this.getLoadedViewCount() > 1
        ) {
            Promise.resolve(this.runUvBake({ applyToMesh: true })).catch((err) => {
                console.warn("[YedpFace3D][UVBaker] Re-bake on UV option toggle failed:", err);
            });
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
            this.renderFov = Number(state.fov || 36);
            this.applyResize(this.viewportWidth, this.viewportHeight);
        } else if (changedKey === "showGate") {
            this.applyResize(this.viewportWidth, this.viewportHeight);
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
            if (changedKey === "reset") {
                this.renderFov = Number(state.fov || 36);
            }
            if (state.projection !== prevProj || changedKey === "reset") {
                this.switchCameraProjection(state.projection);
            }
            if (changedKey === "reset") {
                this.resetCamera();
            }
            this.applyResize(this.viewportWidth, this.viewportHeight);
        }
        if (
            changedKey === "motionMode"
            || changedKey === "spinMode"
            || changedKey === "clipStart"
            || changedKey === "lockHeadPose"
            || changedKey === "reset"
        ) {
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
            this.orthoHalfH = dist * Math.tan(THREE.MathUtils.degToRad(this.renderFov * 0.5));
            this.orthoCamera.zoom = 1.0;
            this.orthoCamera.position.copy(oldCam.position);
        } else {
            const visHalfH = this.orthoHalfH / Math.max(this.orthoCamera.zoom, 1e-4);
            this.renderFov = Number(this.renderState.fov || 36);
            const newDist = Math.max(visHalfH / Math.tan(THREE.MathUtils.degToRad(this.renderFov * 0.5)), 0.45);
            this.perspCamera.position.copy(target).addScaledVector(dir, newDist);
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
        const limitingAspect = outAspect;
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
            this.renderFov = Number(rs.fov || 36);
            const fovRad = THREE.MathUtils.degToRad(this.renderFov * 0.5);
            const dist = Math.max(reqHalfH / Math.tan(fovRad), size.z * 0.8 + 0.6);
            this.perspCamera.position.set(
                center.x,
                center.y + Math.sin(elRad) * dist,
                center.z + Math.cos(elRad) * dist
            );
            this.applyResize(this.viewportWidth, this.viewportHeight);
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
        this.videoLoadToken += 1;
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
            this.ui.renderPanel.setState({ motionMode: "snap", spinMode: "single" }, false);
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
        if (this.renderState?.motionMode === "snap" && this.ui.renderPanel) {
            this.ui.renderPanel.setState({ motionMode: "rest" }, false);
            this.renderState = this.ui.renderPanel.getState();
        }
        this.updateMotionUI();
        this.updateRenderUI();
        this.updateReadout();
    }

    toggleRecording() {
        if (!this.webcamActive) return;
        if (this.isRecording) {
            this.isRecording = false;
            const elapsedSec = Math.max((performance.now() - this.recStartMs) / 1000, 0.1);
            if (this.recordedClip.length > 0) {
                this.clipFps = Math.max(8, Math.min(30, Math.round(this.recordedClip.length / elapsedSec)));
                this.clipCursor = 0;
            }
            this.updateMotionUI();
            if (this.recordedClip.length > 0 && this.ui.renderPanel) {
                const curSpin = this.renderState?.spinMode || "turntable";
                this.ui.renderPanel.setState(
                    { motionMode: "clip", spinMode: curSpin === "single" ? "static" : curSpin },
                    true
                );
            }
            return;
        }
        this.recordedClip = [];
        this.clipFps = 24;
        this.clipCursor = 0;
        this.clipSource = "webcam";
        this.recStartMs = performance.now();
        this.motionDriver.calibrateNextFrame();
        this.isRecording = true;
        this.updateMotionUI();
    }

    toggleClipPlayback() {
        if (!this.recordedClip.length) return;
        if (this.clipPlaying) {
            this.clipPlaying = false;
            this.updateMotionUI();
            return;
        }
        this.stopWebcam();
        this.snapshotActive = false;
        this.turntablePlaying = false;
        if (this.clipCursor >= this.recordedClip.length) {
            this.clipCursor = 0;
        }
        this.clipPlaying = true;
        this.updateMotionUI();
        this.updateRenderUI();
        this.updateReadout();
    }

    scrubClipTo(index) {
        if (!this.recordedClip.length) return;
        const clamped = Math.max(0, Math.min(this.recordedClip.length - 1, Math.floor(index)));
        this.clipPlaying = false;
        this.snapshotActive = false;
        this.turntablePlaying = false;
        this.clipCursor = clamped;
        if (this.reconstructedFace?.hasMesh) {
            this.reconstructedFace.applyMotionFrame(this.recordedClip[clamped]);
        }
        this.updateMotionUI();
    }

    captureClipSnapshot(index = Math.floor(this.clipCursor)) {
        if (!this.recordedClip.length) return false;
        const clamped = Math.max(0, Math.min(this.recordedClip.length - 1, Math.floor(index)));
        const srcFrame = this.recordedClip[clamped];
        if (!srcFrame) return false;

        this.clipPlaying = false;
        this.turntablePlaying = false;
        this.snapshotPose = {
            motionFrame: cloneMotionFrame(srcFrame),
            cameraPose: this.captureCameraPose(),
        };
        this.snapshotActive = true;
        if (this.ui.renderPanel) {
            this.ui.renderPanel.setState({ motionMode: "snap", spinMode: "single" }, false);
            this.renderState = this.ui.renderPanel.getState();
        }
        if (this.reconstructedFace?.hasMesh) {
            this.reconstructedFace.applyMotionFrame(this.snapshotPose.motionFrame);
        }
        this.updateMotionUI();
        this.updateRenderUI();
        this.updateReadout();
        this.setStatus(`Snapshot saved from frame ${clamped + 1} (Single mode ready)`);
        return true;
    }

    recalibrateClipAtFrame(index = Math.floor(this.clipCursor)) {
        if (!this.recordedClip.length) return false;
        const clamped = Math.max(0, Math.min(this.recordedClip.length - 1, Math.floor(index)));
        const refRaw = this.recordedClip[clamped]?._rawDet;
        if (!refRaw) {
            this.setStatus("Current clip lacks raw landmarks for neutral recalibration", true);
            return false;
        }
        const mirror = Boolean(this.recordedClip[clamped].mirror);
        const drv = new MotionDriver();
        drv.processDetection(refRaw, { mirrorYaw: mirror, smoothAlpha: 1.0 });
        drv.hasPrev = false;
        drv.smoothedDeltas = null;

        const nextClip = [];
        for (let i = 0; i < this.recordedClip.length; i++) {
            const raw = this.recordedClip[i]?._rawDet;
            if (!raw) continue;
            const m = drv.processDetection(raw, {
                mirrorYaw: Boolean(this.recordedClip[i].mirror),
                smoothAlpha: i === clamped ? 1.0 : (mirror ? 0.45 : 0.55),
            });
            if (m) {
                m._rawDet = raw;
                nextClip.push(m);
            }
        }
        if (!nextClip.length) return false;

        this.recordedClip = nextClip;
        this.clipCursor = Math.min(clamped, nextClip.length - 1);
        if (this.reconstructedFace?.hasMesh) {
            this.reconstructedFace.applyMotionFrame(this.recordedClip[this.clipCursor]);
        }
        this.updateMotionUI();
        this.setStatus(`Neutral recalibrated at frame ${this.clipCursor + 1}`);
        return true;
    }

    async loadMocapVideoFile(file) {
        if (!file || this.disposed) return;
        const token = ++this.videoLoadToken;
        this.stopWebcam();
        this.turntablePlaying = false;
        this.clipPlaying = false;
        this.snapshotActive = false;

        const mp = this.ui.mocapPanel;
        const objectUrl = URL.createObjectURL(file);
        const video = document.createElement("video");
        video.muted = true;
        video.playsInline = true;
        video.preload = "auto";

        try {
            mp.motionStatus.textContent = "Loading video metadata...";
            mp.motionStatus.style.color = "#38bdf8";

            await new Promise((resolve, reject) => {
                video.onloadedmetadata = () => resolve();
                video.onerror = () => reject(new Error("Unsupported or unreadable video file."));
                video.src = objectUrl;
            });
            if (this.disposed || token !== this.videoLoadToken) return;

            const duration = Number(video.duration);
            if (!Number.isFinite(duration) || duration <= 0) {
                throw new Error("Could not read video duration.");
            }

            const targetFps = 24;
            const totalSteps = Math.min(MAX_RECORDED_FRAMES, Math.max(1, Math.floor(duration * targetFps)));
            const baseTs = performance.now();
            this.motionDriver.reset();
            const clip = [];

            for (let i = 0; i < totalSteps; i++) {
                if (this.disposed || token !== this.videoLoadToken) return;
                const targetTime = Math.min(Math.max(duration - 0.001, 0), i / targetFps);
                if (Math.abs(video.currentTime - targetTime) > 1e-4) {
                    await new Promise((resolve) => {
                        const onSeeked = () => {
                            video.removeEventListener("seeked", onSeeked);
                            resolve();
                        };
                        video.addEventListener("seeked", onSeeked);
                        video.currentTime = targetTime;
                    });
                }
                if (this.disposed || token !== this.videoLoadToken) return;

                const det = await detectFaceForVideo(video, baseTs + i * (1000 / targetFps));
                if (this.disposed || token !== this.videoLoadToken) return;

                if (det) {
                    const m = this.motionDriver.processDetection(det, { mirrorYaw: false, smoothAlpha: 0.55 });
                    if (m) {
                        m._rawDet = det;
                        clip.push(m);
                    }
                } else if (clip.length > 0) {
                    clip.push(clip[clip.length - 1]);
                }

                if ((i + 1) % 4 === 0 || i + 1 === totalSteps) {
                    mp.motionStatus.textContent = `Loading video ${i + 1}/${totalSteps}f...`;
                }
                if (i % 2 === 1) {
                    await new Promise((resolve) => setTimeout(resolve, 0));
                }
            }

            if (this.disposed || token !== this.videoLoadToken) return;
            if (!clip.length) {
                throw new Error("No face detected in video.");
            }

            this.recordedClip = clip;
            this.clipFps = targetFps;
            this.clipSource = "video";
            this.clipCursor = 0;
            this.clipPlaying = true;
            if (this.ui.renderPanel) {
                const curSpin = this.renderState?.spinMode || "turntable";
                this.ui.renderPanel.setState(
                    { motionMode: "clip", spinMode: curSpin === "single" ? "static" : curSpin },
                    false
                );
                this.renderState = this.ui.renderPanel.getState();
            }
            if (this.reconstructedFace?.hasMesh) {
                this.reconstructedFace.applyMotionFrame(clip[0]);
            }
            this.updateMotionUI();
            this.updateRenderUI();
            this.updateReadout();
            this.setStatus(`Video Mocap Ready (${clip.length}f @ ${targetFps}fps)`);
        } catch (err) {
            if (this.disposed || token !== this.videoLoadToken) return;
            console.error("[YedpFace3D] Video mocap load failed:", err);
            mp.motionStatus.textContent = `Video error: ${err.message}`;
            mp.motionStatus.style.color = "#f87171";
            this.setStatus(`Video error: ${err.message}`, true);
        } finally {
            URL.revokeObjectURL(objectUrl);
            video.removeAttribute("src");
            video.load?.();
        }
    }

    clearMotionClip() {
        this.videoLoadToken += 1;
        this.clipPlaying = false;
        this.isRecording = false;
        this.recordedClip = [];
        this.clipFps = 24;
        this.clipCursor = 0;
        this.clipSource = null;
        this.motionDriver.reset();
        this.reconstructedFace?.resetPose();
        if (this.renderState?.motionMode === "clip" && this.ui.renderPanel) {
            this.ui.renderPanel.setState({ motionMode: "rest" }, false);
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
        const curIdx = hasClip ? Math.max(0, Math.min(this.recordedClip.length - 1, Math.floor(this.clipCursor))) : 0;
        mp.playbackRow.style.display = hasClip ? "flex" : "none";
        mp.playClipBtn.style.borderColor = this.clipPlaying ? "#38bdf8" : "rgba(255,255,255,0.16)";
        mp.playClipBtn.textContent = this.clipPlaying ? "Pause" : "Play";
        mp.clipScrubber.max = String(Math.max(0, this.recordedClip.length - 1));
        mp.clipScrubber.value = String(curIdx);
        mp.clipFrameLabel.textContent = hasClip ? `${curIdx + 1}/${this.recordedClip.length}` : "0/0";
        mp.setNeutralBtn.disabled = !hasClip || !this.recordedClip[curIdx]?._rawDet;
        mp.setNeutralBtn.style.opacity = mp.setNeutralBtn.disabled ? "0.45" : "1.0";
        this.ui.exportPanel?.setAnimEnabled(hasClip, this.recordedClip.length);

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
            mp.motionStatus.textContent = `Playing ${curIdx + 1}/${this.recordedClip.length}f @ ${this.clipFps || 24}fps`;
            mp.motionStatus.style.color = "#38bdf8";
        } else if (this.snapshotActive && hasSnap) {
            mp.motionStatus.textContent = "Snapshot active · Single ready";
            mp.motionStatus.style.color = "#38bdf8";
        } else if (hasSnap) {
            mp.motionStatus.textContent = "Snapshot saved (mesh neutral)";
            mp.motionStatus.style.color = "#94a3b8";
        } else {
            mp.motionStatus.textContent = hasClip ? `Clip ready: ${this.recordedClip.length}f @ ${this.clipFps || 24}fps` : "No active mocap";
            mp.motionStatus.style.color = "#94a3b8";
        }
    }

    getExportPlan() {
        const rs = this.renderState || {};
        const motionMode = rs.motionMode || "rest";
        const spinMode = rs.spinMode || "turntable";
        const reqFrames = Math.max(0, Math.floor(this.frameCount));
        const hasClip = this.recordedClip.length > 0;
        const startIdx = hasClip
            ? Math.max(0, Math.min(this.recordedClip.length - 1, Math.floor((rs.clipStart ?? 1) - 1)))
            : 0;
        const availClip = hasClip ? Math.max(1, this.recordedClip.length - startIdx) : 0;

        let frames;
        if (spinMode === "single") {
            frames = 1;
        } else if (motionMode === "clip") {
            frames = hasClip ? (reqFrames > 0 ? Math.min(reqFrames, availClip) : availClip) : 0;
        } else {
            frames = reqFrames > 0 ? reqFrames : 1;
        }

        const mLabel = motionMode === "clip" ? (hasClip ? "Clip" : "No Clip") : motionMode === "snap" ? "Snap" : "Rest";
        const sLabel = spinMode === "single" ? "1f" : spinMode === "turntable" ? "Turn" : "Static";
        return { motionMode, spinMode, startIdx, label: `${mLabel}·${sLabel}`, frames };
    }

    _triggerLocalDownload(bytes, filename, mimeType) {
        const blob = new Blob([bytes], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.style.display = "none";
        document.body?.appendChild?.(a);
        a.click();
        a.remove?.();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    async exportUvTexturePng() {
        const ep = this.ui.exportPanel;
        if (!this.reconstructedFace?.hasMesh) {
            ep?.setStatus("Load or wait for 3D mesh first", true);
            return;
        }
        try {
            ep?.setStatus("Exporting UV PNG...");
            const pngBytes = await bakeFaceTexturePngBytes(this.reconstructedFace, 1024);
            const filename = "yedp_face_uv.png";
            this._triggerLocalDownload(pngBytes, filename, "image/png");
            ep?.setStatus(`Saved ${filename} (${(pngBytes.byteLength / 1024).toFixed(0)} KB)`);
        } catch (err) {
            console.error("[YedpFace3D] UV PNG export failed:", err);
            ep?.setStatus(`Export error: ${err.message}`, true);
        }
    }

    async exportGlb(mode, opts = {}) {
        const ep = this.ui.exportPanel;
        if (!this.reconstructedFace?.hasMesh) {
            ep?.setStatus("Load or wait for 3D mesh first", true);
            return;
        }
        try {
            ep?.setStatus("Building GLB...");
            const glbBytes = await exportFaceGlbBuffer(this.reconstructedFace, {
                mode,
                scope: opts.scope || "head",
                subdiv: opts.subdiv || "smooth",
                pose: opts.pose || "rest",
                includeEyeballs: opts.includeEyeballs !== false,
                animMode: opts.animMode || "arkit",
                motionClip: this.recordedClip,
                clipFps: this.clipFps || 24,
            });
            const scopeTag = opts.scope === "face" ? "face" : "head";
            const nameMap = {
                static: `yedp_${scopeTag}_static.glb`,
                blendshapes: `yedp_${scopeTag}_blendshapes.glb`,
                animation: `yedp_${scopeTag}_anim.glb`,
            };
            const filename = nameMap[mode] || "yedp_face3d.glb";
            this._triggerLocalDownload(glbBytes, filename, "model/gltf-binary");
            const mb = (glbBytes.byteLength / (1024 * 1024)).toFixed(2);
            ep?.setStatus(`Saved ${filename} (${mb} MB)`);
        } catch (err) {
            console.error("[YedpFace3D] GLB export failed:", err);
            ep?.setStatus(`Export error: ${err.message}`, true);
        }
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
            const [extracted, segMaskPng, refMeta] = await Promise.all([
                extractFaceFromDataUrl(dataUrl),
                segmentImageDataUrl(dataUrl),
                uploadPromise,
            ]);
            if (this.disposed || resetToken !== this.resetToken || slotToken !== this.viewTokens[viewKey]) return;
            if (extracted) {
                extracted.segMaskPng = segMaskPng || null;
            }
            this.lastBakeResult = null;
            this.bakeToken++;
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
        this.bakeToken++;
        this.lastBakeResult = null;
        this.refUrls[viewKey] = null;
        this.persistedRefs[viewKey] = null;
        this.extractedViews[viewKey] = null;
        this.ui.refStack.setPreview(viewKey, null, false);
        this.updateRefButtonsUI();

        if (this.weightDebugView === viewKey) {
            this.weightDebugView = "combined";
            this.reconstructedFace?.setWeightDebugView("combined");
        }
        if (this.getLoadedViewCount() <= 1) {
            this.uvDebugMode = "off";
            this.reconstructedFace?.setUvDebugMode("off");
            this.reconstructedFace?.clearBakedTextures();
            this.syncWeightDebugButtons();
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
                    front: this.stripSegMaskForReconstruct(this.extractedViews.front, false),
                    three_quarter: this.stripSegMaskForReconstruct(this.extractedViews.three_quarter, false),
                    three_quarter_2: this.stripSegMaskForReconstruct(this.extractedViews.three_quarter_2, false),
                    side: this.stripSegMaskForReconstruct(this.extractedViews.side, true),
                    side_2: this.stripSegMaskForReconstruct(this.extractedViews.side_2, true),
                    triangles,
                    depth_weight: this.depthWeight,
                    enable_reliability_weighting: this.enableReliabilityWeighting,
                    enable_anatomical_weighting: this.enableAnatomicalWeighting,
                    enable_regularization: this.enableRegularization,
                    base_geometry_anchoring: Number(wState.baseAnchor ?? 0.25),
                    surface_deformation_smoothness: Number(wState.dispSmooth ?? 0.45),
                    max_local_displacement: Number(wState.maxDisp ?? 0.18),
                    symmetry_strength: Number(wState.symmetry ?? 0.0),
                    enable_silhouette_fit: wState.enableSilhouetteFit !== false,
                    silhouette_strength: normalizeSilhStrength(wState.silhouetteStrength),
                }),
            });

            if (!response.ok) {
                const errBody = await response.json().catch(() => ({}));
                throw new Error(errBody.error || `HTTP ${response.status}`);
            }

            const meshData = await response.json();
            if (this.disposed || token !== this.reconstructToken) return;

            const shouldBakeUv = this.mvTexEnabled && this.getLoadedViewCount() > 1;
            let bakeData = null;
            if (shouldBakeUv) {
                try {
                    bakeData = await this.fetchUvBakeData();
                } catch (bakeErr) {
                    console.warn("[YedpFace3D] UV bake failed; falling back to front projection:", bakeErr);
                    bakeData = null;
                }
                if (this.disposed || token !== this.reconstructToken) return;
            } else if (this.getLoadedViewCount() <= 1) {
                this.lastBakeResult = null;
                this.uvDebugMode = "off";
            }

            this.reconstructedFace.setSculptState(this.ui.sculptPanel.getState());
            this.reconstructedFace.setRegionWeights(this.ui.weightPanel.getState());
            await this.reconstructedFace.updateFromReconstruction(meshData, this.refUrls.front);
            if (this.disposed || token !== this.reconstructToken) return;

            this.reconstructedFace.setMvTexEnabled(this.mvTexEnabled);
            if (shouldBakeUv && bakeData) {
                await this.reconstructedFace.setBakedTextures(bakeData);
                if (this.disposed || token !== this.reconstructToken) return;
                this.reconstructedFace.setUvDebugMode(this.uvDebugMode);
            } else {
                this.reconstructedFace.clearBakedTextures();
                this.reconstructedFace.setUvDebugMode("off");
            }

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
            depth_weight = null,
            render_width = this.outputWidth,
            render_height = this.outputHeight,
            frame_count = this.frameCount,
            anatomical_guides = this.guidesVisible,
        } = detail;

        try {
            if (!this.reconstructedFace?.hasMesh && this._defaultHeadPromise) {
                try {
                    await this._defaultHeadPromise;
                } catch {
                    // Never hang the export queue if default asset loading failed
                }
                if (this.disposed) return;
            }

            this.setGuidesVisible(anatomical_guides);
            this.setOutputResolution(render_width, render_height);
            this.setFrameCount(frame_count);
            if (depth_weight !== null && depth_weight !== undefined) {
                this.setDepthWeight(depth_weight);
            }

            if (this.disposed) return;

            if (this.mvTexEnabled && this.getLoadedViewCount() > 1 && !this.reconstructedFace?.hasBakedTextures?.()) {
                try {
                    await this.runUvBake({ applyToMesh: true });
                } catch (bakeErr) {
                    console.warn("[YedpFace3D] Pre-export UV bake failed; using front projection:", bakeErr);
                }
                if (this.disposed) return;
            }

            this.renderState = this.ui.renderPanel.getState();
            this.setStatus(`Rendering ${render_width}x${render_height}...`);
            this.isExporting = true;
            this.reconstructedFace?.setWeightDebugVisible(false);
            this.reconstructedFace?.setUvDebugMode("off");
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
                    clipFps: this.clipFps || 24,
                    snapshotFrame: this.snapshotActive ? this.snapshotPose?.motionFrame : null,
                    frameCount: frame_count,
                    renderState: this.renderState,
                    orthoHalfH: this.orthoHalfH,
                    renderFov: this.renderFov,
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
                this.reconstructedFace?.setUvDebugMode(this.mvTexEnabled ? this.uvDebugMode : "off");
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
        this.bakeToken++;
        this.videoLoadToken++;
        this.lastBakeResult = null;
        this.pendingRestoredCamera = null;
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
        this.mvTexEnabled = true;
        this.uvDebugMode = "off";
        this.syncWeightTrackingFromPanel(this.ui.weightPanel.getState());
        this.syncWeightDebugButtons();
        this.reconstructedFace.updateLights(this.ui.lightPanel.getState());
        this.reconstructedFace.setRegionWeights(this.ui.weightPanel.getState());
        this.reconstructedFace.setIrisMaskScale(this.ui.weightPanel.getState().irisMaskScale ?? 1.08);
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
                enableReliabilityWeighting: Boolean(this.enableReliabilityWeighting),
                enableAnatomicalWeighting: Boolean(this.enableAnatomicalWeighting),
                enableRegularization: Boolean(this.enableRegularization),
                mvTexEnabled: Boolean(this.mvTexEnabled),
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
        this.stopWebcam();
        this.cancelSnapCountdown();
        this.turntablePlaying = false;
        this.clipPlaying = false;

        if (saved.sculpt && this.ui.sculptPanel?.setState) {
            this.ui.sculptPanel.setState(saved.sculpt, false);
        }
        if (saved.weights && this.ui.weightPanel?.setState) {
            this.ui.weightPanel.setState(
                {
                    ...saved.weights,
                    enableSilhouetteFit: saved.weights.enableSilhouetteFit !== false,
                    silhouetteStrength: normalizeSilhStrength(saved.weights.silhouetteStrength),
                    enableColorMatch: saved.weights.enableColorMatch !== false,
                    enableSeamBlend: saved.weights.enableSeamBlend !== false,
                    irisMaskScale: saved.weights.irisMaskScale ?? 1.08,
                },
                false,
            );
            this.syncWeightTrackingFromPanel(this.ui.weightPanel.getState());
        }
        if (saved.lights && this.ui.lightPanel?.setState) {
            this.ui.lightPanel.setState(saved.lights, false);
        }
        if (saved.render && this.ui.renderPanel?.setState) {
            const sr = { ...saved.render };
            if (!sr.motionMode || !sr.spinMode) {
                const legacy = sr.exportMode || "turntable";
                if (legacy === "clip") {
                    sr.motionMode = "clip";
                    sr.spinMode = "static";
                } else if (legacy === "single") {
                    sr.motionMode = "snap";
                    sr.spinMode = "single";
                } else {
                    sr.motionMode = "rest";
                    sr.spinMode = "turntable";
                }
                delete sr.exportMode;
            }
            this.ui.renderPanel.setState(
                {
                    ...sr,
                    clipStart: Math.max(1, Math.floor(Number(sr.clipStart) || 1)),
                    lockHeadPose: Boolean(sr.lockHeadPose),
                    showGate: sr.showGate !== false,
                },
                false,
            );
            this.renderState = this.ui.renderPanel.getState();
            this.renderFov = Number(this.renderState.fov || 36);
            this.ui.renderPanel.syncProjectionUI?.();
        }

        const vp = saved.viewport || {};
        this.enableReliabilityWeighting = Boolean(vp.enableReliabilityWeighting ?? true);
        this.enableAnatomicalWeighting = Boolean(vp.enableAnatomicalWeighting ?? true);
        this.enableRegularization = Boolean(vp.enableRegularization ?? true);
        this.mvTexEnabled = Boolean(vp.mvTexEnabled ?? true);
        this.uvDebugMode = "off";
        this.lastBakeResult = null;
        this.bakeToken += 1;
        this.reconstructedFace?.setMvTexEnabled(this.mvTexEnabled);
        this.syncWeightDebugButtons();

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
        this.reconstructedFace?.setIrisMaskScale(this.ui.weightPanel.getState().irisMaskScale ?? 1.08);
        this.reconstructedFace?.setSculptState(this.ui.sculptPanel.getState());
        this.reconstructedFace?.applyRenderState(this.renderState);

        const camPose = unpackCameraPose(saved.camera, THREE);
        if (camPose) {
            this.applyCameraPoseState(camPose);
        }

        this.snapshotPose = unpackSnapshotPose(saved.snapshot, THREE);
        this.snapshotActive = Boolean(saved.snapshotActive && this.snapshotPose);

        const refs = normalizePersistedRefs(saved.refs);
        this.pendingRestoredCamera = camPose || null;

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
            const settled = await Promise.allSettled(
                REF_VIEW_KEYS.map(async (key) => {
                    const url = urls[key];
                    if (!url) return null;
                    const cacheKey = getRefCacheKey(refs[key]);
                    const cached = await getCachedLandmarks(cacheKey);
                    if (cached) {
                        if (cached.segMaskPng === undefined) {
                            const segMaskPng = await segmentImageDataUrl(url);
                            cached.segMaskPng = segMaskPng || null;
                            if (cacheKey) {
                                setCachedLandmarks(cacheKey, cached);
                            }
                        }
                        return cached;
                    }
                    const [extracted, segMaskPng] = await Promise.all([
                        extractFaceFromDataUrl(url),
                        segmentImageDataUrl(url),
                    ]);
                    if (extracted) {
                        extracted.segMaskPng = segMaskPng || null;
                    }
                    if (extracted && cacheKey) {
                        setCachedLandmarks(cacheKey, extracted);
                    }
                    return extracted;
                })
            );
            if (this.disposed || resetToken !== this.resetToken) return;

            const nextViews = createRefSlotMap(null);
            REF_VIEW_KEYS.forEach((key, idx) => {
                const outcome = settled[idx];
                const hadUrl = Boolean(urls[key]);
                if (outcome.status === "fulfilled" && outcome.value) {
                    nextViews[key] = outcome.value;
                } else if (hadUrl) {
                    nextViews[key] = null;
                    this.refUrls[key] = null;
                    this.persistedRefs[key] = null;
                    this.ui.refStack.setPreview(key, null, false);
                    if (key === "front") {
                        console.warn(
                            "[YedpFace3D] Missing/unreadable front ref during state restore:",
                            outcome.status === "rejected" ? outcome.reason : "empty result"
                        );
                    } else {
                        console.warn(
                            `[YedpFace3D] Skipping missing/unreadable auxiliary ref: ${key}`,
                            outcome.status === "rejected" ? outcome.reason : ""
                        );
                    }
                }
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
                await this.loadDefaultCanonicalHead();
            }
            this.updateMotionUI();
            this.updateRenderUI();
            this.updateReadout();
        } catch (err) {
            if (this.disposed || resetToken !== this.resetToken) return;
            console.warn("[YedpFace3D] State restore failed; falling back to default canonical head:", err);
            await this.loadDefaultCanonicalHead();
            this.updateMotionUI();
            this.updateRenderUI();
            this.updateReadout();
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

    setFrameCount(frames) {
        this.frameCount = Math.max(0, Number(frames) ?? 24);
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
        this.renderFov = Number(this.renderState.fov || 36);
        this.orthoHalfH = dist * Math.tan(THREE.MathUtils.degToRad(this.renderFov * 0.5));
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
        this.applyResize(this.viewportWidth, this.viewportHeight);
        this.updateRenderUI();
    }

    computeResolutionGate(vw, vh, outW = this.outputWidth, outH = this.outputHeight, showGate = true) {
        const safeOutW = Math.max(1, Number(outW) || 512);
        const safeOutH = Math.max(1, Number(outH) || 512);
        const overscan = showGate ? 0.92 : 1.0;
        const fitScale = Math.min(vw / safeOutW, vh / safeOutH) * overscan;
        const gateW = safeOutW * fitScale;
        const gateH = safeOutH * fitScale;
        return {
            gateW,
            gateH,
            gateLeft: (vw - gateW) * 0.5,
            gateTop: (vh - gateH) * 0.5,
        };
    }

    updateResolutionGateOverlay(gate, showGate) {
        const rg = this.ui?.resGate;
        if (!rg) return;
        rg.container.style.display = showGate ? "block" : "none";
        if (!showGate) return;

        rg.box.style.left = `${gate.gateLeft.toFixed(2)}px`;
        rg.box.style.top = `${gate.gateTop.toFixed(2)}px`;
        rg.box.style.width = `${gate.gateW.toFixed(2)}px`;
        rg.box.style.height = `${gate.gateH.toFixed(2)}px`;

        const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
        const wInt = Math.round(this.outputWidth);
        const hInt = Math.round(this.outputHeight);
        const d = gcd(wInt, hInt) || 1;
        const rw = Math.round(wInt / d);
        const rh = Math.round(hInt / d);
        const ratioText = rw <= 32 && rh <= 32 ? `${rw}:${rh}` : (wInt / Math.max(hInt, 1)).toFixed(2);
        rg.label.textContent = `${wInt}×${hInt} (${ratioText})`;
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
        const viewAspect = w / h;
        const showGate = this.renderState?.showGate !== false;
        const gate = this.computeResolutionGate(w, h, this.outputWidth, this.outputHeight, showGate);
        const heightScale = h / Math.max(gate.gateH, 1e-4);

        // Pass updateStyle=false so canvas CSS stays 100% and avoids ResizeObserver loops.
        this.renderer.setSize(w, h, false);

        const halfOutFovRad = THREE.MathUtils.degToRad((this.renderFov || 36) * 0.5);
        this.perspCamera.fov = THREE.MathUtils.radToDeg(2.0 * Math.atan(Math.tan(halfOutFovRad) * heightScale));
        this.perspCamera.aspect = viewAspect;
        this.perspCamera.updateProjectionMatrix();

        const vpHalfH = this.orthoHalfH * heightScale;
        this.orthoCamera.left = -vpHalfH * viewAspect;
        this.orthoCamera.right = vpHalfH * viewAspect;
        this.orthoCamera.top = vpHalfH;
        this.orthoCamera.bottom = -vpHalfH;
        this.orthoCamera.updateProjectionMatrix();

        this.updateResolutionGateOverlay(gate, showGate);
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
            const fps = this.clipFps || 24;
            const plan = this.getExportPlan();
            const total = Math.max(1, plan.frames > 1 ? plan.frames : (this.frameCount > 0 ? this.frameCount : 24));
            this.turntableCursor = (this.turntableCursor + dt * fps) % total;
            const t = total > 1 ? this.turntableCursor / (total - 1) : 0.5;
            const startDeg = Number(this.renderState.startAngle ?? -90);
            const endDeg = Number(this.renderState.endAngle ?? 90);
            const yawDeg = startDeg + (endDeg - startDeg) * t;
            const yawRad = THREE.MathUtils.degToRad(yawDeg);
            const frameIdx = Math.floor(this.turntableCursor);

            if (this.reconstructedFace.hasMesh) {
                if (plan.motionMode === "clip" && this.recordedClip.length > 0) {
                    const mf = this.recordedClip[Math.min(this.recordedClip.length - 1, plan.startIdx + frameIdx)];
                    this.reconstructedFace.applyMotionFrame(mf);
                    this.reconstructedFace.setTurntableYaw(yawRad, this.renderState.lockHeadPose ? null : mf?.quat);
                } else if (plan.motionMode === "snap" && this.snapshotPose?.motionFrame) {
                    this.reconstructedFace.applyMotionFrame(this.snapshotPose.motionFrame);
                    this.reconstructedFace.setTurntableYaw(
                        yawRad,
                        this.renderState.lockHeadPose ? null : this.snapshotPose.motionFrame.quat
                    );
                } else {
                    this.reconstructedFace.setTurntableYaw(yawRad);
                }
            }
            this.ui.renderPanel.renderStatus.textContent = `Turntable F${frameIdx + 1}/${total} (${Math.round(yawDeg)}°)`;
            this.ui.renderPanel.renderStatus.style.color = "#38bdf8";
            return;
        }

        if (this.clipPlaying && this.recordedClip.length > 0 && this.reconstructedFace.hasMesh) {
            const fps = this.clipFps || 24;
            this.clipCursor = (this.clipCursor + dt * fps) % this.recordedClip.length;
            const idx = Math.floor(this.clipCursor);
            this.reconstructedFace.applyMotionFrame(this.recordedClip[idx]);
            const mp = this.ui.mocapPanel;
            mp.clipScrubber.value = String(idx);
            mp.clipFrameLabel.textContent = `${idx + 1}/${this.recordedClip.length}`;
            mp.motionStatus.textContent = `F${idx + 1}/${this.recordedClip.length} @ ${fps}fps`;
            mp.motionStatus.style.color = "#38bdf8";
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
                            frame._rawDet = det;
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
        this.bakeToken++;
        this.videoLoadToken++;
        this.lastBakeResult = null;
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
