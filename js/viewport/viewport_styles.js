import { createLightPanel } from "./light_controls.js";
import { createWeightPanel } from "./weight_controls.js";
import { createSculptPanel } from "./sculpt_controls.js";
import { createMotionBar } from "./motion_controls.js";
import { createWebcamOverlay } from "./webcam_overlay.js";
import { createRefStackPanel } from "./ref_stack_panel.js";
import { createMocapPanel } from "./mocap_controls.js";
import { createRenderPanel } from "./render_controls.js";
import { createExportPanel } from "./export_controls.js";
import { createRightSidebar } from "./right_sidebar.js";
import { stopViewportEvents } from "./ui_icons.js";

/**
 * Builds the parent DOM container (`position: relative`) required by ComfyUI Nodes 2.0:
 * - Left-center Three.js stage with floating reference stack on left, Reset View + orbit hint
 *   at top center, and shading mode bar + dimensions at bottom center
 * - Right docked vertical accordion panel (Sculpt, Weights, Lights, Mocap, Render)
 */
export function createViewportDOM(handlers) {
    const {
        onLightsChange,
        onWeightsChange,
        onSculptChange,
        onRenderChange,
        onFitHead,
        onSelectRefFile,
        onResetAll,
        onStatusError,
        onExportStatic,
        onExportTexture,
        onExportBlendshapes,
        onExportAnimation,
    } = handlers;

    const container = document.createElement("div");
    container.setAttribute?.("data-capture-wheel", "true");
    container.tabIndex = -1;
    Object.assign(container.style, {
        position: "relative",
        width: "100%",
        height: "100%",
        minHeight: "380px",
        overflow: "hidden",
        borderRadius: "8px",
        border: "1px solid rgba(255, 255, 255, 0.12)",
        background: "radial-gradient(circle at 40% 42%, #242936 0%, #12151c 100%)",
        userSelect: "none",
        boxSizing: "border-box",
        outline: "none",
        fontFamily: "ui-sans-serif, system-ui, -apple-system, sans-serif",
    });
    container.style.setProperty?.("--comfy-widget-min-height", "380");

    // Nodes 2.0 onWheelCapture only forwards wheel events to the widget when it is focused.
    const ensureViewportFocus = () => {
        const active = typeof document !== "undefined" ? document.activeElement : null;
        if (active && container.contains?.(active)) return;
        container.focus?.({ preventScroll: true });
    };
    container.addEventListener("pointerenter", ensureViewportFocus);
    container.addEventListener("pointerdown", ensureViewportFocus, true);

    // Prevent wheel zoom inside the 3D viewport from zooming the ComfyUI node graph.
    container.addEventListener("wheel", (e) => e.stopPropagation());

    // Open 3D stage (canvas + left ref stack + top-center controls + bottom-center modes)
    const stage = document.createElement("div");
    Object.assign(stage.style, {
        position: "relative",
        flex: "1",
        minWidth: "0",
        height: "100%",
        overflow: "hidden",
    });

    const canvas = document.createElement("canvas");
    Object.assign(canvas.style, {
        position: "absolute",
        inset: "0",
        width: "100%",
        height: "100%",
        display: "block",
    });
    stage.appendChild(canvas);

    // Maya-style Resolution Gate overlay (between WebGL canvas and HUD controls)
    const resGateContainer = document.createElement("div");
    Object.assign(resGateContainer.style, {
        position: "absolute",
        inset: "0",
        pointerEvents: "none",
        overflow: "hidden",
        zIndex: "5",
    });

    const resGateBox = document.createElement("div");
    Object.assign(resGateBox.style, {
        position: "absolute",
        left: "0px",
        top: "0px",
        width: "0px",
        height: "0px",
        boxSizing: "border-box",
        border: "1px solid rgba(226, 232, 240, 0.42)",
        boxShadow: "0 0 0 9999px rgba(8, 10, 15, 0.52)",
        pointerEvents: "none",
    });

    const bracketSpecs = [
        { top: "-1px", left: "-1px", borderTop: "2px solid #38bdf8", borderLeft: "2px solid #38bdf8" },
        { top: "-1px", right: "-1px", borderTop: "2px solid #38bdf8", borderRight: "2px solid #38bdf8" },
        { bottom: "-1px", left: "-1px", borderBottom: "2px solid #38bdf8", borderLeft: "2px solid #38bdf8" },
        { bottom: "-1px", right: "-1px", borderBottom: "2px solid #38bdf8", borderRight: "2px solid #38bdf8" },
    ];
    for (const spec of bracketSpecs) {
        const corner = document.createElement("div");
        Object.assign(corner.style, {
            position: "absolute",
            width: "10px",
            height: "10px",
            pointerEvents: "none",
            ...spec,
        });
        resGateBox.appendChild(corner);
    }

    const resGateLabel = document.createElement("span");
    resGateLabel.textContent = "512×512 (1:1)";
    Object.assign(resGateLabel.style, {
        position: "absolute",
        bottom: "4px",
        right: "6px",
        fontSize: "9px",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
        fontWeight: "600",
        color: "rgba(226, 232, 240, 0.85)",
        backgroundColor: "rgba(15, 18, 25, 0.72)",
        border: "1px solid rgba(255, 255, 255, 0.14)",
        borderRadius: "4px",
        padding: "1px 5px",
        whiteSpace: "nowrap",
        pointerEvents: "none",
    });
    resGateBox.appendChild(resGateLabel);
    resGateContainer.appendChild(resGateBox);
    stage.appendChild(resGateContainer);

    const hudLayer = document.createElement("div");
    Object.assign(hudLayer.style, {
        position: "absolute",
        inset: "0",
        pointerEvents: "none",
        padding: "10px",
        boxSizing: "border-box",
    });
    stage.appendChild(hudLayer);

    // 1. Left reference stack (front, three-quarter, side, RESET)
    const refStack = createRefStackPanel({
        onSelectFile: onSelectRefFile,
        onResetAll,
        onError: onStatusError,
    });
    Object.assign(refStack.element.style, {
        position: "absolute",
        top: "10px",
        left: "10px",
        zIndex: "10",
    });
    hudLayer.appendChild(refStack.element);

    // 2. Top-center Reset View + navigation hint + compact status pill
    const topCenter = document.createElement("div");
    Object.assign(topCenter.style, {
        position: "absolute",
        top: "10px",
        left: "50%",
        transform: "translateX(-50%)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "4px",
        zIndex: "10",
    });

    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.textContent = "Reset View";
    Object.assign(resetBtn.style, {
        fontSize: "11px",
        fontWeight: "600",
        color: "#f8fafc",
        backgroundColor: "rgba(25, 30, 42, 0.88)",
        border: "1px solid rgba(255, 255, 255, 0.22)",
        borderRadius: "6px",
        padding: "4px 12px",
        cursor: "pointer",
        pointerEvents: "auto",
        backdropFilter: "blur(4px)",
    });
    stopViewportEvents(resetBtn);

    const hintReadout = document.createElement("span");
    hintReadout.textContent = "LMB Orbit · RMB Pan · Wheel Zoom";
    Object.assign(hintReadout.style, {
        fontSize: "10px",
        color: "#94a3b8",
        backgroundColor: "rgba(15, 18, 25, 0.65)",
        borderRadius: "5px",
        padding: "2px 8px",
        whiteSpace: "nowrap",
    });

    const badge = document.createElement("span");
    badge.textContent = "Yedp Face3D · Placeholder";
    Object.assign(badge.style, {
        fontSize: "10px",
        fontWeight: "500",
        color: "#cbd5e1",
        backgroundColor: "rgba(15, 18, 25, 0.72)",
        border: "1px solid rgba(255, 255, 255, 0.12)",
        borderRadius: "5px",
        padding: "2px 8px",
        whiteSpace: "nowrap",
    });

    topCenter.append(resetBtn, hintReadout, badge);
    hudLayer.appendChild(topCenter);

    // 3. Bottom-center mode bar + viewport/output size readout
    const bottomCenter = document.createElement("div");
    Object.assign(bottomCenter.style, {
        position: "absolute",
        bottom: "10px",
        left: "50%",
        transform: "translateX(-50%)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "4px",
        zIndex: "10",
    });

    const modeBar = createMotionBar();
    const sizeReadout = document.createElement("span");
    sizeReadout.textContent = "Viewport: --x-- | Output: 512x512";
    Object.assign(sizeReadout.style, {
        fontSize: "10px",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
        color: "#94a3b8",
        backgroundColor: "rgba(15, 18, 25, 0.72)",
        border: "1px solid rgba(255, 255, 255, 0.1)",
        borderRadius: "5px",
        padding: "2px 8px",
        whiteSpace: "nowrap",
    });

    bottomCenter.append(modeBar.bar, sizeReadout);
    hudLayer.appendChild(bottomCenter);

    // 4. Right vertical accordion sidebar
    const webcamVideo = document.createElement("video");
    webcamVideo.setAttribute("playsinline", "");
    webcamVideo.muted = true;
    webcamVideo.autoplay = true;
    webcamVideo.style.display = "none";

    const sculptPanel = createSculptPanel(onSculptChange);
    const weightPanel = createWeightPanel(onWeightsChange);
    const lightPanel = createLightPanel(onLightsChange);
    const webcamOverlay = createWebcamOverlay();
    const mocapPanel = createMocapPanel(webcamOverlay);
    const renderPanel = createRenderPanel(onRenderChange, onFitHead);
    const exportPanel = createExportPanel({
        onExportStatic,
        onExportTexture,
        onExportBlendshapes,
        onExportAnimation,
    });

    const rightSidebar = createRightSidebar([
        { id: "sculpt", title: "SCULPT", icon: "sculpt", body: sculptPanel.panel },
        { id: "weights", title: "WEIGHTS", icon: "weights", body: weightPanel.panel },
        { id: "lights", title: "LIGHTS", icon: "lights", body: lightPanel.panel },
        { id: "mocap", title: "MOCAP", icon: "mocap", body: mocapPanel.panel },
        { id: "render", title: "RENDER", icon: "render", body: renderPanel.panel },
        { id: "export", title: "EXPORT", icon: "export", body: exportPanel.panel },
    ]);

    const viewportBody = document.createElement("div");
    Object.assign(viewportBody.style, {
        position: "absolute",
        inset: "0",
        display: "flex",
        flexDirection: "row",
        overflow: "hidden",
    });
    viewportBody.append(stage, rightSidebar.element);
    container.append(viewportBody, webcamVideo);

    return {
        container,
        stage,
        canvas,
        resGate: {
            container: resGateContainer,
            box: resGateBox,
            label: resGateLabel,
        },
        hudLayer,
        refStack,
        badge,
        resetBtn,
        modeBar,
        motionBar: modeBar, // alias preserved for internal consumers
        sizeReadout,
        rightSidebar,
        sculptPanel,
        weightPanel,
        lightPanel,
        webcamVideo,
        webcamOverlay,
        mocapPanel,
        renderPanel,
        exportPanel,
    };
}
