import { createLightPanel } from "./light_controls.js";
import { createWeightPanel } from "./weight_controls.js";
import { createSculptPanel } from "./sculpt_controls.js";
import { createMotionBar } from "./motion_controls.js";
import { createWebcamOverlay } from "./webcam_overlay.js";
import { createRefStackPanel } from "./ref_stack_panel.js";
import { createMocapPanel } from "./mocap_controls.js";
import { createRenderPanel } from "./render_controls.js";
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
        onRemoveRefView,
        onResetAll,
        onStatusError,
    } = handlers;

    const container = document.createElement("div");
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
        display: "flex",
        flexDirection: "row",
        fontFamily: "ui-sans-serif, system-ui, -apple-system, sans-serif",
    });
    container.style.setProperty("--comfy-widget-min-height", "380");

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
        onRemoveView: onRemoveRefView,
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

    const rightSidebar = createRightSidebar([
        { id: "sculpt", title: "SCULPT", icon: "sculpt", body: sculptPanel.panel },
        { id: "weights", title: "WEIGHTS", icon: "weights", body: weightPanel.panel },
        { id: "lights", title: "LIGHTS", icon: "lights", body: lightPanel.panel },
        { id: "mocap", title: "MOCAP", icon: "mocap", body: mocapPanel.panel },
        { id: "render", title: "RENDER", icon: "render", body: renderPanel.panel },
    ]);

    container.append(stage, rightSidebar.element, webcamVideo);

    return {
        container,
        stage,
        canvas,
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
    };
}
