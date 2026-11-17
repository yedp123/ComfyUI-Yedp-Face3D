import { stopViewportEvents } from "./ui_icons.js";

function makePanelButton(label, title = "") {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    if (title) btn.title = title;
    Object.assign(btn.style, {
        fontSize: "10px",
        fontWeight: "500",
        color: "#f8fafc",
        backgroundColor: "rgba(30, 41, 59, 0.9)",
        border: "1px solid rgba(255, 255, 255, 0.16)",
        borderRadius: "5px",
        padding: "4px 7px",
        cursor: "pointer",
        flex: "1",
        minWidth: "0",
    });
    stopViewportEvents(btn);
    return btn;
}

/**
 * Collapsible MOCAP category body hosting the live webcam canvas preview,
 * tracking/calibration buttons, clip recording, and playback controls.
 */
export function createMocapPanel(webcamOverlay) {
    const panel = document.createElement("div");
    Object.assign(panel.style, {
        width: "100%",
        boxSizing: "border-box",
        padding: "8px 10px 10px 10px",
        display: "flex",
        flexDirection: "column",
        gap: "6px",
        color: "#e2e8f0",
        fontSize: "10px",
    });
    stopViewportEvents(panel);

    const webcamRow = document.createElement("div");
    Object.assign(webcamRow.style, { display: "flex", gap: "4px", alignItems: "center" });

    const webcamBtn = makePanelButton("Webcam", "Toggle live CPU webcam face tracking");
    const recordBtn = makePanelButton("Rec", "Record webcam expression + head pose clip");
    recordBtn.style.display = "none";
    const calibrateBtn = makePanelButton("Calibrate", "Set current webcam expression as neutral");
    calibrateBtn.style.display = "none";

    webcamRow.append(webcamBtn, recordBtn, calibrateBtn);

    const playbackRow = document.createElement("div");
    Object.assign(playbackRow.style, { display: "flex", gap: "4px", alignItems: "center" });

    const playClipBtn = makePanelButton("Play Clip", "Play recorded webcam or driving video clip");
    playClipBtn.style.display = "none";
    const clearClipBtn = makePanelButton("×", "Clear recorded motion clip");
    clearClipBtn.style.flex = "0 0 auto";
    clearClipBtn.style.padding = "4px 8px";
    clearClipBtn.style.color = "#fca5a5";
    clearClipBtn.style.display = "none";

    playbackRow.append(playClipBtn, clearClipBtn);

    const motionStatus = document.createElement("div");
    Object.assign(motionStatus.style, {
        fontSize: "10px",
        fontFamily: "ui-monospace, monospace",
        color: "#94a3b8",
        minHeight: "12px",
    });

    panel.append(webcamOverlay.element, webcamRow, playbackRow, motionStatus);

    return {
        panel,
        webcamBtn,
        recordBtn,
        calibrateBtn,
        playClipBtn,
        clearClipBtn,
        motionStatus,
    };
}
