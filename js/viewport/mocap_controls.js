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
    const calibrateBtn = makePanelButton("Calibrate", "Set current webcam expression as neutral");
    calibrateBtn.style.display = "none";

    webcamRow.append(webcamBtn, calibrateBtn);

    const captureRow = document.createElement("div");
    Object.assign(captureRow.style, { display: "none", gap: "4px", alignItems: "center" });

    const recordBtn = makePanelButton("Rec", "Record multi-frame webcam expression + head pose clip");
    const snapBtn = makePanelButton("Snap", "Capture current single expression + angle snapshot and stop webcam");
    const snapTimedBtn = makePanelButton("3s", "3-second delayed expression + angle snapshot");
    snapTimedBtn.style.flex = "0 0 auto";
    snapTimedBtn.style.padding = "4px 7px";

    captureRow.append(recordBtn, snapBtn, snapTimedBtn);

    const snapshotRow = document.createElement("div");
    Object.assign(snapshotRow.style, { display: "none", gap: "4px", alignItems: "center" });

    const snapToggleBtn = makePanelButton("Snap: ON", "Toggle between saved snapshot expression and neutral rest pose");
    const snapAngleBtn = makePanelButton("Snap Angle", "Restore camera angle captured at snapshot time");
    snapAngleBtn.style.flex = "0 0 auto";
    const clearSnapBtn = makePanelButton("×", "Clear snapshot from memory and reset pose");
    clearSnapBtn.style.flex = "0 0 auto";
    clearSnapBtn.style.padding = "4px 8px";
    clearSnapBtn.style.color = "#fca5a5";

    snapshotRow.append(snapToggleBtn, snapAngleBtn, clearSnapBtn);

    const playbackRow = document.createElement("div");
    Object.assign(playbackRow.style, { display: "none", gap: "4px", alignItems: "center" });

    const playClipBtn = makePanelButton("Play Clip", "Play recorded webcam or driving video clip");
    const clearClipBtn = makePanelButton("×", "Clear recorded motion clip");
    clearClipBtn.style.flex = "0 0 auto";
    clearClipBtn.style.padding = "4px 8px";
    clearClipBtn.style.color = "#fca5a5";

    playbackRow.append(playClipBtn, clearClipBtn);

    const motionStatus = document.createElement("div");
    Object.assign(motionStatus.style, {
        fontSize: "10px",
        fontFamily: "ui-monospace, monospace",
        color: "#94a3b8",
        minHeight: "12px",
    });

    panel.append(webcamOverlay.element, webcamRow, captureRow, snapshotRow, playbackRow, motionStatus);

    return {
        panel,
        webcamBtn,
        calibrateBtn,
        captureRow,
        recordBtn,
        snapBtn,
        snapTimedBtn,
        snapshotRow,
        snapToggleBtn,
        snapAngleBtn,
        clearSnapBtn,
        playbackRow,
        playClipBtn,
        clearClipBtn,
        motionStatus,
    };
}
