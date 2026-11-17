import { stopViewportEvents } from "./ui_icons.js";

function makeRenderButton(label, title = "") {
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
        padding: "5px 8px",
        cursor: "pointer",
        width: "100%",
    });
    stopViewportEvents(btn);
    return btn;
}

/**
 * Collapsible RENDER category body holding the 180-deg Turntable button,
 * guide visibility toggle, and active output spec readout.
 */
export function createRenderPanel() {
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

    const turntableBtn = makeRenderButton("Turntable (180°)", "Play/pause 180-degree turntable sweep (-90 to +90 deg)");
    const guidesBtn = makeRenderButton("Guides: ON", "Toggle ear and neck guide meshes");

    const renderStatus = document.createElement("div");
    renderStatus.textContent = "Idle · Ready to queue";
    Object.assign(renderStatus.style, {
        fontSize: "10px",
        fontFamily: "ui-monospace, monospace",
        color: "#94a3b8",
    });

    const outputSpec = document.createElement("div");
    outputSpec.textContent = "Output: 512x512 · 24f";
    Object.assign(outputSpec.style, {
        fontSize: "9px",
        fontFamily: "ui-monospace, monospace",
        color: "#64748b",
        borderTop: "1px solid rgba(255,255,255,0.08)",
        paddingTop: "5px",
    });

    panel.append(turntableBtn, guidesBtn, renderStatus, outputSpec);

    const setGuidesActive = (visible) => {
        guidesBtn.textContent = `Guides: ${visible ? "ON" : "OFF"}`;
        guidesBtn.style.borderColor = visible ? "#38bdf8" : "rgba(255, 255, 255, 0.16)";
        guidesBtn.style.color = visible ? "#38bdf8" : "#cbd5e1";
    };
    setGuidesActive(true);

    const setOutputSpec = (width, height, frames) => {
        outputSpec.textContent = `Output: ${width}x${height} · ${frames}f`;
    };

    return {
        panel,
        turntableBtn,
        guidesBtn,
        renderStatus,
        setGuidesActive,
        setOutputSpec,
    };
}
