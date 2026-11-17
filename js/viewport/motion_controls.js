import { stopViewportEvents } from "./ui_icons.js";

function makeBarButton(label, title = "") {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    if (title) btn.title = title;
    Object.assign(btn.style, {
        fontSize: "10px",
        fontWeight: "500",
        color: "#f8fafc",
        backgroundColor: "rgba(25, 30, 42, 0.88)",
        border: "1px solid rgba(255, 255, 255, 0.16)",
        borderRadius: "5px",
        padding: "3px 8px",
        cursor: "pointer",
    });
    stopViewportEvents(btn);
    return btn;
}

/**
 * Bottom-center shading mode switcher (`Textured | Wire | Normals | Depth | Smooth`).
 */
export function createMotionBar() {
    const bar = document.createElement("div");
    Object.assign(bar.style, {
        display: "inline-flex",
        alignItems: "center",
        gap: "3px",
        padding: "3px",
        borderRadius: "7px",
        backgroundColor: "rgba(15, 18, 25, 0.78)",
        border: "1px solid rgba(255, 255, 255, 0.14)",
        backdropFilter: "blur(5px)",
        pointerEvents: "auto",
    });
    stopViewportEvents(bar);

    const modes = [
        { id: "textured", label: "Textured" },
        { id: "wireframe", label: "Wire" },
        { id: "normals", label: "Normals" },
        { id: "depth", label: "Depth" },
    ];

    const modeButtons = {};
    for (const m of modes) {
        const btn = makeBarButton(m.label);
        modeButtons[m.id] = btn;
        bar.appendChild(btn);
    }

    const smoothBtn = makeBarButton("Smooth", "Toggle 1-level Phong curved subdivision (852 -> 3,408 tris)");
    smoothBtn.style.borderColor = "#38bdf8";
    smoothBtn.style.color = "#38bdf8";
    bar.appendChild(smoothBtn);

    const setActiveMode = (activeId) => {
        for (const [id, btn] of Object.entries(modeButtons)) {
            btn.style.borderColor = id === activeId ? "#38bdf8" : "rgba(255, 255, 255, 0.16)";
            btn.style.color = id === activeId ? "#38bdf8" : "#f8fafc";
        }
    };
    setActiveMode("textured");

    return {
        bar,
        modeButtons,
        smoothBtn,
        setActiveMode,
    };
}
