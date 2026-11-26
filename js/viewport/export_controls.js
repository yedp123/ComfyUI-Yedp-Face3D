import { stopViewportEvents } from "./ui_icons.js";

export const DEFAULT_EXPORT_STATE = {
    scope: "head",
    subdiv: "smooth",
    includeEyeballs: true,
    includeTeeth: false,
    pose: "rest",
    animMode: "arkit",
};

function makeExportButton(label, title = "") {
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
        textAlign: "center",
    });
    stopViewportEvents(btn);
    return btn;
}

/**
 * Collapsible EXPORT category body: Static Mesh (.GLB + UV .PNG) and
 * Rig/Animation (.GLB with 52 ARKit blendshapes or Mocap Animation).
 */
export function createExportPanel(handlers = {}) {
    const state = { ...DEFAULT_EXPORT_STATE };

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

    const addSection = (title) => {
        const h = document.createElement("div");
        h.textContent = title;
        Object.assign(h.style, {
            fontWeight: "600",
            color: "#94a3b8",
            borderBottom: "1px solid rgba(255,255,255,0.08)",
            paddingBottom: "2px",
            marginTop: "2px",
        });
        panel.appendChild(h);
    };

    const addSegmented = (label, key, options) => {
        const row = document.createElement("div");
        Object.assign(row.style, { display: "flex", alignItems: "center", justifyContent: "space-between", gap: "4px" });
        const grp = document.createElement("div");
        Object.assign(grp.style, { display: "flex", gap: "3px", flex: label ? "0 0 auto" : "1" });

        const btns = {};
        const sync = () => {
            for (const opt of options) {
                const active = state[key] === opt.value;
                btns[opt.value].style.borderColor = active ? "#38bdf8" : "rgba(255,255,255,0.16)";
                btns[opt.value].style.color = active ? "#38bdf8" : "#cbd5e1";
            }
        };

        for (const opt of options) {
            const b = makeExportButton(opt.label, opt.title || "");
            b.style.width = "auto";
            b.style.padding = "2px 7px";
            if (!label) b.style.flex = "1";
            b.addEventListener("click", (e) => {
                e.stopPropagation();
                state[key] = opt.value;
                sync();
            });
            btns[opt.value] = b;
            grp.appendChild(b);
        }
        sync();
        if (label) {
            const lbl = document.createElement("span");
            lbl.textContent = label;
            row.append(lbl, grp);
        } else {
            row.appendChild(grp);
        }
        panel.appendChild(row);
    };

    // --- Mesh Options ---
    addSection("Mesh Options");
    addSegmented("Scope", "scope", [
        { label: "Face + Head", value: "head", title: "Export welded face + cranium + ears + neck" },
        { label: "Face Only", value: "face", title: "Export 478-vert face mesh only (zero skull/neck vertices)" },
    ]);
    addSegmented("Subdiv", "subdiv", [
        { label: "Smooth", value: "smooth", title: "1-level Phong curved subdivision" },
        { label: "Base", value: "base", title: "Lightweight base topology (recommended for rigs)" },
    ]);

    const eyeRow = document.createElement("label");
    Object.assign(eyeRow.style, { display: "inline-flex", alignItems: "center", gap: "5px", cursor: "pointer" });
    const eyeCb = document.createElement("input");
    eyeCb.type = "checkbox";
    eyeCb.checked = state.includeEyeballs;
    stopViewportEvents(eyeCb);
    eyeCb.addEventListener("change", () => {
        state.includeEyeballs = eyeCb.checked;
    });
    const eyeLbl = document.createElement("span");
    eyeLbl.textContent = "Include Eyeballs";
    eyeRow.append(eyeCb, eyeLbl);
    panel.appendChild(eyeRow);

    const teethRow = document.createElement("label");
    Object.assign(teethRow.style, { display: "inline-flex", alignItems: "center", gap: "5px", cursor: "pointer" });
    const teethCb = document.createElement("input");
    teethCb.type = "checkbox";
    teethCb.checked = state.includeTeeth;
    stopViewportEvents(teethCb);
    teethCb.addEventListener("change", () => {
        state.includeTeeth = teethCb.checked;
    });
    const teethLbl = document.createElement("span");
    teethLbl.textContent = "Include 3D Teeth";
    teethRow.append(teethCb, teethLbl);
    panel.appendChild(teethRow);

    // --- 1. Static Mesh ---
    addSection("1. Static Mesh");
    addSegmented("Pose", "pose", [
        { label: "Rest", value: "rest", title: "Export sculpted neutral rest pose" },
        { label: "Current", value: "current", title: "Export current deformed expression/pose" },
    ]);

    const exportStaticBtn = makeExportButton(
        "Export Static Mesh (.GLB)",
        "Download static 3D mesh with embedded UV texture"
    );
    exportStaticBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        handlers.onExportStatic?.({ ...state });
    });

    const exportTextureBtn = makeExportButton(
        "Export UV Texture (.PNG)",
        "Download 1024x1024 UV texture map as PNG"
    );
    exportTextureBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        handlers.onExportTexture?.({ ...state });
    });

    panel.append(exportStaticBtn, exportTextureBtn);

    // --- 2. Rig & Mocap ---
    addSection("2. Rig & Mocap");
    const exportBlendshapesBtn = makeExportButton(
        "Export Mesh + Blendshapes (.GLB)",
        "Download mesh with 52 named ARKit blendshape morph targets"
    );
    exportBlendshapesBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        handlers.onExportBlendshapes?.({ ...state });
    });

    addSegmented("Anim", "animMode", [
        { label: "ARKit 52", value: "arkit", title: "Export 52 ARKit blendshapes + animated weight/head/eye curves" },
        { label: "Exact Bake", value: "exact", title: "Export vertex-identical per-frame retarget morphs" },
    ]);

    const exportAnimBtn = makeExportButton(
        "Export Mesh + Animation (.GLB)",
        "Record webcam or load a video in MOCAP to enable animation export"
    );
    exportAnimBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        handlers.onExportAnimation?.({ ...state });
    });

    const exportStatus = document.createElement("div");
    exportStatus.textContent = "Idle";
    Object.assign(exportStatus.style, {
        fontSize: "9px",
        fontFamily: "ui-monospace, monospace",
        color: "#94a3b8",
        minHeight: "12px",
    });

    panel.append(exportBlendshapesBtn, exportAnimBtn, exportStatus);

    const setAnimEnabled = (enabled, clipFrames = 0) => {
        exportAnimBtn.disabled = !enabled;
        exportAnimBtn.style.opacity = enabled ? "1.0" : "0.45";
        exportAnimBtn.style.cursor = enabled ? "pointer" : "default";
        exportAnimBtn.title = enabled
            ? `Export ${clipFrames}-frame mocap animation as .GLB`
            : "Record webcam or load a video in MOCAP to enable animation export";
    };
    setAnimEnabled(false, 0);

    const setStatus = (text, isError = false) => {
        exportStatus.textContent = text;
        exportStatus.style.color = isError ? "#fca5a5" : "#94a3b8";
    };

    return {
        panel,
        getState: () => ({ ...state }),
        exportStaticBtn,
        exportTextureBtn,
        exportBlendshapesBtn,
        exportAnimBtn,
        exportStatus,
        setAnimEnabled,
        setStatus,
    };
}
