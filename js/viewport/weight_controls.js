import { stopViewportEvents } from "./ui_icons.js";

export const DEFAULT_WEIGHT_STATE = {
    eyes: 1.15,
    brows: 1.0,
    mouth: 1.0,
    mouthWidth: 0.55,
    jaw: 0.85,
    cheeks: 0.45,
    baseAnchor: 0.25,
    dispSmooth: 0.45,
    maxDisp: 0.18,
    symmetry: 0.0,
    enableSilhouetteFit: true,
    silhouetteStrength: 0.75,
    enableColorMatch: true,
    enableSeamBlend: true,
    irisMaskScale: 1.08,
};

/**
 * Collapsible category body for Laplacian-blended regional retargeting weights.
 */
export function createWeightPanel(onChange) {
    const state = { ...DEFAULT_WEIGHT_STATE };

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

    const setters = [];

    const addSlider = (label, key, min, max, step) => {
        const row = document.createElement("div");
        Object.assign(row.style, { display: "flex", alignItems: "center", gap: "6px" });

        const lbl = document.createElement("span");
        lbl.textContent = label;
        lbl.style.width = "64px";

        const input = document.createElement("input");
        input.type = "range";
        input.min = String(min);
        input.max = String(max);
        input.step = String(step);
        input.value = String(state[key]);
        input.style.flex = "1";
        input.style.minWidth = "0";
        stopViewportEvents(input);

        const valEl = document.createElement("span");
        valEl.style.width = "34px";
        valEl.style.textAlign = "right";
        valEl.style.fontFamily = "ui-monospace, monospace";
        valEl.textContent = state[key].toFixed(2);

        input.addEventListener("input", () => {
            state[key] = Number(input.value);
            valEl.textContent = state[key].toFixed(2);
            onChange({ ...state });
        });

        setters.push(() => {
            const raw = state[key];
            const num = (raw !== null && raw !== undefined && raw !== "" && Number.isFinite(Number(raw)))
                ? Number(raw)
                : DEFAULT_WEIGHT_STATE[key];
            state[key] = num;
            input.value = String(num);
            valEl.textContent = num.toFixed(2);
        });

        row.append(lbl, input, valEl);
        panel.appendChild(row);
    };

    addSlider("Eyes", "eyes", 0, 2, 0.05);
    addSlider("Brows", "brows", 0, 2, 0.05);
    addSlider("Mouth", "mouth", 0, 2, 0.05);
    addSlider("Mouth Width", "mouthWidth", 0, 1.5, 0.05);
    addSlider("Jaw", "jaw", 0, 2, 0.05);
    addSlider("Cheeks", "cheeks", 0, 1.5, 0.05);

    const regHeader = document.createElement("div");
    regHeader.textContent = "Geometry Regularization";
    Object.assign(regHeader.style, {
        fontWeight: "600",
        color: "#94a3b8",
        borderBottom: "1px solid rgba(255,255,255,0.08)",
        paddingBottom: "2px",
        marginTop: "4px",
    });
    panel.appendChild(regHeader);

    addSlider("Base Anchor", "baseAnchor", 0, 1, 0.05);
    addSlider("Disp. Smooth", "dispSmooth", 0, 1, 0.05);
    addSlider("Max Disp.", "maxDisp", 0.05, 0.50, 0.01);
    addSlider("Symmetry", "symmetry", 0, 1, 0.05);

    const addToggle = (label, key, withDivider = false) => {
        const row = document.createElement("div");
        Object.assign(row.style, {
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            ...(withDivider
                ? { marginTop: "2px", paddingTop: "4px", borderTop: "1px solid rgba(71, 85, 105, 0.45)" }
                : {}),
        });
        const lbl = document.createElement("span");
        lbl.textContent = label;
        Object.assign(lbl.style, {
            fontSize: "10px",
            color: "#cbd5e1",
            fontWeight: "500",
        });
        const btn = document.createElement("button");
        btn.type = "button";
        btn.setAttribute("data-weight-toggle", key);
        Object.assign(btn.style, {
            fontSize: "9px",
            fontWeight: "600",
            padding: "2px 8px",
            borderRadius: "4px",
            cursor: "pointer",
            lineHeight: "1.2",
        });
        const sync = () => {
            const enabled = state[key] !== false;
            state[key] = enabled;
            btn.textContent = enabled ? "ON" : "OFF";
            btn.style.background = enabled ? "rgba(16, 185, 129, 0.22)" : "rgba(51, 65, 85, 0.5)";
            btn.style.border = enabled ? "1px solid rgba(52, 211, 153, 0.55)" : "1px solid rgba(71, 85, 105, 0.65)";
            btn.style.color = enabled ? "#6ee7b7" : "#94a3b8";
        };
        stopViewportEvents(btn);
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            state[key] = state[key] === false;
            sync();
            onChange({ ...state });
        });
        sync();
        setters.push(sync);
        row.appendChild(lbl);
        row.appendChild(btn);
        panel.appendChild(row);
    };

    addToggle("Silh. Fit", "enableSilhouetteFit", true);
    addSlider("Silh. Strength", "silhouetteStrength", 0, 1, 0.05);
    addToggle("Color Match", "enableColorMatch", true);
    addToggle("Seam Blend", "enableSeamBlend", false);
    addSlider("Iris Mask", "irisMaskScale", 0.75, 1.60, 0.02);

    const note = document.createElement("div");
    note.textContent = "Intermediate vertices blend smoothly across mesh edges.";
    Object.assign(note.style, {
        fontSize: "9px",
        color: "#64748b",
        lineHeight: "1.25",
    });
    panel.appendChild(note);

    const setState = (partial, notify = false) => {
        if (!partial) return;
        Object.assign(state, partial);
        for (const s of setters) s();
        if (notify) onChange({ ...state });
    };

    const reset = (notify = true) => {
        setState(DEFAULT_WEIGHT_STATE, notify);
    };

    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.textContent = "Reset Weights";
    Object.assign(resetBtn.style, {
        marginTop: "2px",
        padding: "4px 6px",
        borderRadius: "5px",
        border: "1px solid rgba(255,255,255,0.16)",
        backgroundColor: "rgba(30, 41, 59, 0.9)",
        color: "#f8fafc",
        fontSize: "10px",
        cursor: "pointer",
    });
    stopViewportEvents(resetBtn);
    resetBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        reset(true);
    });
    panel.appendChild(resetBtn);

    return {
        panel,
        getState: () => ({ ...state }),
        setState,
        reset,
    };
}
