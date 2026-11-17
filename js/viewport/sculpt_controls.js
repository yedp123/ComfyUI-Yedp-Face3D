import { stopViewportEvents } from "./ui_icons.js";

export const DEFAULT_SCULPT_STATE = {
    faceDepth: 1.0,
    noseBridge: 0.0,
    noseTip: 0.0,
    lipsDepth: 0.0,
    chin: 0.0,
    cheeks: 0.0,
    jawWidth: 0.0,
};

/**
 * Collapsible category body for global Face Depth and anatomical push-in / push-out sculpting.
 */
export function createSculptPanel(onChange) {
    const state = { ...DEFAULT_SCULPT_STATE };

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

    const formatVal = (key, v) => {
        if (key === "faceDepth") return `${v.toFixed(2)}×`;
        return (v > 0 ? "+" : "") + v.toFixed(2);
    };

    const addSlider = (label, key, min = -1, max = 1, step = 0.05) => {
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
        valEl.style.width = "38px";
        valEl.style.textAlign = "right";
        valEl.style.fontFamily = "ui-monospace, monospace";
        valEl.textContent = formatVal(key, state[key]);

        input.addEventListener("input", () => {
            state[key] = Number(input.value);
            valEl.textContent = formatVal(key, state[key]);
            onChange({ ...state });
        });

        setters.push(() => {
            input.value = String(state[key]);
            valEl.textContent = formatVal(key, state[key]);
        });

        row.append(lbl, input, valEl);
        panel.appendChild(row);
    };

    addSlider("Face Depth", "faceDepth", 0.35, 2.0, 0.05);
    addSlider("Nose Bridge", "noseBridge");
    addSlider("Nose Tip", "noseTip");
    addSlider("Lips Depth", "lipsDepth");
    addSlider("Chin", "chin");
    addSlider("Cheeks", "cheeks");
    addSlider("Jaw Width", "jawWidth");

    const reset = (notify = true) => {
        Object.assign(state, DEFAULT_SCULPT_STATE);
        for (const s of setters) s();
        if (notify) onChange({ ...state });
    };

    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.textContent = "Reset Sculpt";
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
        reset,
    };
}
