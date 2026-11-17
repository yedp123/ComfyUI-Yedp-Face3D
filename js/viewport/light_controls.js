import { stopViewportEvents } from "./ui_icons.js";

export const DEFAULT_LIGHT_STATE = {
    keyAzimuth: 35,
    keyElevation: 28,
    keyIntensity: 0.78,
    keyColor: "#fff5ea",
    rimAzimuth: -135,
    rimElevation: 22,
    rimIntensity: 0.42,
    rimColor: "#7dd3fc",
    ambientIntensity: 0.52,
};

/**
 * Collapsible category body for Key, Rim, and Ambient 3D lighting.
 */
export function createLightPanel(onChange) {
    const state = { ...DEFAULT_LIGHT_STATE };

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

    const addSlider = (label, key, min, max, step, suffix = "") => {
        const row = document.createElement("div");
        Object.assign(row.style, { display: "flex", alignItems: "center", gap: "6px" });

        const lbl = document.createElement("span");
        lbl.textContent = label;
        lbl.style.width = "48px";

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
        valEl.textContent = `${state[key]}${suffix}`;

        input.addEventListener("input", () => {
            state[key] = Number(input.value);
            valEl.textContent = `${state[key]}${suffix}`;
            onChange({ ...state });
        });

        setters.push(() => {
            input.value = String(state[key]);
            valEl.textContent = `${state[key]}${suffix}`;
        });

        row.append(lbl, input, valEl);
        panel.appendChild(row);
    };

    const addColor = (label, key) => {
        const row = document.createElement("div");
        Object.assign(row.style, { display: "flex", alignItems: "center", justifyContent: "space-between" });
        const lbl = document.createElement("span");
        lbl.textContent = label;
        const input = document.createElement("input");
        input.type = "color";
        input.value = state[key];
        Object.assign(input.style, { width: "40px", height: "18px", border: "none", background: "transparent", cursor: "pointer" });
        stopViewportEvents(input);
        input.addEventListener("input", () => {
            state[key] = input.value;
            onChange({ ...state });
        });
        setters.push(() => { input.value = state[key]; });
        row.append(lbl, input);
        panel.appendChild(row);
    };

    addSection("Key Light");
    addSlider("Angle", "keyAzimuth", -180, 180, 5, "°");
    addSlider("Height", "keyElevation", -60, 75, 5, "°");
    addSlider("Power", "keyIntensity", 0, 2, 0.05);
    addColor("Color", "keyColor");

    addSection("Rim Light");
    addSlider("Angle", "rimAzimuth", -180, 180, 5, "°");
    addSlider("Height", "rimElevation", -60, 75, 5, "°");
    addSlider("Power", "rimIntensity", 0, 2, 0.05);
    addColor("Color", "rimColor");

    addSection("Environment");
    addSlider("Ambient", "ambientIntensity", 0, 1.2, 0.05);

    const reset = (notify = true) => {
        Object.assign(state, DEFAULT_LIGHT_STATE);
        for (const s of setters) s();
        if (notify) onChange({ ...state });
    };

    const resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.textContent = "Reset Lights";
    Object.assign(resetBtn.style, {
        marginTop: "4px",
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
