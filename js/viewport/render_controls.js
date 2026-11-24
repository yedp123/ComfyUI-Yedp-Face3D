import { stopViewportEvents } from "./ui_icons.js";

export const DEFAULT_RENDER_STATE = {
    motionMode: "rest",
    spinMode: "turntable",
    clipStart: 1,
    lockHeadPose: false,
    projection: "ortho",
    showGate: true,
    fov: 36,
    elevation: 0,
    fitMargin: 0.12,
    startAngle: -90,
    endAngle: 90,
    depthMode: "fixed",
    depthInvert: false,
    flipX: false,
    flipY: false,
    flipZ: false,
    normalBg: "black",
    clayKeyAz: 30,
    clayKeyEl: 35,
    clayKeyPower: 0.85,
    clayFill: 0.35,
    bgBrightness: 0.0,
    maskMode: "full",
    antialias: "normal",
    poseMode: "head_face",
    poseDotScale: 1.0,
    poseOcclude: true,
    poseOverlay: false,
};

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
        padding: "4px 7px",
        cursor: "pointer",
    });
    stopViewportEvents(btn);
    return btn;
}

/**
 * Collapsible RENDER category body: Camera framing (Ortho/Persp, Fit Head, Elevation),
 * Depth/Normal/Clay/Mask pass controls, and Turntable Start/End sweep settings.
 */
export function createRenderPanel(onChange, onFitHead) {
    const state = { ...DEFAULT_RENDER_STATE };

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

    const addSlider = (label, key, min, max, step, suffix = "", scalePct = false) => {
        const row = document.createElement("div");
        Object.assign(row.style, { display: "flex", alignItems: "center", gap: "5px" });

        const lbl = document.createElement("span");
        lbl.textContent = label;
        lbl.style.width = "54px";

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
        valEl.style.width = "36px";
        valEl.style.textAlign = "right";
        valEl.style.fontFamily = "ui-monospace, monospace";
        const fmt = (v) => (scalePct ? `${Math.round(v * 100)}%` : `${step < 1 ? v.toFixed(2) : v}${suffix}`);
        valEl.textContent = fmt(state[key]);

        input.addEventListener("input", () => {
            state[key] = Number(input.value);
            valEl.textContent = fmt(state[key]);
            onChange?.({ ...state }, key);
        });

        setters.push(() => {
            input.value = String(state[key]);
            valEl.textContent = fmt(state[key]);
        });

        row.append(lbl, input, valEl);
        panel.appendChild(row);
        return row;
    };

    const addSegmented = (label, key, options) => {
        const row = document.createElement("div");
        Object.assign(row.style, { display: "flex", alignItems: "center", justifyContent: "space-between", gap: "4px" });
        const grp = document.createElement("div");
        Object.assign(grp.style, { display: "flex", gap: "3px", width: label ? "auto" : "100%" });

        const btns = {};
        const sync = () => {
            for (const opt of options) {
                const active = state[key] === opt.value;
                btns[opt.value].style.borderColor = active ? "#38bdf8" : "rgba(255,255,255,0.16)";
                btns[opt.value].style.color = active ? "#38bdf8" : "#cbd5e1";
            }
        };

        for (const opt of options) {
            const b = makeRenderButton(opt.label, opt.title || "");
            b.style.padding = "2px 6px";
            if (!label) b.style.flex = "1";
            b.addEventListener("click", (e) => {
                e.stopPropagation();
                state[key] = opt.value;
                sync();
                if (key === "projection") syncFovVisibility();
                onChange?.({ ...state }, key);
            });
            btns[opt.value] = b;
            grp.appendChild(b);
        }
        sync();
        setters.push(sync);
        if (label) {
            const lbl = document.createElement("span");
            lbl.textContent = label;
            row.append(lbl, grp);
        } else {
            row.appendChild(grp);
        }
        panel.appendChild(row);
    };

    const addToggleRow = (items) => {
        const row = document.createElement("div");
        Object.assign(row.style, { display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center" });
        for (const { label, key } of items) {
            const wrap = document.createElement("label");
            Object.assign(wrap.style, { display: "inline-flex", alignItems: "center", gap: "4px", cursor: "pointer" });
            const cb = document.createElement("input");
            cb.type = "checkbox";
            cb.checked = Boolean(state[key]);
            stopViewportEvents(cb);
            cb.addEventListener("change", () => {
                state[key] = cb.checked;
                onChange?.({ ...state }, key);
            });
            setters.push(() => {
                cb.checked = Boolean(state[key]);
            });
            const span = document.createElement("span");
            span.textContent = label;
            wrap.append(cb, span);
            row.appendChild(wrap);
        }
        panel.appendChild(row);
    };

    // --- Queue Export Mode ---
    addSection("Queue Export");
    addSegmented("Motion", "motionMode", [
        { label: "Rest", value: "rest", title: "Neutral sculpted rest pose" },
        { label: "Snap", value: "snap", title: "Use saved expression snapshot" },
        { label: "Clip", value: "clip", title: "Play recorded webcam or loaded video clip" },
    ]);
    addSegmented("Camera", "spinMode", [
        { label: "Single 1f", value: "single", title: "Export 1 frame using exact live viewport camera & pose" },
        { label: "Static", value: "static", title: "Render frame_count frames from live camera without spinning" },
        { label: "Turntable", value: "turntable", title: "Rotate from Start Deg to End Deg over rendered frames" },
    ]);

    const clipStartRow = document.createElement("div");
    Object.assign(clipStartRow.style, { display: "flex", alignItems: "center", justifyContent: "space-between", gap: "6px" });
    const clipStartLbl = document.createElement("span");
    clipStartLbl.textContent = "Clip Start F";
    const clipStartInput = document.createElement("input");
    clipStartInput.type = "number";
    clipStartInput.min = "1";
    clipStartInput.max = "600";
    clipStartInput.step = "1";
    clipStartInput.value = String(state.clipStart);
    Object.assign(clipStartInput.style, {
        width: "64px",
        fontSize: "10px",
        fontFamily: "ui-monospace, monospace",
        color: "#f8fafc",
        backgroundColor: "rgba(15, 23, 42, 0.85)",
        border: "1px solid rgba(255, 255, 255, 0.16)",
        borderRadius: "4px",
        padding: "2px 6px",
    });
    stopViewportEvents(clipStartInput);
    clipStartInput.addEventListener("input", () => {
        const val = Math.max(1, Math.min(600, Math.floor(Number(clipStartInput.value) || 1)));
        state.clipStart = val;
        onChange?.({ ...state }, "clipStart");
    });
    const syncClipStartVisibility = () => {
        clipStartInput.value = String(Math.max(1, Math.floor(Number(state.clipStart) || 1)));
        clipStartRow.style.display = state.motionMode === "clip" ? "flex" : "none";
    };
    syncClipStartVisibility();
    setters.push(syncClipStartVisibility);
    clipStartRow.append(clipStartLbl, clipStartInput);
    panel.appendChild(clipStartRow);

    // --- Camera ---
    addSection("Camera & Geometry");
    addSegmented("Projection", "projection", [
        { label: "Ortho", value: "ortho" },
        { label: "Persp", value: "persp" },
    ]);
    const fovRow = addSlider("Persp FOV", "fov", 18, 65, 1, "°");
    const syncFovVisibility = () => {
        fovRow.style.display = state.projection === "persp" ? "flex" : "none";
    };
    syncFovVisibility();
    setters.push(syncFovVisibility);

    addSlider("Elevation", "elevation", -35, 35, 1, "°");
    addSlider("Fit Margin", "fitMargin", 0.0, 0.35, 0.02, "", true);

    const camBtnRow = document.createElement("div");
    Object.assign(camBtnRow.style, { display: "flex", gap: "4px" });
    const fitHeadBtn = makeRenderButton("Fit Head", "Frame reconstructed head with margin");
    const guidesBtn = makeRenderButton("Geo: Head+Neck", "Toggle full head shell + ears + neck vs. face only (viewport & all passes)");
    fitHeadBtn.style.flex = "0 0 auto";
    guidesBtn.style.flex = "1";
    fitHeadBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        onFitHead?.(state);
    });
    camBtnRow.append(fitHeadBtn, guidesBtn);
    panel.appendChild(camBtnRow);
    addToggleRow([{ label: "Res Gate", key: "showGate" }]);

    // --- Turntable ---
    addSection("Turntable Preview & Export");
    addSlider("Start Deg", "startAngle", -180, 180, 5, "°");
    addSlider("End Deg", "endAngle", -180, 360, 5, "°");
    addToggleRow([{ label: "Lock Head Pose", key: "lockHeadPose" }]);
    const turntableBtn = makeRenderButton("Play Turntable", "Play/pause turntable sweep between Start and End angles");
    turntableBtn.style.width = "100%";
    panel.appendChild(turntableBtn);

    const renderStatus = document.createElement("div");
    renderStatus.textContent = "Idle · Ready to queue";
    Object.assign(renderStatus.style, {
        fontSize: "10px",
        fontFamily: "ui-monospace, monospace",
        color: "#94a3b8",
    });
    panel.appendChild(renderStatus);

    // --- Conditioning Passes ---
    addSection("Depth & Normal Passes");
    addSegmented("Depth", "depthMode", [
        { label: "Fixed", value: "fixed" },
        { label: "Dynamic", value: "dynamic" },
    ]);
    addToggleRow([{ label: "Invert Depth", key: "depthInvert" }]);
    addToggleRow([
        { label: "Flip X", key: "flipX" },
        { label: "Flip Y", key: "flipY" },
        { label: "Flip Z", key: "flipZ" },
    ]);
    addSegmented("Normal BG", "normalBg", [
        { label: "Black", value: "black" },
        { label: "Flat", value: "flat" },
    ]);

    addSection("Clay, Mask & AA");
    addSlider("Clay Key", "clayKeyPower", 0, 1.8, 0.05);
    addSlider("Clay Fill", "clayFill", 0, 1.0, 0.05);
    addSlider("Light Az", "clayKeyAz", -180, 180, 5, "°");
    addSlider("Light El", "clayKeyEl", -45, 75, 5, "°");
    addSlider("BG Luma", "bgBrightness", 0, 0.5, 0.02);
    addSegmented("Mask Pass", "maskMode", [
        { label: "Match Geo", value: "full", title: "Mask matches active Geometry (Head+Neck or Face Only)" },
        { label: "Face Only", value: "face", title: "Mask inner face region only even when Head+Neck geometry is active" },
    ]);
    addSegmented("MSAA", "antialias", [
        { label: "Draft", value: "draft" },
        { label: "4x", value: "normal" },
    ]);

    addSection("OpenPose Pass");
    addSegmented("Pose Parts", "poseMode", [
        { label: "Head+Face", value: "head_face", title: "Render Head/Neck OpenPose skeleton + 70-pt white facial keypoints" },
        { label: "Face 70pt", value: "face_only", title: "Render strictly the 70-pt white facial expression keypoints" },
    ]);
    addSlider("Dot Scale", "poseDotScale", 0.4, 2.0, 0.05);
    addToggleRow([
        { label: "Self-Occlude", key: "poseOcclude" },
        { label: "Overlay", key: "poseOverlay" },
    ]);

    const outputSpec = document.createElement("div");
    outputSpec.textContent = "Output: 512x512 · 24f";
    Object.assign(outputSpec.style, {
        fontSize: "9px",
        fontFamily: "ui-monospace, monospace",
        color: "#64748b",
        borderTop: "1px solid rgba(255,255,255,0.08)",
        paddingTop: "5px",
    });
    panel.appendChild(outputSpec);

    const setState = (partial, notify = false) => {
        Object.assign(state, partial);
        for (const s of setters) s();
        if (notify) {
            const firstKey = Object.keys(partial || {})[0] || null;
            onChange?.({ ...state }, firstKey);
        }
    };

    const reset = (notify = true) => {
        Object.assign(state, DEFAULT_RENDER_STATE);
        for (const s of setters) s();
        if (notify) onChange?.({ ...state }, "reset");
    };

    const resetRenderBtn = makeRenderButton("Reset Render Settings");
    resetRenderBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        reset(true);
    });
    panel.appendChild(resetRenderBtn);

    const setGuidesActive = (visible) => {
        guidesBtn.textContent = visible ? "Geo: Head+Neck" : "Geo: Face Only";
        guidesBtn.style.borderColor = visible ? "#38bdf8" : "rgba(255, 255, 255, 0.16)";
        guidesBtn.style.color = visible ? "#38bdf8" : "#cbd5e1";
    };
    setGuidesActive(true);

    const setOutputSpec = (width, height, frames, modeLabel = "Turntable") => {
        outputSpec.textContent = `Output: ${width}x${height} · ${frames}f (${modeLabel})`;
    };

    return {
        panel,
        getState: () => ({ ...state }),
        setState,
        syncProjectionUI: syncFovVisibility,
        turntableBtn,
        guidesBtn,
        renderStatus,
        setGuidesActive,
        setOutputSpec,
        reset,
    };
}
