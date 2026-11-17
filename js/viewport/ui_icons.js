const SVG_NS = "http://www.w3.org/2000/svg";

const ICON_PATHS = {
    upload: [
        { tag: "path", d: "M12 4v10m0 0l-4-4m4 4l4-4", stroke: true },
        { tag: "path", d: "M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3", stroke: true },
    ],
    close: [
        { tag: "path", d: "M6 6l12 12M18 6L6 18", stroke: true, width: "2.4" },
    ],
    sculpt: [
        { tag: "path", d: "M13.5 3.5c2.2 0 4 1.8 4 4.2 0 2.1-1 3.7-2.2 5.3-.9 1.2-1.2 2.4-.5 3.8.5 1 .1 2.2-1 2.7-1.6.8-3.8 1-5.6.3-1.4-.5-2-1.8-1.5-3.2.5-1.5 1.8-2.6 2-4.4.2-1.6-.6-3-.2-4.6.5-2.4 2.5-4.1 5-4.1z", fill: "currentColor", opacity: "0.9" },
        { tag: "circle", cx: "16.8", cy: "11.5", r: "1.4", fill: "#38bdf8" },
    ],
    weights: [
        { tag: "path", d: "M4 7h16M4 12h16M4 17h16", stroke: true },
        { tag: "circle", cx: "9", cy: "7", r: "2.1", fill: "currentColor" },
        { tag: "circle", cx: "15", cy: "12", r: "2.1", fill: "currentColor" },
        { tag: "circle", cx: "10", cy: "17", r: "2.1", fill: "currentColor" },
    ],
    lights: [
        { tag: "circle", cx: "12", cy: "12", r: "4", stroke: true },
        { tag: "path", d: "M12 2v3M12 19v3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M2 12h3M19 12h3M4.9 19.1L7 17M17 7l2.1-2.1", stroke: true },
    ],
    mocap: [
        { tag: "rect", x: "3", y: "6", width: "13", height: "12", rx: "2.2", stroke: true },
        { tag: "path", d: "M16 10l5-3v10l-5-3z", stroke: true },
        { tag: "circle", cx: "9.5", cy: "12", r: "2.2", fill: "currentColor" },
    ],
    render: [
        { tag: "circle", cx: "12", cy: "12", r: "8.5", stroke: true },
        { tag: "path", d: "M12 3.5a8.5 8.5 0 0 1 8.5 8.5M3.5 12h17M12 3.5c-2.6 2.5-2.6 14.5 0 17M12 3.5c2.6 2.5 2.6 14.5 0 17", stroke: true, width: "1.5" },
    ],
    chevron: [
        { tag: "path", d: "M6 9l6 6 6-6z", fill: "currentColor" },
    ],
};

/**
 * Prevent HUD pointer and wheel events from propagating into Three.js OrbitControls or LiteGraph zoom.
 */
export function stopViewportEvents(el) {
    el.addEventListener("pointerdown", (e) => e.stopPropagation());
    el.addEventListener("wheel", (e) => e.stopPropagation());
}

/**
 * Build a lightweight inline SVG vector icon element with zero external asset dependencies.
 */
export function createSvgIcon(name, size = 16, color = "currentColor") {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("width", String(size));
    svg.setAttribute("height", String(size));
    svg.style.display = "block";
    svg.style.flexShrink = "0";
    svg.style.color = color;

    const specs = ICON_PATHS[name] || [];
    for (const spec of specs) {
        const node = document.createElementNS(SVG_NS, spec.tag);
        for (const [k, v] of Object.entries(spec)) {
            if (k === "tag" || k === "stroke" || k === "width") continue;
            node.setAttribute(k, v);
        }
        if (spec.stroke) {
            node.setAttribute("fill", spec.fill || "none");
            node.setAttribute("stroke", "currentColor");
            node.setAttribute("stroke-width", spec.width || "1.8");
            node.setAttribute("stroke-linecap", "round");
            node.setAttribute("stroke-linejoin", "round");
        } else if (!spec.fill) {
            node.setAttribute("fill", "currentColor");
        }
        svg.appendChild(node);
    }
    return svg;
}
