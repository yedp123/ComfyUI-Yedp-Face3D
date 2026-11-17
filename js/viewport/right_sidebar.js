import { createSvgIcon, stopViewportEvents } from "./ui_icons.js";

/**
 * Right-hand vertical panel with collapsible icon-labeled accordion categories
 * (`SCULPT`, `WEIGHTS`, `LIGHTS`, `MOCAP`, `RENDER`).
 */
export function createRightSidebar(sections) {
    const sidebar = document.createElement("div");
    Object.assign(sidebar.style, {
        width: "224px",
        minWidth: "200px",
        flexShrink: "0",
        height: "100%",
        backgroundColor: "rgba(13, 17, 24, 0.96)",
        borderLeft: "1px solid rgba(255, 255, 255, 0.12)",
        display: "flex",
        flexDirection: "column",
        overflowY: "auto",
        overflowX: "hidden",
        pointerEvents: "auto",
        boxSizing: "border-box",
    });
    stopViewportEvents(sidebar);

    const state = {};
    const controllers = {};

    for (const { id, title, icon, body, defaultOpen = false } of sections) {
        state[id] = Boolean(defaultOpen);

        const group = document.createElement("div");
        Object.assign(group.style, {
            display: "flex",
            flexDirection: "column",
            borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
        });

        const header = document.createElement("button");
        header.type = "button";
        Object.assign(header.style, {
            width: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "8px",
            padding: "10px 12px",
            backgroundColor: "rgba(255, 255, 255, 0.03)",
            border: "none",
            borderBottom: "1px solid rgba(255, 255, 255, 0.06)",
            color: "#cbd5e1",
            cursor: "pointer",
            textAlign: "left",
        });
        stopViewportEvents(header);

        const leftCluster = document.createElement("div");
        Object.assign(leftCluster.style, {
            display: "flex",
            alignItems: "center",
            gap: "9px",
            minWidth: "0",
        });

        const leftIcon = createSvgIcon(icon, 18, "#e2e8f0");
        const titleEl = document.createElement("span");
        titleEl.textContent = title;
        Object.assign(titleEl.style, {
            fontSize: "11px",
            fontWeight: "600",
            letterSpacing: "0.06em",
            color: "#e2e8f0",
        });
        leftCluster.append(leftIcon, titleEl);

        const chevron = createSvgIcon("chevron", 14, "#cbd5e1");
        chevron.style.transition = "transform 0.15s ease";

        const bodyWrap = document.createElement("div");
        Object.assign(bodyWrap.style, {
            backgroundColor: "rgba(10, 13, 19, 0.72)",
        });
        bodyWrap.appendChild(body);

        const applyOpen = (open) => {
            state[id] = open;
            bodyWrap.style.display = open ? "block" : "none";
            header.style.backgroundColor = open
                ? "rgba(56, 189, 248, 0.08)"
                : "rgba(255, 255, 255, 0.03)";
            titleEl.style.color = open ? "#f8fafc" : "#cbd5e1";
            chevron.style.transform = open ? "rotate(0deg)" : "rotate(-90deg)";
        };
        applyOpen(state[id]);

        header.addEventListener("click", (e) => {
            e.stopPropagation();
            applyOpen(!state[id]);
        });

        header.append(leftCluster, chevron);
        group.append(header, bodyWrap);
        sidebar.appendChild(group);

        controllers[id] = applyOpen;
    }

    return {
        element: sidebar,
        setSectionOpen(id, open) {
            controllers[id]?.(Boolean(open));
        },
        isSectionOpen(id) {
            return Boolean(state[id]);
        },
        collapseAll() {
            for (const fn of Object.values(controllers)) fn(false);
        },
    };
}
