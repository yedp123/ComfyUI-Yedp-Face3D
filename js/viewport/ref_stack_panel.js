import { createSvgIcon, stopViewportEvents } from "./ui_icons.js";

const PRIMARY_SLOT = { key: "front", label: "Front", size: 74, iconSize: 28 };
const PAIR_ROWS = [
    [
        { key: "three_quarter", label: "3/4 A", size: 46, iconSize: 18 },
        { key: "three_quarter_2", label: "3/4 B", size: 46, iconSize: 18 },
    ],
    [
        { key: "side", label: "Side A", size: 46, iconSize: 18 },
        { key: "side_2", label: "Side B", size: 46, iconSize: 18 },
    ],
];

function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("Failed to read local image file."));
        reader.readAsDataURL(file);
    });
}

/**
 * Left-side compact reference stack: `Front` on top, followed by `3/4 A/B` and `Side A/B` pairs
 * with per-slot upload/drop/clear and full `RESET`.
 */
export function createRefStackPanel({ onSelectFile, onClearSlot, onResetAll, onError }) {
    let clearHandler = onClearSlot || null;

    const column = document.createElement("div");
    Object.assign(column.style, {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "5px",
        pointerEvents: "auto",
        width: "100px",
        flexShrink: "0",
    });
    stopViewportEvents(column);

    const slots = {};

    function setPreview(viewKey, dataUrl, active = Boolean(dataUrl)) {
        const slot = slots[viewKey];
        if (!slot) return;
        if (dataUrl) {
            slot.thumb.src = dataUrl;
            slot.thumb.style.display = "block";
            slot.uploadIcon.style.display = "none";
            slot.clearBtn.style.display = "flex";
            slot.card.style.borderColor = active ? "#38bdf8" : "rgba(255, 255, 255, 0.28)";
        } else {
            slot.thumb.removeAttribute("src");
            slot.thumb.style.display = "none";
            slot.uploadIcon.style.display = "block";
            slot.clearBtn.style.display = "none";
            slot.card.style.borderColor = "rgba(255, 255, 255, 0.14)";
        }
    }

    function buildSlotElement({ key, label, size, iconSize }) {
        const item = document.createElement("div");
        Object.assign(item.style, {
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "2px",
        });

        const caption = document.createElement("span");
        caption.textContent = label;
        Object.assign(caption.style, {
            fontSize: "9px",
            fontWeight: "500",
            color: "#cbd5e1",
            letterSpacing: "0.01em",
            textShadow: "0 1px 2px rgba(0,0,0,0.8)",
            lineHeight: "1.1",
            whiteSpace: "nowrap",
        });

        const card = document.createElement("div");
        Object.assign(card.style, {
            position: "relative",
            width: `${size}px`,
            height: `${size}px`,
            borderRadius: "7px",
            border: "1px solid rgba(255, 255, 255, 0.14)",
            backgroundColor: "rgba(15, 19, 28, 0.86)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            overflow: "hidden",
            boxSizing: "border-box",
            transition: "border-color 0.15s ease",
        });
        card.title = `Import ${label} portrait`;

        const uploadIcon = createSvgIcon("upload", iconSize, "#64748b");

        const thumb = document.createElement("img");
        thumb.alt = `${label} preview`;
        Object.assign(thumb.style, {
            width: "100%",
            height: "100%",
            objectFit: "cover",
            display: "none",
        });

        const clearBtn = document.createElement("button");
        clearBtn.type = "button";
        clearBtn.textContent = "×";
        clearBtn.title = `Clear ${label}`;
        Object.assign(clearBtn.style, {
            position: "absolute",
            top: "2px",
            right: "2px",
            width: "15px",
            height: "15px",
            padding: "0",
            borderRadius: "4px",
            border: "1px solid rgba(255, 255, 255, 0.28)",
            backgroundColor: "rgba(15, 23, 42, 0.85)",
            color: "#f8fafc",
            fontSize: "11px",
            lineHeight: "13px",
            fontWeight: "700",
            display: "none",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            zIndex: "2",
        });
        stopViewportEvents(clearBtn);
        clearBtn.addEventListener("click", async (e) => {
            e.preventDefault();
            e.stopPropagation();
            setPreview(key, null, false);
            await clearHandler?.(key);
        });

        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.style.display = "none";

        card.addEventListener("click", (e) => {
            e.stopPropagation();
            input.click();
        });

        const handleFile = async (file) => {
            if (!file) return;
            try {
                const dataUrl = await readFileAsDataUrl(file);
                setPreview(key, dataUrl);
                await onSelectFile(key, dataUrl, file);
            } catch (err) {
                onError?.(`Error: ${err.message}`);
            }
        };

        input.addEventListener("change", async () => {
            const file = input.files?.[0];
            input.value = "";
            await handleFile(file);
        });

        card.addEventListener("dragover", (e) => {
            e.preventDefault();
            e.stopPropagation();
            card.style.borderColor = "#38bdf8";
        });
        card.addEventListener("dragleave", (e) => {
            e.stopPropagation();
            card.style.borderColor = thumb.style.display === "block"
                ? "rgba(255, 255, 255, 0.28)"
                : "rgba(255, 255, 255, 0.14)";
        });
        card.addEventListener("drop", async (e) => {
            e.preventDefault();
            e.stopPropagation();
            const file = e.dataTransfer?.files?.[0];
            if (file && (!file.type || file.type.startsWith("image/"))) {
                await handleFile(file);
            }
        });

        card.append(uploadIcon, thumb, clearBtn, input);
        item.append(caption, card);

        slots[key] = { card, uploadIcon, thumb, clearBtn };
        return item;
    }

    column.appendChild(buildSlotElement(PRIMARY_SLOT));

    for (const pair of PAIR_ROWS) {
        const row = document.createElement("div");
        Object.assign(row.style, {
            display: "flex",
            flexDirection: "row",
            justifyContent: "center",
            gap: "5px",
            width: "100%",
        });
        for (const spec of pair) {
            row.appendChild(buildSlotElement(spec));
        }
        column.appendChild(row);
    }

    const resetAllBtn = document.createElement("button");
    resetAllBtn.type = "button";
    resetAllBtn.textContent = "RESET";
    resetAllBtn.title = "Clear all portraits, sculpts, weights, mocap clips, and cached state";
    Object.assign(resetAllBtn.style, {
        width: "96px",
        marginTop: "2px",
        padding: "5px 0",
        borderRadius: "6px",
        border: "1px solid rgba(248, 113, 113, 0.45)",
        backgroundColor: "rgba(127, 29, 29, 0.45)",
        color: "#f8fafc",
        fontSize: "11px",
        fontWeight: "700",
        letterSpacing: "0.08em",
        cursor: "pointer",
    });
    stopViewportEvents(resetAllBtn);
    resetAllBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        onResetAll();
    });
    column.appendChild(resetAllBtn);

    function setSlotActive(viewKey, active) {
        const slot = slots[viewKey];
        if (!slot) return;
        slot.card.style.borderColor = active ? "#38bdf8" : "rgba(255, 255, 255, 0.14)";
    }

    function setClearHandler(fn) {
        clearHandler = typeof fn === "function" ? fn : null;
    }

    function clearAll() {
        for (const key of Object.keys(slots)) {
            setPreview(key, null, false);
        }
    }

    return {
        element: column,
        resetAllBtn,
        setPreview,
        setSlotActive,
        setClearHandler,
        clearAll,
    };
}
