import { createSvgIcon, stopViewportEvents } from "./ui_icons.js";

const VIEW_SLOTS = [
    { key: "front", label: "front" },
    { key: "three_quarter", label: "three-quarter" },
    { key: "side", label: "side" },
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
 * Left-side vertical stack of portrait import/preview slots (`front`, `three-quarter`, `side`)
 * and a prominent full-reset `RESET` button.
 */
export function createRefStackPanel({ onSelectFile, onResetAll, onError }) {
    const column = document.createElement("div");
    Object.assign(column.style, {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "6px",
        pointerEvents: "auto",
        width: "84px",
        flexShrink: "0",
    });
    stopViewportEvents(column);

    const slots = {};

    for (const { key, label } of VIEW_SLOTS) {
        const item = document.createElement("div");
        Object.assign(item.style, {
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "3px",
            width: "100%",
        });

        const caption = document.createElement("span");
        caption.textContent = label;
        Object.assign(caption.style, {
            fontSize: "10px",
            fontWeight: "500",
            color: "#cbd5e1",
            letterSpacing: "0.01em",
            textShadow: "0 1px 2px rgba(0,0,0,0.8)",
            lineHeight: "1.1",
        });

        const card = document.createElement("div");
        Object.assign(card.style, {
            position: "relative",
            width: "76px",
            height: "76px",
            borderRadius: "8px",
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

        const uploadIcon = createSvgIcon("upload", 30, "#64748b");

        const thumb = document.createElement("img");
        thumb.alt = `${label} preview`;
        Object.assign(thumb.style, {
            width: "100%",
            height: "100%",
            objectFit: "cover",
            display: "none",
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

        card.append(uploadIcon, thumb, input);
        item.append(caption, card);
        column.appendChild(item);

        slots[key] = { card, uploadIcon, thumb };
    }

    const resetAllBtn = document.createElement("button");
    resetAllBtn.type = "button";
    resetAllBtn.textContent = "RESET";
    resetAllBtn.title = "Clear all portraits, sculpts, weights, mocap clips, and cached state";
    Object.assign(resetAllBtn.style, {
        width: "76px",
        marginTop: "4px",
        padding: "6px 0",
        borderRadius: "6px",
        border: "1px solid rgba(248, 113, 113, 0.45)",
        backgroundColor: "rgba(127, 29, 29, 0.45)",
        color: "#f8fafc",
        fontSize: "12px",
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

    function setPreview(viewKey, dataUrl, active = Boolean(dataUrl)) {
        const slot = slots[viewKey];
        if (!slot) return;
        if (dataUrl) {
            slot.thumb.src = dataUrl;
            slot.thumb.style.display = "block";
            slot.uploadIcon.style.display = "none";
            slot.card.style.borderColor = active ? "#38bdf8" : "rgba(255, 255, 255, 0.28)";
        } else {
            slot.thumb.removeAttribute("src");
            slot.thumb.style.display = "none";
            slot.uploadIcon.style.display = "block";
            slot.card.style.borderColor = "rgba(255, 255, 255, 0.14)";
        }
    }

    function setSlotActive(viewKey, active) {
        const slot = slots[viewKey];
        if (!slot) return;
        slot.card.style.borderColor = active ? "#38bdf8" : "rgba(255, 255, 255, 0.14)";
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
        clearAll,
    };
}
