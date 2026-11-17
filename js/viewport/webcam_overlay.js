import { stopViewportEvents } from "./ui_icons.js";

const FACE_CONTOUR_GROUPS = [
    [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10], // Jaw/Face Oval
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33], // Right Eye
    [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362], // Left Eye
    [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61], // Outer Lips
    [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78], // Inner Lips
    [46, 53, 52, 65, 55, 107, 66, 105, 63, 70], // Right Brow
    [276, 283, 282, 295, 285, 336, 296, 334, 293, 300], // Left Brow
    [168, 6, 197, 195, 5, 4, 1, 2], // Nose Bridge
];

/**
 * Mirrored webcam canvas preview with MediaPipe landmark contours and ARKit telemetry,
 * sized to embed cleanly inside the right-hand MOCAP accordion section.
 */
export function createWebcamOverlay() {
    const wrapper = document.createElement("div");
    Object.assign(wrapper.style, {
        position: "relative",
        width: "100%",
        height: "52px",
        borderRadius: "6px",
        border: "1px solid rgba(255, 255, 255, 0.12)",
        backgroundColor: "#090d14",
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        boxSizing: "border-box",
    });
    stopViewportEvents(wrapper);

    const idleLabel = document.createElement("span");
    idleLabel.textContent = "Camera preview inactive";
    Object.assign(idleLabel.style, {
        fontSize: "10px",
        color: "#64748b",
        pointerEvents: "none",
    });

    const canvas = document.createElement("canvas");
    Object.assign(canvas.style, {
        width: "100%",
        height: "100%",
        display: "none",
        objectFit: "cover",
    });

    const telemetry = document.createElement("div");
    Object.assign(telemetry.style, {
        position: "absolute",
        left: "4px",
        right: "4px",
        bottom: "4px",
        fontSize: "9px",
        fontFamily: "ui-monospace, monospace",
        color: "#4ade80",
        backgroundColor: "rgba(0, 0, 0, 0.68)",
        padding: "2px 5px",
        borderRadius: "4px",
        whiteSpace: "nowrap",
        overflow: "hidden",
        textOverflow: "ellipsis",
        pointerEvents: "none",
        display: "none",
    });
    telemetry.textContent = "Waiting for face...";

    wrapper.append(idleLabel, canvas, telemetry);

    function drawFrame(videoEl, detection, bs) {
        const w = videoEl.videoWidth || 640;
        const h = videoEl.videoHeight || 480;
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
        const ctx = canvas.getContext("2d");
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-w, 0);
        ctx.drawImage(videoEl, 0, 0, w, h);
        ctx.restore();

        if (detection?.landmarks?.length) {
            const lms = detection.landmarks;
            ctx.strokeStyle = "rgba(56, 189, 248, 0.75)";
            ctx.lineWidth = Math.max(1, w / 320);
            for (const group of FACE_CONTOUR_GROUPS) {
                ctx.beginPath();
                for (let i = 0; i < group.length; i++) {
                    const p = lms[group[i]];
                    if (!p) continue;
                    const px = (1 - p.x) * w;
                    const py = p.y * h;
                    if (i === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                }
                ctx.stroke();
            }

            ctx.fillStyle = "#22c55e";
            const dotR = Math.max(1.2, w / 320);
            for (let i = 0; i < lms.length; i += 2) {
                const p = lms[i];
                ctx.beginPath();
                ctx.arc((1 - p.x) * w, p.y * h, dotR, 0, 2 * Math.PI);
                ctx.fill();
            }

            if (bs) {
                const jaw = (bs.jawOpen || 0).toFixed(2);
                const smile = Math.max(bs.mouthSmileLeft || 0, bs.mouthSmileRight || 0).toFixed(2);
                const blink = Math.max(bs.eyeBlinkLeft || 0, bs.eyeBlinkRight || 0).toFixed(2);
                const brow = Math.max(bs.browInnerUp || 0, bs.browOuterUpLeft || 0, bs.browOuterUpRight || 0).toFixed(2);
                telemetry.textContent = `jaw:${jaw} sml:${smile} blk:${blink} brw:${brow}`;
            }
        } else {
            telemetry.textContent = "No face detected";
        }
    }

    return {
        element: wrapper,
        show() {
            wrapper.style.height = "auto";
            wrapper.style.aspectRatio = "4 / 3";
            wrapper.style.borderColor = "rgba(56, 189, 248, 0.55)";
            idleLabel.style.display = "none";
            canvas.style.display = "block";
            telemetry.style.display = "block";
        },
        hide() {
            wrapper.style.aspectRatio = "auto";
            wrapper.style.height = "52px";
            wrapper.style.borderColor = "rgba(255, 255, 255, 0.12)";
            idleLabel.style.display = "block";
            canvas.style.display = "none";
            telemetry.style.display = "none";
            telemetry.textContent = "Waiting for face...";
            const ctx = canvas.getContext("2d");
            if (ctx && canvas.width > 0 && canvas.height > 0) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        },
        drawFrame,
    };
}
