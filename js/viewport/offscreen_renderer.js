import * as THREE from "../lib/three.module.js";
import { api } from "../../../../scripts/api.js";

function buildOffscreenCamera(liveCamera, target, renderState, width, height, keepLivePose) {
    const outAspect = width / Math.max(height, 1);
    const dist = Math.max(liveCamera.position.distanceTo(target), 0.3);

    let offCam;
    if (renderState.projection === "ortho") {
        let halfH;
        if (liveCamera.isOrthographicCamera) {
            halfH = ((liveCamera.top - liveCamera.bottom) * 0.5) / Math.max(liveCamera.zoom, 1e-4);
        } else {
            halfH = dist * Math.tan(THREE.MathUtils.degToRad(liveCamera.fov || 36) * 0.5);
        }
        offCam = new THREE.OrthographicCamera(
            -halfH * outAspect,
            halfH * outAspect,
            halfH,
            -halfH,
            0.05,
            50.0
        );
    } else {
        offCam = new THREE.PerspectiveCamera(renderState.fov || 36, outAspect, 0.05, 50.0);
    }

    if (keepLivePose) {
        // Keep exact user-orbited camera pose for Single frame or Mocap/Video clip export
        offCam.position.copy(liveCamera.position);
        offCam.quaternion.copy(liveCamera.quaternion);
    } else {
        // In Turntable mode, lock azimuth=0 and apply elevation angle cleanly around target
        const elRad = THREE.MathUtils.degToRad(renderState.elevation || 0);
        offCam.position.set(
            target.x,
            target.y + Math.sin(elRad) * dist,
            target.z + Math.cos(elRad) * dist
        );
        offCam.lookAt(target);
    }
    offCam.updateMatrixWorld(true);
    return offCam;
}

function packRgb(rgbaBuf, outBuf, outOff, nPx) {
    for (let i = 0; i < nPx; i++) {
        const s = i * 4;
        const d = outOff + i * 3;
        outBuf[d] = rgbaBuf[s];
        outBuf[d + 1] = rgbaBuf[s + 1];
        outBuf[d + 2] = rgbaBuf[s + 2];
    }
}

function packChannel(rgbaBuf, channel, outBuf, outOff, nPx) {
    for (let i = 0; i < nPx; i++) {
        outBuf[outOff + i] = rgbaBuf[i * 4 + channel];
    }
}

async function postBinaryChunk(jobId, payload) {
    const res = await api.fetchApi(`/yedp_face3d/render_chunk/${jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: payload,
    });
    if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`render_chunk HTTP ${res.status}: ${txt}`);
    }
}

/**
 * Render all batch frames offscreen across 5 synchronized geometry, color & OpenPose passes:
 * depth (1ch), normal (3ch), clay shaded (3ch) + mask alpha (1ch), textured lit (3ch),
 * and unlit OpenPose Head+Face skeleton (3ch), uploading lossless chunks directly to ComfyUI.
 */
export async function renderAndUploadPasses({
    jobId,
    renderer,
    scene,
    liveCamera,
    controlsTarget,
    grid,
    reconstructedFace,
    placeholderHead,
    motionClip,
    turntableFrames,
    renderState,
    width,
    height,
    onProgress,
}) {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    const nPx = w * h;
    const frameStride = nPx * 14; // depth(1) + normal(3) + shaded(3) + textured(3) + mask(1) + pose(3)
    const framesPerChunk = Math.max(1, Math.min(8, Math.floor((24 * 1024 * 1024) / frameStride)));

    const exportMode = renderState.exportMode || "turntable";
    const hasClip = Array.isArray(motionClip) && motionClip.length > 0 && reconstructedFace.hasMesh;
    if (exportMode === "clip" && !hasClip) {
        throw new Error(
            "Export Mode is 'Clip', but no webcam recording or driving_video is loaded. Switch to Turntable or Single."
        );
    }
    const useClip = exportMode === "clip";
    const isSingle = exportMode === "single";
    const keepLivePose = useClip || isSingle;
    const totalFrames = useClip
        ? motionClip.length
        : isSingle
            ? 1
            : Math.max(1, Math.floor(turntableFrames));
    const startAngle = Number(renderState.startAngle ?? -90);
    const endAngle = Number(renderState.endAngle ?? 90);
    const dynamicDepth = renderState.depthMode === "dynamic";

    const samples = renderState.antialias === "draft" ? 0 : 4;
    const rtSharp = new THREE.WebGLRenderTarget(w, h, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        samples: 0,
    });
    const rtSmooth = new THREE.WebGLRenderTarget(w, h, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        samples,
    });

    // Preserve live viewport visual state across offscreen capture
    const prevTarget = renderer.getRenderTarget();
    const prevClearColor = renderer.getClearColor(new THREE.Color());
    const prevClearAlpha = renderer.getClearAlpha();
    const prevGridVis = grid ? grid.visible : false;
    const prevMode = reconstructedFace.viewMode;
    const prevPos = reconstructedFace.group.position.clone();
    const prevRot = reconstructedFace.group.rotation.clone();
    const prevQuat = reconstructedFace.group.quaternion.clone();
    const prevLastMotion = reconstructedFace.lastMotionFrame;
    const prevPhVis = placeholderHead?.group?.visible ?? false;
    const prevPhRot = placeholderHead?.group?.rotation?.clone();

    const readBuf = new Uint8Array(nPx * 4);
    const chunkBuf = new Uint8Array(16 + framesPerChunk * frameStride);
    const headerView = new DataView(chunkBuf.buffer);
    const bgLuma = Math.max(0.0, Math.min(0.5, Number(renderState.bgBrightness || 0)));
    const bgColor = new THREE.Color(bgLuma, bgLuma, bgLuma);
    const normalBgColor = renderState.normalBg === "flat"
        ? new THREE.Color(128 / 255, 128 / 255, 1.0)
        : new THREE.Color(0, 0, 0);

    try {
        if (grid) grid.visible = false;
        const offCam = buildOffscreenCamera(liveCamera, controlsTarget, renderState, w, h, keepLivePose);
        const activeMesh = reconstructedFace.hasMesh ? reconstructedFace : null;

        reconstructedFace.applyRenderState({ ...renderState, poseOverlay: false });

        // Precompute stable Fixed depth range before looping frames
        let fixedDepthBounds = { near: 0.5, far: 3.0 };
        if (activeMesh && !dynamicDepth) {
            if (!isSingle) {
                reconstructedFace.resetPose();
            }
            fixedDepthBounds = reconstructedFace.updateDepthBounds(offCam, false) || fixedDepthBounds;
        }

        const frameAngles = [];
        const depthRangeLog = dynamicDepth && activeMesh
            ? { near: Infinity, far: -Infinity }
            : { near: fixedDepthBounds.near, far: fixedDepthBounds.far };
        let chunkFrames = 0;
        let chunkStartFrame = 0;

        for (let f = 0; f < totalFrames; f++) {
            const t = totalFrames > 1 ? f / (totalFrames - 1) : 0.5;
            const yawDeg = useClip
                ? 0
                : isSingle
                    ? THREE.MathUtils.radToDeg(activeMesh ? prevRot.y : (prevPhRot?.y || 0))
                    : startAngle + (endAngle - startAngle) * t;
            frameAngles.push(Number(yawDeg.toFixed(2)));

            if (activeMesh) {
                if (useClip) {
                    reconstructedFace.applyMotionFrame(motionClip[f]);
                } else if (!isSingle) {
                    reconstructedFace.resetPose();
                    reconstructedFace.setTurntableYaw(THREE.MathUtils.degToRad(yawDeg));
                }
                if (dynamicDepth) {
                    const b = reconstructedFace.updateDepthBounds(offCam, true);
                    if (b) {
                        depthRangeLog.near = Math.min(depthRangeLog.near, b.near);
                        depthRangeLog.far = Math.max(depthRangeLog.far, b.far);
                    }
                } else {
                    reconstructedFace.material.uniforms.uDepthNear.value = fixedDepthBounds.near;
                    reconstructedFace.material.uniforms.uDepthFar.value = fixedDepthBounds.far;
                    reconstructedFace.guideDepthMat.uniforms.uDepthNear.value = fixedDepthBounds.near;
                    reconstructedFace.guideDepthMat.uniforms.uDepthFar.value = fixedDepthBounds.far;
                }
            } else if (placeholderHead?.group && !isSingle) {
                placeholderHead.group.rotation.set(0, THREE.MathUtils.degToRad(yawDeg), 0);
            }

            const baseOff = 16 + chunkFrames * frameStride;
            const depthOff = baseOff;
            const normalOff = baseOff + nPx;
            const shadedOff = baseOff + nPx * 4;
            const texturedOff = baseOff + nPx * 7;
            const maskOff = baseOff + nPx * 10;
            const poseOff = baseOff + nPx * 11;

            // 1. Depth Pass (unfiltered, background = 0)
            reconstructedFace.setViewMode("depth");
            renderer.setRenderTarget(rtSharp);
            renderer.setClearColor(0x000000, 0.0);
            renderer.clear();
            renderer.render(scene, offCam);
            renderer.readRenderTargetPixels(rtSharp, 0, 0, w, h, readBuf);
            packChannel(readBuf, 0, chunkBuf, depthOff, nPx);

            // 2. Normal Pass (unfiltered camera-space geometry normals)
            reconstructedFace.setViewMode("normals");
            renderer.setRenderTarget(rtSharp);
            renderer.setClearColor(normalBgColor, 0.0);
            renderer.clear();
            renderer.render(scene, offCam);
            renderer.readRenderTargetPixels(rtSharp, 0, 0, w, h, readBuf);
            packRgb(readBuf, chunkBuf, normalOff, nPx);

            // 3. Shaded Clay Pass + Silhouette/Face Mask (Alpha channel)
            reconstructedFace.setViewMode("clay");
            renderer.setRenderTarget(rtSmooth);
            renderer.setClearColor(bgColor, 0.0);
            renderer.clear();
            renderer.render(scene, offCam);
            renderer.readRenderTargetPixels(rtSmooth, 0, 0, w, h, readBuf);
            packRgb(readBuf, chunkBuf, shadedOff, nPx);
            packChannel(readBuf, 3, chunkBuf, maskOff, nPx);

            // 4. Textured Lit Pass
            reconstructedFace.setViewMode("textured");
            renderer.setRenderTarget(rtSmooth);
            renderer.setClearColor(bgColor, 0.0);
            renderer.clear();
            renderer.render(scene, offCam);
            renderer.readRenderTargetPixels(rtSmooth, 0, 0, w, h, readBuf);
            packRgb(readBuf, chunkBuf, texturedOff, nPx);

            // 5. Unlit OpenPose Head + 70-Pt Face Pass (pure black background)
            if (placeholderHead?.group && !activeMesh) {
                placeholderHead.group.visible = false;
            }
            reconstructedFace.setViewMode("pose");
            renderer.setRenderTarget(rtSmooth);
            renderer.setClearColor(0x000000, 0.0);
            renderer.clear();
            renderer.render(scene, offCam);
            renderer.readRenderTargetPixels(rtSmooth, 0, 0, w, h, readBuf);
            packRgb(readBuf, chunkBuf, poseOff, nPx);
            if (placeholderHead?.group && !activeMesh) {
                placeholderHead.group.visible = prevPhVis;
            }

            chunkFrames++;
            onProgress?.(f + 1, totalFrames);

            if (chunkFrames === framesPerChunk || f + 1 === totalFrames) {
                headerView.setUint32(0, chunkStartFrame, true);
                headerView.setUint32(4, chunkFrames, true);
                headerView.setUint32(8, w, true);
                headerView.setUint32(12, h, true);
                const sliceLen = 16 + chunkFrames * frameStride;
                await postBinaryChunk(jobId, chunkBuf.subarray(0, sliceLen));
                chunkStartFrame += chunkFrames;
                chunkFrames = 0;
            }
        }

        if (!Number.isFinite(depthRangeLog.near) || !Number.isFinite(depthRangeLog.far)) {
            depthRangeLog.near = fixedDepthBounds.near;
            depthRangeLog.far = fixedDepthBounds.far;
        }

        const metadata = {
            export_mode: exportMode,
            projection: renderState.projection,
            resolution: [w, h],
            frame_count: totalFrames,
            motion_source: useClip ? "mocap_clip" : isSingle ? "single_frame" : "turntable",
            has_reconstructed_face: Boolean(activeMesh),
            has_head_shell: Boolean(reconstructedFace.shellRig),
            geometry_mode: reconstructedFace.guidesVisible ? "head_neck" : "face_only",
            angles_deg: frameAngles,
            elevation_deg: Number(renderState.elevation || 0),
            depth_mode: renderState.depthMode,
            depth_invert: Boolean(renderState.depthInvert),
            depth_range: [Number(depthRangeLog.near.toFixed(4)), Number(depthRangeLog.far.toFixed(4))],
            normal_flips: {
                x: Boolean(renderState.flipX),
                y: Boolean(renderState.flipY),
                z: Boolean(renderState.flipZ),
            },
            mask_mode: renderState.maskMode,
            pose_mode: renderState.poseMode || "head_face",
            pose_dot_scale: Number(renderState.poseDotScale ?? 1.0),
            pose_occlude: renderState.poseOcclude !== false,
        };

        const finishRes = await api.fetchApi(`/yedp_face3d/render_finish/${jobId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ metadata }),
        });
        if (!finishRes.ok) {
            throw new Error(`render_finish HTTP ${finishRes.status}`);
        }
    } finally {
        rtSharp.dispose();
        rtSmooth.dispose();
        renderer.setRenderTarget(prevTarget);
        renderer.setClearColor(prevClearColor, prevClearAlpha);
        if (grid) grid.visible = prevGridVis;
        reconstructedFace.applyRenderState(renderState);
        reconstructedFace.setViewMode(prevMode);
        reconstructedFace.group.position.copy(prevPos);
        reconstructedFace.group.rotation.copy(prevRot);
        reconstructedFace.group.quaternion.copy(prevQuat);
        if (prevLastMotion) {
            reconstructedFace.applyMotionFrame(prevLastMotion);
        } else if (reconstructedFace.hasMesh && reconstructedFace.restPositions) {
            reconstructedFace.uploadFaceFrame(reconstructedFace.restPositions);
        }
        if (placeholderHead?.group) {
            placeholderHead.group.visible = prevPhVis;
            if (prevPhRot) placeholderHead.group.rotation.copy(prevPhRot);
        }
    }
}
