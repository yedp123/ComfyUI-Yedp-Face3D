import * as THREE from "../lib/three.module.js";
import { api } from "../../../../scripts/api.js";

function buildOffscreenCamera(liveCamera, target, renderState, width, height, keepLivePose, orthoHalfH = null, renderFov = null) {
    const outAspect = width / Math.max(height, 1);
    const dist = Math.max(liveCamera.position.distanceTo(target), 0.3);
    const outFov = Number(renderFov ?? renderState.fov ?? 36);

    let offCam;
    if (renderState.projection === "ortho") {
        let halfH;
        if (liveCamera.isOrthographicCamera) {
            const baseHalfH = Number.isFinite(orthoHalfH) ? orthoHalfH : ((liveCamera.top - liveCamera.bottom) * 0.5);
            halfH = baseHalfH / Math.max(liveCamera.zoom, 1e-4);
        } else {
            halfH = dist * Math.tan(THREE.MathUtils.degToRad(outFov * 0.5));
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
        offCam = new THREE.PerspectiveCamera(outFov, outAspect, 0.05, 50.0);
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
    clipFps = 24,
    snapshotFrame = null,
    frameCount = 24,
    turntableFrames = null,
    renderState,
    orthoHalfH = null,
    renderFov = null,
    width,
    height,
    onProgress,
}) {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    const nPx = w * h;
    const frameStride = nPx * 14; // depth(1) + normal(3) + shaded(3) + textured(3) + mask(1) + pose(3)
    const framesPerChunk = Math.max(1, Math.min(4, Math.floor((12 * 1024 * 1024) / frameStride)));

    let motionMode = renderState.motionMode;
    let spinMode = renderState.spinMode;
    if (!motionMode || !spinMode) {
        const legacy = renderState.exportMode || "turntable";
        if (legacy === "clip") {
            motionMode = "clip";
            spinMode = "static";
        } else if (legacy === "single") {
            motionMode = snapshotFrame ? "snap" : "rest";
            spinMode = "single";
        } else {
            motionMode = snapshotFrame ? "snap" : "rest";
            spinMode = "turntable";
        }
    }

    const hasClip = Array.isArray(motionClip) && motionClip.length > 0 && reconstructedFace.hasMesh;
    if (motionMode === "clip" && !hasClip) {
        throw new Error(
            "Motion is set to 'Clip', but no webcam recording or video is loaded. Switch Motion to Rest/Snap or load a clip."
        );
    }
    const useClip = motionMode === "clip";
    const useSnap = motionMode === "snap" && Boolean(snapshotFrame);
    const isSingle = spinMode === "single";
    const isTurntable = spinMode === "turntable";
    const keepLivePose = !isTurntable;

    const reqFrames = Math.max(0, Math.floor(turntableFrames ?? frameCount ?? 24));
    const startIdx = useClip
        ? Math.max(0, Math.min(motionClip.length - 1, Math.floor((renderState.clipStart ?? 1) - 1)))
        : 0;
    const availClip = useClip ? Math.max(1, motionClip.length - startIdx) : 0;
    const totalFrames = isSingle
        ? 1
        : useClip
            ? (reqFrames > 0 ? Math.min(reqFrames, availClip) : availClip)
            : (reqFrames > 0 ? reqFrames : 1);

    const startAngle = Number(renderState.startAngle ?? -90);
    const endAngle = Number(renderState.endAngle ?? 90);
    const lockHeadPose = Boolean(renderState.lockHeadPose);
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
        const outFov = Number(renderFov ?? renderState.fov ?? 36);
        const offCam = buildOffscreenCamera(liveCamera, controlsTarget, renderState, w, h, keepLivePose, orthoHalfH, outFov);
        const activeMesh = reconstructedFace.hasMesh ? reconstructedFace : null;

        reconstructedFace.applyRenderState({ ...renderState, poseOverlay: false });

        // Precompute static pose & Fixed depth range ONCE before the loop (eliminates redundant per-angle skinning/subdiv)
        let fixedDepthBounds = { near: 0.5, far: 3.0 };
        if (activeMesh && !useClip && !isSingle) {
            if (useSnap) {
                reconstructedFace.applyMotionFrame(snapshotFrame);
            } else {
                reconstructedFace.resetPose();
            }
        } else if (activeMesh && isSingle) {
            if (useClip) {
                reconstructedFace.applyMotionFrame(motionClip[startIdx]);
            } else if (useSnap) {
                reconstructedFace.applyMotionFrame(snapshotFrame);
            }
        }
        if (activeMesh && !dynamicDepth) {
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
            const yawDeg = isTurntable
                ? startAngle + (endAngle - startAngle) * t
                : isSingle
                    ? THREE.MathUtils.radToDeg(activeMesh ? prevRot.y : (prevPhRot?.y || 0))
                    : 0;
            frameAngles.push(Number(yawDeg.toFixed(2)));

            if (activeMesh) {
                if (useClip && !isSingle) {
                    const mf = motionClip[startIdx + f];
                    reconstructedFace.applyMotionFrame(mf);
                    if (isTurntable) {
                        reconstructedFace.setTurntableYaw(
                            THREE.MathUtils.degToRad(yawDeg),
                            lockHeadPose ? null : mf?.quat
                        );
                    }
                } else if (isTurntable) {
                    const baseQuat = (!lockHeadPose && useSnap) ? snapshotFrame?.quat : null;
                    reconstructedFace.setTurntableYaw(THREE.MathUtils.degToRad(yawDeg), baseQuat);
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

            if (totalFrames > 1 && f + 1 < totalFrames) {
                // Yield between WebGL readPixels stalls so Windows DWM and browser UI stay responsive
                await new Promise((resolve) => setTimeout(resolve, 0));
            }
        }

        if (!Number.isFinite(depthRangeLog.near) || !Number.isFinite(depthRangeLog.far)) {
            depthRangeLog.near = fixedDepthBounds.near;
            depthRangeLog.far = fixedDepthBounds.far;
        }

        const motionSource = `${motionMode}_${spinMode}`;

        const metadata = {
            motion_mode: motionMode,
            spin_mode: spinMode,
            clip_start_frame: useClip ? startIdx + 1 : null,
            lock_head_pose: lockHeadPose,
            projection: renderState.projection,
            fov_deg: outFov,
            fps: Number(clipFps || 24),
            resolution: [w, h],
            frame_count: totalFrames,
            motion_source: motionSource,
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
        if (prevLastMotion) {
            reconstructedFace.applyMotionFrame(prevLastMotion);
        } else if (reconstructedFace.hasMesh && reconstructedFace.restPositions) {
            reconstructedFace.uploadFaceFrame(reconstructedFace.restPositions);
        }
        reconstructedFace.group.position.copy(prevPos);
        reconstructedFace.group.rotation.copy(prevRot);
        reconstructedFace.group.quaternion.copy(prevQuat);
        if (placeholderHead?.group) {
            placeholderHead.group.visible = prevPhVis;
            if (prevPhRot) placeholderHead.group.rotation.copy(prevPhRot);
        }
    }
}
