import * as THREE from "../lib/three.module.js";

const N_RINGS = 8;
const N_SEGS = 24;
const THETA_MAX = 1.82; // ~104 deg dome extending slightly past equator so grazing side turns never expose edges

export const EYE_SPECS = [
    {
        side: "right", // Subject right (-X in canonical space)
        outerIdx: 33,
        innerIdx: 133,
        upperMidIdx: 159,
        lowerMidIdx: 145,
        pupilIdx: 468,
        irisRingIndices: [469, 470, 471, 472],
        rimIndices: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        upperLid: [246, 161, 160, 159, 158, 157, 173],
        upperFold: [27, 28, 29, 30, 56, 190, 247],
        lowerLid: [7, 163, 144, 145, 153, 154, 155],
    },
    {
        side: "left", // Subject left (+X in canonical space)
        outerIdx: 263,
        innerIdx: 362,
        upperMidIdx: 386,
        lowerMidIdx: 374,
        pupilIdx: 473,
        irisRingIndices: [474, 475, 476, 477],
        rimIndices: [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
        upperLid: [466, 388, 387, 386, 385, 384, 398],
        upperFold: [257, 258, 259, 260, 286, 414, 467],
        lowerLid: [249, 390, 373, 374, 380, 381, 382],
    },
];

function smoothstep(edge0, edge1, x) {
    const t = Math.max(0, Math.min(1, (x - edge0) / Math.max(edge1 - edge0, 1e-6)));
    return t * t * (3 - 2 * t);
}

function createDomeTopology() {
    const dirs = [0, 0, 1];
    const indices = [];

    for (let r = 1; r <= N_RINGS; r++) {
        const theta = (r / N_RINGS) * THETA_MAX;
        const sinT = Math.sin(theta);
        const cosT = Math.cos(theta);
        for (let s = 0; s < N_SEGS; s++) {
            const phi = (s / N_SEGS) * Math.PI * 2;
            dirs.push(sinT * Math.cos(phi), sinT * Math.sin(phi), cosT);
        }
    }

    const vertAt = (r, s) => 1 + (r - 1) * N_SEGS + (s % N_SEGS);

    // Center cap around pupil pole (+Z)
    for (let s = 0; s < N_SEGS; s++) {
        indices.push(0, vertAt(1, s), vertAt(1, s + 1));
    }
    // Concentric rings with strictly outward +Z CCW winding
    for (let r = 1; r < N_RINGS; r++) {
        for (let s = 0; s < N_SEGS; s++) {
            const a = vertAt(r, s);
            const b = vertAt(r, s + 1);
            const c = vertAt(r + 1, s);
            const d = vertAt(r + 1, s + 1);
            indices.push(a, c, d);
            indices.push(a, d, b);
        }
    }

    const vertCount = Math.floor(dirs.length / 3);

    return {
        unitDirs: new Float32Array(dirs),
        vertCount,
        indices,
    };
}

function measureRestIrisAndGaze(positions, spec, socket) {
    const hasIris = positions.length >= 478 * 3;
    const irisCenterIdx = hasIris ? spec.pupilIdx : spec.upperMidIdx;
    const irisCx = hasIris ? positions[irisCenterIdx * 3] : socket.cx;
    const irisCy = hasIris ? positions[irisCenterIdx * 3 + 1] : socket.cy;

    const restGx = Math.max(-0.35, Math.min(0.35, (irisCx - socket.cx) / Math.max(socket.rx, 1e-4)));
    const restGy = Math.max(-0.22, Math.min(0.22, (irisCy - socket.cy) / Math.max(socket.ry, 1e-4)));
    const baseYaw = Math.asin(restGx);
    const basePitch = -Math.asin(restGy);

    let irisRadPx = socket.eyeW * 0.22;
    if (hasIris) {
        let sumR = 0;
        for (const ri of spec.irisRingIndices) {
            sumR += Math.hypot(
                positions[ri * 3] - irisCx,
                positions[ri * 3 + 1] - irisCy
            );
        }
        irisRadPx = Math.max(sumR / spec.irisRingIndices.length, socket.eyeW * 0.16);
    }
    return { hasIris, irisCenterIdx, irisCx, irisCy, baseYaw, basePitch, irisRadPx };
}

/**
 * Fit an ellipsoidal eyeball socket strictly from post-sculpt, post-depth eyelid geometry
 * so the eyeball adapts proportionally when Face Depth or depth_weight changes.
 */
export function fitEyeSocket(positions, spec) {
    const outO = spec.outerIdx * 3;
    const inO = spec.innerIdx * 3;
    const upO = spec.upperMidIdx * 3;
    const loO = spec.lowerMidIdx * 3;

    const ox = positions[outO], oy = positions[outO + 1], oz = positions[outO + 2];
    const ix = positions[inO], iy = positions[inO + 1], iz = positions[inO + 2];
    const ux = positions[upO], uy = positions[upO + 1], uz = positions[upO + 2];
    const lx = positions[loO], ly = positions[loO + 1], lz = positions[loO + 2];

    const cx = 0.5 * (ox + ix);
    const cy = 0.5 * (oy + iy + 0.35 * ((uy + ly) - (oy + iy)));

    const eyeW = Math.max(Math.hypot(ix - ox, iy - oy, iz - oz), 0.03);
    const rx = 0.62 * eyeW;
    const ry = 0.60 * eyeW;

    const zMid = 0.5 * (uz + lz);
    const zCorner = 0.5 * (oz + iz);

    const sAt = (x, y) => {
        const u2 = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2;
        return Math.sqrt(Math.max(0.05, 1.0 - Math.min(0.95, u2)));
    };
    const sMid = 0.5 * (sAt(ux, uy) + sAt(lx, ly));
    const sCorner = 0.5 * (sAt(ox, oy) + sAt(ix, iy));

    // Derive depth radius from actual canthus-to-lid curvature; flat faces shrink rz automatically
    const dz = Math.max(0.0, zMid - zCorner);
    const rz = Math.max(0.10 * rx, Math.min(1.12 * rx, dz / Math.max(sMid - sCorner, 0.14)));

    const lidThickness = 0.032 * eyeW;
    const cz = (zMid - lidThickness) - rz * sMid;

    return {
        cx,
        cy,
        cz,
        rx,
        ry,
        rz,
        eyeW,
        lidThickness,
        cornerMidX: 0.5 * (ox + ix),
        cornerMidY: 0.5 * (oy + iy),
        cornerMidZ: 0.5 * (oz + iz),
    };
}

/**
 * Clamp all deformed eyelid & fold vertices outside the convex eyeball ellipsoid
 * so eye blinks and wide openings glide smoothly over the eyeball without clipping.
 */
export function clampEyelidsOverEyeball(outArray, socket, vertIndices, cx = socket.cx, cy = socket.cy, cz = socket.cz) {
    const { rx, ry, rz, lidThickness } = socket;
    for (let k = 0; k < vertIndices.length; k++) {
        const o = vertIndices[k] * 3;
        const dx = outArray[o] - cx;
        const dy = outArray[o + 1] - cy;
        const r2 = (dx / rx) ** 2 + (dy / ry) ** 2;
        if (r2 < 0.96) {
            const blend = 1.0 - smoothstep(0.80, 0.96, r2);
            const surfaceZ = cz + rz * Math.sqrt(1.0 - r2) + lidThickness * blend;
            if (outArray[o + 2] < surfaceZ) {
                outArray[o + 2] = surfaceZ;
            }
        }
    }
}

/**
 * Dual watertight, depth-adaptive 3D eyeball domes sharing the face ShaderMaterial
 * so Textured, Clay, Normals, Depth, and Mask passes all render solid eyeballs.
 */
export class EyeballRig {
    constructor(material) {
        this.group = new THREE.Group();
        this.group.visible = false;
        this.material = material;
        this.topo = createDomeTopology();

        this.eyes = EYE_SPECS.map((spec) => {
            const geo = new THREE.BufferGeometry();
            const pos = new Float32Array(this.topo.vertCount * 3);
            const norm = new Float32Array(this.topo.vertCount * 3);
            const uv = new Float32Array(this.topo.vertCount * 2);
            const mask = new Float32Array(this.topo.vertCount).fill(1.0);
            const scleraBlend = new Float32Array(this.topo.vertCount);

            geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
            geo.setAttribute("normal", new THREE.BufferAttribute(norm, 3));
            geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
            geo.setAttribute("aFaceMask", new THREE.BufferAttribute(mask, 1));
            geo.setAttribute("aScleraBlend", new THREE.BufferAttribute(scleraBlend, 1));
            geo.setIndex(this.topo.indices);

            const mesh = new THREE.Mesh(geo, material);
            mesh.frustumCulled = false;
            this.group.add(mesh);

            return {
                spec,
                geo,
                mesh,
                pos,
                norm,
                uv,
                scleraBlend,
                restSocket: null,
                baseYaw: 0,
                basePitch: 0,
                lastGazeX: 999,
                lastGazeY: 999,
                lastPosChecksum: NaN,
            };
        });
    }

    /**
     * Fit eyeball sockets, portrait-matched iris/sclera UVs, and normalized radial iris distances.
     */
    configureRest(restPositions, restUvs) {
        if (!restPositions || !restUvs || restPositions.length < 468 * 3) return;
        const unit = this.topo.unitDirs;
        const nVerts = this.topo.vertCount;

        for (const eye of this.eyes) {
            const spec = eye.spec;
            const socket = fitEyeSocket(restPositions, spec);
            eye.restSocket = socket;

            // Measure horizontal & vertical 2D basis + UV basis around the open eye
            const outO = spec.outerIdx * 3;
            const inO = spec.innerIdx * 3;
            const upO = spec.upperMidIdx * 3;
            const loO = spec.lowerMidIdx * 3;

            const hx = restPositions[inO] - restPositions[outO];
            const hy = restPositions[inO + 1] - restPositions[outO + 1];
            const vx = restPositions[upO] - restPositions[loO];
            const vy = restPositions[upO + 1] - restPositions[loO + 1];
            const det = hx * vy - vx * hy;

            const uOut = restUvs[spec.outerIdx * 2];
            const vOut = restUvs[spec.outerIdx * 2 + 1];
            const uIn = restUvs[spec.innerIdx * 2];
            const vIn = restUvs[spec.innerIdx * 2 + 1];
            const uUp = restUvs[spec.upperMidIdx * 2];
            const vUp = restUvs[spec.upperMidIdx * 2 + 1];
            const uLo = restUvs[spec.lowerMidIdx * 2];
            const vLo = restUvs[spec.lowerMidIdx * 2 + 1];

            const { hasIris, irisCenterIdx, baseYaw, basePitch, irisRadPx } = measureRestIrisAndGaze(
                restPositions,
                spec,
                socket
            );
            const irisU0 = hasIris ? restUvs[irisCenterIdx * 2] : 0.5 * (uOut + uIn);
            const irisV0 = hasIris ? restUvs[irisCenterIdx * 2 + 1] : 0.5 * (vUp + vLo);

            eye.baseYaw = baseYaw;
            eye.basePitch = basePitch;

            const hueX = uIn - uOut;
            const hueY = vIn - vOut;
            const vueX = uUp - uLo;
            const vueY = vUp - vLo;
            const invIrisRad = 1.0 / Math.max(irisRadPx, 1e-4);

            // Clamp sclera sample offsets cleanly inside open eye strip so rotated dome never samples eyelashes
            const toBasis = (dx, dy) => {
                if (Math.abs(det) < 1e-8) return [0, 0];
                const a = (dx * vy - vx * dy) / det;
                const b = (hx * dy - dx * hy) / det;
                return [a, b];
            };

            for (let i = 0; i < nVerts; i++) {
                const ux = unit[i * 3];
                const uy = unit[i * 3 + 1];
                const uz = unit[i * 3 + 2];

                const localX = ux * socket.rx;
                const localY = uy * socket.ry;
                const radial = Math.hypot(localX, localY);
                eye.scleraBlend[i] = radial * invIrisRad;

                // Inside iris radius use direct local coords; smoothly clamp outer sclera vertically into open eye strip
                const scleraBlend = smoothstep(irisRadPx * 0.85, irisRadPx * 1.45, radial);
                const maxScleraY = Math.max(Math.hypot(vx, vy) * 0.36, irisRadPx * 0.55);
                const clampedY = Math.max(-maxScleraY, Math.min(maxScleraY, localY));
                const maxScleraX = socket.eyeW * 0.38;
                const clampedX = Math.max(-maxScleraX, Math.min(maxScleraX, localX));

                const sampleDx = localX * (1.0 - scleraBlend) + clampedX * scleraBlend;
                const sampleDy = localY * (1.0 - scleraBlend) + clampedY * scleraBlend;
                const [alpha, beta] = toBasis(sampleDx, sampleDy);

                // Also softly push samples near rear hemisphere (uz < 0.2) toward sclera white corners
                const edgeAtten = uz < 0.25 ? (0.45 + 0.55 * Math.max(0, uz / 0.25)) : 1.0;
                const u = irisU0 + (alpha * hueX + beta * vueX) * edgeAtten;
                const v = irisV0 + (alpha * hueY + beta * vueY) * edgeAtten;

                eye.uv[i * 2] = Math.max(0, Math.min(1, u));
                eye.uv[i * 2 + 1] = Math.max(0, Math.min(1, v));
            }
            eye.geo.getAttribute("uv").needsUpdate = true;
            eye.geo.getAttribute("aScleraBlend").needsUpdate = true;

            eye.lastGazeX = 999;
            eye.lastGazeY = 999;
            eye.lastPosChecksum = NaN;
        }

        this.updateFromPositions(restPositions);
    }

    /**
     * Update eyeball socket placement & pupil-driven gaze rotation from the latest face frame.
     */
    updateFromPositions(positions) {
        if (!positions || positions.length < 468 * 3) return;
        const unit = this.topo.unitDirs;
        const nVerts = this.topo.vertCount;
        const hasIris = positions.length >= 478 * 3;

        for (const eye of this.eyes) {
            const socket = eye.restSocket;
            if (!socket) continue;
            const spec = eye.spec;

            // Track rigid corner shift so eyeball stays anchored in its orbital socket during cheek/smile lifts
            const outO = spec.outerIdx * 3;
            const inO = spec.innerIdx * 3;
            const shiftX = 0.5 * (positions[outO] + positions[inO]) - socket.cornerMidX;
            const shiftY = 0.5 * (positions[outO + 1] + positions[inO + 1]) - socket.cornerMidY;
            const shiftZ = 0.5 * (positions[outO + 2] + positions[inO + 2]) - socket.cornerMidZ;
            const cx = socket.cx + shiftX;
            const cy = socket.cy + shiftY;
            const cz = socket.cz + shiftZ;

            let gx = Math.sin(eye.baseYaw);
            let gy = -Math.sin(eye.basePitch);
            if (hasIris) {
                const px = positions[spec.pupilIdx * 3];
                const py = positions[spec.pupilIdx * 3 + 1];
                gx = Math.max(-0.35, Math.min(0.35, (px - cx) / Math.max(socket.rx, 1e-4)));
                gy = Math.max(-0.22, Math.min(0.22, (py - cy) / Math.max(socket.ry, 1e-4)));
            }

            const checksum = cx * 7.1 + cy * 11.3 + cz * 13.7 + socket.rx * 17.9 + socket.rz * 19.1;
            if (
                Math.abs(gx - eye.lastGazeX) < 1e-4 &&
                Math.abs(gy - eye.lastGazeY) < 1e-4 &&
                Math.abs(checksum - eye.lastPosChecksum) < 1e-5
            ) {
                continue;
            }
            eye.lastGazeX = gx;
            eye.lastGazeY = gy;
            eye.lastPosChecksum = checksum;

            const cosY = Math.sqrt(Math.max(0, 1 - gx * gx));
            const sinY = gx;
            const cosP = Math.sqrt(Math.max(0, 1 - gy * gy));
            const sinP = -gy;

            const invRx = 1.0 / Math.max(socket.rx, 1e-4);
            const invRy = 1.0 / Math.max(socket.ry, 1e-4);
            const invRz = 1.0 / Math.max(socket.rz, 1e-4);

            for (let i = 0; i < nVerts; i++) {
                const ux = unit[i * 3];
                const uy = unit[i * 3 + 1];
                const uz = unit[i * 3 + 2];

                // Apply pitch around X then yaw around Y to unit sphere direction
                const y1 = uy * cosP - uz * sinP;
                const z1 = uy * sinP + uz * cosP;
                const dx = ux * cosY + z1 * sinY;
                const dy = y1;
                const dz = -ux * sinY + z1 * cosY;

                const sx = dx * socket.rx;
                const sy = dy * socket.ry;
                const sz = dz * socket.rz;

                eye.pos[i * 3] = cx + sx;
                eye.pos[i * 3 + 1] = cy + sy;
                eye.pos[i * 3 + 2] = cz + sz;

                // Exact analytic ellipsoid outward normal
                const nx = dx * invRx;
                const ny = dy * invRy;
                const nz = dz * invRz;
                const len = Math.max(Math.hypot(nx, ny, nz), 1e-6);
                eye.norm[i * 3] = nx / len;
                eye.norm[i * 3 + 1] = ny / len;
                eye.norm[i * 3 + 2] = nz / len;
            }

            eye.geo.getAttribute("position").needsUpdate = true;
            eye.geo.getAttribute("normal").needsUpdate = true;
        }
    }

    clear() {
        this.group.visible = false;
        for (const eye of this.eyes) {
            eye.restSocket = null;
            eye.scleraBlend.fill(0);
            eye.geo.getAttribute("aScleraBlend").needsUpdate = true;
            eye.lastPosChecksum = NaN;
        }
    }

    dispose() {
        for (const eye of this.eyes) {
            eye.geo.dispose();
        }
    }
}
