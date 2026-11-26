import * as THREE from "../lib/three.module.js";

const ARCH_SEGMENTS = 84;
const MAX_JAW_OPEN_RAD = THREE.MathUtils.degToRad(26); // ~0.4538 rad

// Exact ordered 20-vertex MediaPipe inner lip loop (clockwise from +Z camera: 78 -> 13 -> 308 -> 14 -> 78)
export const INNER_LIP_RING = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
];

// Symmetric half-arch tooth intervals in normalized arc-length s in [0, 1] (7 teeth/side = 14 per arch)
const UPPER_TOOTH_SPANS = [
    [0.000, 0.145, 1.04], // Central incisor (widest front tooth)
    [0.145, 0.265, 0.90], // Lateral incisor
    [0.265, 0.405, 1.06], // Canine (corner cusp)
    [0.405, 0.545, 0.90], // 1st premolar
    [0.545, 0.685, 0.86], // 2nd premolar
    [0.685, 0.845, 0.82], // 1st molar
    [0.845, 1.000, 0.78], // 2nd molar
];

const LOWER_TOOTH_SPANS = [
    [0.000, 0.115, 1.00], // Central incisor (anatomically narrower)
    [0.115, 0.230, 0.94], // Lateral incisor
    [0.230, 0.370, 1.06], // Canine
    [0.370, 0.515, 0.92], // 1st premolar
    [0.515, 0.665, 0.88], // 2nd premolar
    [0.665, 0.830, 0.84], // 1st molar
    [0.830, 1.000, 0.80], // 2nd molar
];

const TONGUE_SEGS_U = 16;
const TONGUE_SEGS_V = 10;

const TEETH_VERTEX_SHADER = /* glsl */ `
    varying vec3 vColor;
    varying vec3 vWorldNormal;
    varying vec3 vViewNormal;
    varying float vViewDepth;

    void main() {
        vColor = color;
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        vViewNormal = normalize(normalMatrix * normal);

        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vViewDepth = -mvPosition.z;
        gl_Position = projectionMatrix * mvPosition;
    }
`;

const TEETH_FRAGMENT_SHADER = /* glsl */ `
    uniform vec3 uBaseColor;
    uniform float uClayScale;
    uniform int uViewMode;
    uniform float uDepthNear;
    uniform float uDepthFar;
    uniform float uDepthInvert;
    uniform vec3 uNormalFlip;

    uniform vec3 uKeyDir;
    uniform vec3 uKeyColor;
    uniform float uKeyIntensity;
    uniform vec3 uRimDir;
    uniform vec3 uRimColor;
    uniform float uRimIntensity;
    uniform float uAmbientIntensity;

    uniform vec3 uClayKeyDir;
    uniform float uClayKeyIntensity;
    uniform float uClayFillIntensity;

    varying vec3 vColor;
    varying vec3 vWorldNormal;
    varying vec3 vViewNormal;
    varying float vViewDepth;

    void main() {
        if (uViewMode == 1) {
            vec3 nView = normalize(vViewNormal) * uNormalFlip;
            gl_FragColor = vec4(nView * 0.5 + 0.5, 1.0);
            return;
        }

        if (uViewMode == 2) {
            float span = max(uDepthFar - uDepthNear, 0.001);
            float d = 1.0 - clamp((vViewDepth - uDepthNear) / span, 0.0, 1.0);
            if (uDepthInvert > 0.5) d = 1.0 - d;
            gl_FragColor = vec4(vec3(mix(0.05, 1.0, d)), 1.0);
            return;
        }

        if (uViewMode == 3) {
            vec3 nClay = normalize(vWorldNormal);
            float clayDiff = max(dot(nClay, normalize(uClayKeyDir)), 0.0);
            float clayHemi = 0.5 + 0.5 * nClay.y;
            vec3 clayColor = vec3(0.66 * uClayScale) * (uClayFillIntensity * (0.7 + 0.3 * clayHemi) + clayDiff * uClayKeyIntensity);
            gl_FragColor = vec4(clamp(clayColor, 0.0, 1.0), 1.0);
            #include <colorspace_fragment>
            return;
        }

        vec3 albedo = uBaseColor * vColor;
        vec3 n = normalize(vWorldNormal);
        float keyDiff = max(dot(n, normalize(uKeyDir)), 0.0);
        float rimDiff = max(dot(n, normalize(uRimDir)), 0.0);
        float hemi = 0.5 + 0.5 * n.y;

        vec3 lighting = vec3(uAmbientIntensity * (0.75 + 0.25 * hemi))
            + uKeyColor * (keyDiff * uKeyIntensity)
            + uRimColor * (rimDiff * uRimIntensity);

        gl_FragColor = vec4(albedo * lighting, 1.0);
        #include <colorspace_fragment>
    }
`;

export function createTeethPassMaterial({ baseColor, clayScale = 1.0, side = THREE.FrontSide }) {
    return new THREE.ShaderMaterial({
        vertexShader: TEETH_VERTEX_SHADER,
        fragmentShader: TEETH_FRAGMENT_SHADER,
        vertexColors: true,
        side,
        blending: THREE.NoBlending,
        uniforms: {
            uBaseColor: { value: new THREE.Color(baseColor).convertSRGBToLinear() },
            uClayScale: { value: clayScale },
            uViewMode: { value: 3 },
            uDepthNear: { value: 1.4 },
            uDepthFar: { value: 2.3 },
            uDepthInvert: { value: 0.0 },
            uNormalFlip: { value: new THREE.Vector3(1, 1, 1) },
            uKeyDir: { value: new THREE.Vector3(1.8, 1.6, 2.4).normalize() },
            uKeyColor: { value: new THREE.Color(0xfff5ea) },
            uKeyIntensity: { value: 0.72 },
            uRimDir: { value: new THREE.Vector3(-2.0, 1.1, -1.8).normalize() },
            uRimColor: { value: new THREE.Color(0x7dd3fc) },
            uRimIntensity: { value: 0.35 },
            uAmbientIntensity: { value: 0.55 },
            uClayKeyDir: { value: new THREE.Vector3(0.45, 0.58, 0.68).normalize() },
            uClayKeyIntensity: { value: 0.85 },
            uClayFillIntensity: { value: 0.35 },
        },
    });
}

function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
}

function smoothstep(edge0, edge1, x) {
    const t = clamp((x - edge0) / Math.max(edge1 - edge0, 1e-6), 0, 1);
    return t * t * (3 - 2 * t);
}

function evalToothProfile(sAbs, isUpper) {
    const spans = isUpper ? UPPER_TOOTH_SPANS : LOWER_TOOTH_SPANS;
    const s = Math.min(0.9999, Math.abs(sAbs));
    let spanIdx = spans.length - 1;
    for (let i = 0; i < spans.length; i++) {
        if (s >= spans[i][0] && s <= spans[i][1]) {
            spanIdx = i;
            break;
        }
    }
    const [a, b, heightMult] = spans[spanIdx];
    const t = clamp((s - a) / Math.max(b - a, 1e-5), 0, 1);
    const bulge = Math.sin(Math.PI * t);
    const groove = 1.0 - Math.pow(bulge, 0.65);

    // Subtle canine point vs flatter incisal edge
    const canineTip = spanIdx === 2 ? 0.06 * Math.pow(bulge, 2.2) : 0.0;
    const edgeScallop = -0.08 * Math.pow(groove, 1.7) + canineTip;

    // Gentle cervical gum papilla hugging each tooth root without jagged gaps
    const cervicalWave = -0.08 * bulge + 0.05 * groove;

    return {
        bulge,
        groove,
        heightMult,
        edgeScallop,
        cervicalWave,
    };
}

/**
 * Sample (x, z, nx, nz, sNorm) along the rounded U-arch curve
 * z(u) = -archDepth * (0.45 * u^2 + 0.55 * u^4) with true 3D arc-length parameterization.
 */
function sampleArchCurve(halfW, archDepth) {
    const cols = ARCH_SEGMENTS + 1;
    const mid = ARCH_SEGMENTS >> 1;
    const xs = new Float32Array(cols);
    const zs = new Float32Array(cols);
    const nxs = new Float32Array(cols);
    const nzs = new Float32Array(cols);
    const sNorm = new Float32Array(cols);

    for (let c = 0; c < cols; c++) {
        const u = -1 + (2 * c) / ARCH_SEGMENTS;
        const u2 = u * u;
        const u4 = u2 * u2;
        xs[c] = u * halfW;
        zs[c] = -archDepth * (0.45 * u2 + 0.55 * u4);

        const dxdu = halfW;
        const dzdu = -archDepth * (0.90 * u + 2.20 * u * u2);
        const len = Math.max(Math.hypot(dxdu, dzdu), 1e-6);
        nxs[c] = -dzdu / len;
        nzs[c] = dxdu / len;
    }

    const halfArc = new Float32Array(mid + 1);
    for (let i = 1; i <= mid; i++) {
        const dx = xs[mid + i] - xs[mid + i - 1];
        const dz = zs[mid + i] - zs[mid + i - 1];
        halfArc[i] = halfArc[i - 1] + Math.hypot(dx, dz);
    }
    const totalHalf = Math.max(halfArc[mid], 1e-6);
    for (let c = 0; c < cols; c++) {
        const distIdx = Math.abs(c - mid);
        sNorm[c] = halfArc[distIdx] / totalHalf;
    }

    return { xs, zs, nxs, nzs, sNorm };
}

/**
 * Allocate a closed-manifold strip BufferGeometry with `ringSize` vertices per column
 * across `ARCH_SEGMENTS + 1` columns plus left/right rear endcaps.
 */
function createStripArchGeometry(ringSize) {
    const cols = ARCH_SEGMENTS + 1;
    const vertCount = cols * ringSize + ringSize * 2;
    const positions = new Float32Array(vertCount * 3);
    const colors = new Float32Array(vertCount * 3).fill(1.0);
    const indices = [];

    for (let c = 0; c < ARCH_SEGMENTS; c++) {
        const a = c * ringSize;
        const b = (c + 1) * ringSize;
        for (let k = 0; k < ringSize; k++) {
            const kNext = (k + 1) % ringSize;
            // Outward-facing CCW winding (+Z on front labial face at u=0)
            indices.push(a + k, b + kNext, b + k);
            indices.push(a + k, a + kNext, b + kNext);
        }
    }

    const capStart = cols * ringSize;
    const capEnd = capStart + ringSize;
    for (let k = 1; k + 1 < ringSize; k++) {
        indices.push(capStart, capStart + k, capStart + k + 1);
        indices.push(capEnd, capEnd + k + 1, capEnd + k);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.setIndex(indices);
    return geo;
}

function createTongueGeometry() {
    const cols = TONGUE_SEGS_U + 1;
    const rows = TONGUE_SEGS_V + 1;
    const vertCount = cols * rows;
    const positions = new Float32Array(vertCount * 3);
    const colors = new Float32Array(vertCount * 3).fill(1.0);
    const indices = [];

    for (let j = 0; j < TONGUE_SEGS_V; j++) {
        for (let i = 0; i < TONGUE_SEGS_U; i++) {
            const a = j * cols + i;
            const b = j * cols + (i + 1);
            const c = (j + 1) * cols + i;
            const d = (j + 1) * cols + (i + 1);
            // Outward upward (+Y) CCW winding: (+X) x (-Z) = +Y
            indices.push(a, b, c);
            indices.push(b, d, c);
        }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.setIndex(indices);
    return geo;
}

function updateTongueGeometry(geo, halfW, lengthZ, heightY, floorDropY) {
    const cols = TONGUE_SEGS_U + 1;
    const rows = TONGUE_SEGS_V + 1;
    const posAttr = geo.getAttribute("position");
    const colAttr = geo.getAttribute("color");
    const arr = posAttr.array;
    const col = colAttr.array;

    for (let j = 0; j < rows; j++) {
        const v = j / TONGUE_SEGS_V; // 0 = front tip, 1 = rear base
        const widthProfile = 0.42 + 0.58 * Math.sin(Math.min(1.0, v * 1.15) * Math.PI * 0.5);
        const vHeight = 0.22 + 0.78 * Math.sin(Math.pow(v, 0.72) * Math.PI * 0.85);
        const rearAo = 1.0 - 0.55 * smoothstep(0.25, 1.0, v);

        for (let i = 0; i < cols; i++) {
            const u = -1 + (2 * i) / TONGUE_SEGS_U;
            const u2 = u * u;
            const x = u * halfW * widthProfile;
            const z = -v * lengthZ - 0.14 * lengthZ * (1.0 - v) * u2;

            const arch = Math.sqrt(Math.max(0, 1.0 - u2));
            const sulcus = 1.0 - 0.18 * Math.exp(-10.0 * u2) * Math.sin(Math.PI * v);
            const y = -floorDropY + heightY * arch * sulcus * vHeight;

            const idx = (j * cols + i) * 3;
            arr[idx] = x;
            arr[idx + 1] = y;
            arr[idx + 2] = z;

            const edgeAo = 0.72 + 0.28 * arch;
            const sulcusAo = 1.0 - 0.14 * Math.exp(-10.0 * u2) * Math.sin(Math.PI * v);
            const shade = rearAo * edgeAo * sulcusAo;
            col[idx] = shade;
            col[idx + 1] = shade;
            col[idx + 2] = shade;
        }
    }

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    geo.computeVertexNormals();
    geo.computeBoundingSphere();
}

function updateSculptedTeethGeometry(geo, halfW, crownH, archDepth, thickness, isUpper) {
    const ringSize = 6; // 0: outer-root, 1: outer-mid, 2: outer-bevel, 3: outer-occlusal, 4: inner-occlusal, 5: inner-root
    const posAttr = geo.getAttribute("position");
    const colAttr = geo.getAttribute("color");
    const arr = posAttr.array;
    const col = colAttr.array;
    const cols = ARCH_SEGMENTS + 1;
    const dirY = isUpper ? 1.0 : -1.0;
    const curve = sampleArchCurve(halfW, archDepth);

    for (let c = 0; c < cols; c++) {
        const x = curve.xs[c];
        const z = curve.zs[c];
        const nx = curve.nxs[c];
        const nz = curve.nzs[c];
        const s = curve.sNorm[c];

        const prof = evalToothProfile(s, isUpper);
        const labialOut = 0.24 * thickness * Math.pow(prof.bulge, 0.65) - 0.08 * thickness * prof.groove;
        const hEff = crownH * prof.heightMult;
        const yRoot = dirY * hEff * (1.02 + prof.cervicalWave);
        const yMid = dirY * hEff * 0.54;
        const yBevel = dirY * hEff * (0.14 - 0.5 * prof.edgeScallop);
        const yEdge = dirY * hEff * (-prof.edgeScallop);

        const ox = x + nx * labialOut;
        const oz = z + nz * labialOut;
        const mx = x + nx * (labialOut + 0.14 * thickness * prof.bulge);
        const mz = z + nz * (labialOut + 0.14 * thickness * prof.bulge);
        const bx = x + nx * (labialOut * 0.65);
        const bz = z + nz * (labialOut * 0.65);
        const ix = x - nx * thickness;
        const iz = z - nz * thickness;

        const base = c * ringSize * 3;
        const setPt = (k, px, py, pz, shade) => {
            const o = base + k * 3;
            arr[o] = px;
            arr[o + 1] = py;
            arr[o + 2] = pz;
            col[o] = shade;
            col[o + 1] = shade;
            col[o + 2] = shade;
        };

        const rearAo = 1.0 - 0.40 * smoothstep(0.35, 1.0, s);
        const grooveAo = 1.0 - 0.16 * prof.groove;
        const rootAo = rearAo * grooveAo * 0.86;
        const midAo = rearAo * grooveAo;
        const edgeAo = rearAo * (1.0 - 0.08 * prof.groove);

        if (isUpper) {
            setPt(0, ox, yRoot, oz, rootAo);
            setPt(1, mx, yMid, mz, midAo);
            setPt(2, bx, yBevel, bz, edgeAo);
            setPt(3, x, yEdge, z, edgeAo);
            setPt(4, ix, yEdge + 0.08 * hEff, iz, edgeAo * 0.82);
            setPt(5, ix, yRoot, iz, rootAo * 0.72);
        } else {
            // Mirror order to preserve outward CCW winding for -Y root direction
            setPt(0, ix, yRoot, iz, rootAo * 0.72);
            setPt(1, ix, yEdge - 0.08 * hEff, iz, edgeAo * 0.82);
            setPt(2, x, yEdge, z, edgeAo);
            setPt(3, bx, yBevel, bz, edgeAo);
            setPt(4, mx, yMid, mz, midAo);
            setPt(5, ox, yRoot, oz, rootAo);
        }
    }

    const capStartOff = cols * ringSize * 3;
    const lastColOff = ARCH_SEGMENTS * ringSize * 3;
    for (let i = 0; i < ringSize * 3; i++) {
        arr[capStartOff + i] = arr[i];
        col[capStartOff + i] = col[i];
        arr[capStartOff + ringSize * 3 + i] = arr[lastColOff + i];
        col[capStartOff + ringSize * 3 + i] = col[lastColOff + i];
    }

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    geo.computeVertexNormals();
    geo.computeBoundingSphere();
}

function updateGumCollarGeometry(geo, halfW, crownH, gumH, archDepth, thickness, isUpper) {
    const ringSize = 4; // 0: outer-base, 1: outer-cervical, 2: inner-cervical, 3: inner-base
    const posAttr = geo.getAttribute("position");
    const colAttr = geo.getAttribute("color");
    const arr = posAttr.array;
    const col = colAttr.array;
    const cols = ARCH_SEGMENTS + 1;
    const dirY = isUpper ? 1.0 : -1.0;
    const curve = sampleArchCurve(halfW, archDepth);

    for (let c = 0; c < cols; c++) {
        const x = curve.xs[c];
        const z = curve.zs[c];
        const nx = curve.nxs[c];
        const nz = curve.nzs[c];
        const s = curve.sNorm[c];

        const prof = evalToothProfile(s, isUpper);
        const hEff = crownH * prof.heightMult;
        // Overlap slightly past the tooth cervical line so no gap appears between gums and roots
        const yCerv = dirY * hEff * (0.90 + prof.cervicalWave);
        const yBase = dirY * (crownH + gumH);

        const gumOut = 0.26 * thickness;
        const ox = x + nx * gumOut;
        const oz = z + nz * gumOut;
        const ix = x - nx * (thickness * 1.15);
        const iz = z - nz * (thickness * 1.15);

        const base = c * ringSize * 3;
        const setPt = (k, px, py, pz, shade) => {
            const o = base + k * 3;
            arr[o] = px;
            arr[o + 1] = py;
            arr[o + 2] = pz;
            col[o] = shade;
            col[o + 1] = shade;
            col[o + 2] = shade;
        };

        const rearAo = 1.0 - 0.45 * smoothstep(0.30, 1.0, s);
        if (isUpper) {
            setPt(0, ox, yBase, oz, rearAo * 0.78);
            setPt(1, ox, yCerv, oz, rearAo * 0.96);
            setPt(2, ix, yCerv, iz, rearAo * 0.75);
            setPt(3, ix, yBase, iz, rearAo * 0.65);
        } else {
            setPt(0, ix, yBase, iz, rearAo * 0.65);
            setPt(1, ix, yCerv, iz, rearAo * 0.75);
            setPt(2, ox, yCerv, oz, rearAo * 0.96);
            setPt(3, ox, yBase, oz, rearAo * 0.78);
        }
    }

    const capStartOff = cols * ringSize * 3;
    const lastColOff = ARCH_SEGMENTS * ringSize * 3;
    for (let i = 0; i < ringSize * 3; i++) {
        arr[capStartOff + i] = arr[i];
        col[capStartOff + i] = col[i];
        arr[capStartOff + ringSize * 3 + i] = arr[lastColOff + i];
        col[capStartOff + ringSize * 3 + i] = col[lastColOff + i];
    }

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    geo.computeVertexNormals();
    geo.computeBoundingSphere();
}

/**
 * Build a closed inward-facing oral cavity pouch stitched to the 20-vertex INNER_LIP_RING:
 *   Ring 0 (0..19):  live lip seal
 *   Ring 1 (20..39): vestibule pocket surrounding dental arches
 *   Ring 2 (40..59): deep oral cavity behind molars
 *   Vertex 60:       throat center cap
 */
function createMouthBagGeometry() {
    const nRing = INNER_LIP_RING.length;
    const vertCount = nRing * 3 + 1;
    const positions = new Float32Array(vertCount * 3);
    const colors = new Float32Array(vertCount * 3).fill(1.0);
    const indices = [];

    for (let r = 0; r < 2; r++) {
        const rowA = r * nRing;
        const rowB = (r + 1) * nRing;
        for (let k = 0; k < nRing; k++) {
            const kNext = (k + 1) % nRing;
            const a = rowA + k;
            const b = rowA + kNext;
            const c = rowB + k;
            const d = rowB + kNext;
            // Inward-facing winding for clockwise INNER_LIP_RING viewed from +Z
            indices.push(a, c, b);
            indices.push(b, c, d);
        }
    }

    const capIdx = nRing * 3;
    const row2 = 2 * nRing;
    for (let k = 0; k < nRing; k++) {
        const kNext = (k + 1) % nRing;
        indices.push(capIdx, row2 + kNext, row2 + k);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.setIndex(indices);
    return geo;
}

/**
 * Self-contained prototype rig for 3D upper/lower sculpted dental arches, gums,
 * dark inward-facing oral cavity pouch, resting tongue, and anatomical TMJ jaw kinematics.
 */
export class TeethRig {
    constructor() {
        this.group = new THREE.Group();
        this.group.visible = false;

        this.upperTeethGeo = createStripArchGeometry(6);
        this.lowerTeethGeo = createStripArchGeometry(6);
        this.upperGumGeo = createStripArchGeometry(4);
        this.lowerGumGeo = createStripArchGeometry(4);
        this.mouthBagGeo = createMouthBagGeometry();
        this.tongueGeo = createTongueGeometry();

        this.enamelMat = createTeethPassMaterial({
            baseColor: 0xe8e2d5,
            clayScale: 1.0,
            side: THREE.FrontSide,
        });
        this.gumMat = createTeethPassMaterial({
            baseColor: 0xb96b6b,
            clayScale: 0.85,
            side: THREE.FrontSide,
        });
        this.mouthBagMat = createTeethPassMaterial({
            baseColor: 0x160a0b,
            clayScale: 0.22,
            side: THREE.FrontSide,
        });
        this.tongueMat = createTeethPassMaterial({
            baseColor: 0x8f474b,
            clayScale: 0.55,
            side: THREE.FrontSide,
        });

        this.upperArch = new THREE.Group();
        this.upperTeethMesh = new THREE.Mesh(this.upperTeethGeo, this.enamelMat);
        this.upperGumMesh = new THREE.Mesh(this.upperGumGeo, this.gumMat);
        this.upperTeethMesh.frustumCulled = false;
        this.upperGumMesh.frustumCulled = false;
        this.upperArch.add(this.upperTeethMesh, this.upperGumMesh);

        this.jawPivot = new THREE.Group();
        this.lowerArch = new THREE.Group();
        this.lowerTeethMesh = new THREE.Mesh(this.lowerTeethGeo, this.enamelMat);
        this.lowerGumMesh = new THREE.Mesh(this.lowerGumGeo, this.gumMat);
        this.lowerTeethMesh.frustumCulled = false;
        this.lowerGumMesh.frustumCulled = false;
        this.lowerArch.add(this.lowerTeethMesh, this.lowerGumMesh);
        this.tongueMesh = new THREE.Mesh(this.tongueGeo, this.tongueMat);
        this.tongueMesh.frustumCulled = false;
        this.jawPivot.add(this.lowerArch, this.tongueMesh);

        this.mouthBagMesh = new THREE.Mesh(this.mouthBagGeo, this.mouthBagMat);
        this.mouthBagMesh.frustumCulled = false;

        // Wireframe-only TMJ condyle markers + axis line
        this.tmjGroup = new THREE.Group();
        this.tmjGroup.visible = false;
        this.tmjGeo = new THREE.SphereGeometry(1, 12, 10);
        this.tmjMat = new THREE.MeshBasicMaterial({ color: 0xfacc15 });
        this.tmjLeftMarker = new THREE.Mesh(this.tmjGeo, this.tmjMat);
        this.tmjRightMarker = new THREE.Mesh(this.tmjGeo, this.tmjMat);
        this.tmjLeftMarker.frustumCulled = false;
        this.tmjRightMarker.frustumCulled = false;

        this.tmjAxisGeo = new THREE.BufferGeometry();
        this.tmjAxisGeo.setAttribute(
            "position",
            new THREE.BufferAttribute(new Float32Array([-1, 0, 0, 1, 0, 0]), 3)
        );
        this.tmjAxisMat = new THREE.LineBasicMaterial({
            color: 0xfacc15,
            transparent: true,
            opacity: 0.75,
        });
        this.tmjAxisLine = new THREE.Line(this.tmjAxisGeo, this.tmjAxisMat);
        this.tmjAxisLine.frustumCulled = false;

        this.tmjGroup.add(this.tmjLeftMarker, this.tmjRightMarker, this.tmjAxisLine);
        this.jawPivot.add(this.tmjGroup);

        this.group.add(this.mouthBagMesh, this.upperArch, this.jawPivot);

        this.rest = null;
        this.restPositionsRef = null;
        this.viewMode = "clay";
        this._lastFitChecksum = NaN;
        this._lastLoggedChecksum = NaN;

        this.settings = {
            testTeeth: false,
            testJawOpen: 0.0,
            teethUpperZ: 0.0,
            teethLowerZ: 0.0,
            teethOffsetY: 0.0,
            teethWidthScale: 1.0,
        };
    }

    setViewMode(mode) {
        this.viewMode = mode || "clay";
        const isWire = this.viewMode === "wireframe";
        this.tmjGroup.visible = isWire;
        this.enamelMat.wireframe = isWire;
        this.gumMat.wireframe = isWire;
        this.tongueMat.wireframe = isWire;
    }

    syncFromFaceMaterial(faceMaterial) {
        const src = faceMaterial?.uniforms;
        if (!src) return;
        const mats = [this.enamelMat, this.gumMat, this.mouthBagMat, this.tongueMat];
        for (let i = 0; i < mats.length; i++) {
            const dst = mats[i].uniforms;
            dst.uViewMode.value = src.uViewMode.value;
            dst.uDepthNear.value = src.uDepthNear.value;
            dst.uDepthFar.value = src.uDepthFar.value;
            dst.uDepthInvert.value = src.uDepthInvert.value;
            dst.uNormalFlip.value.copy(src.uNormalFlip.value);
            dst.uKeyDir.value.copy(src.uKeyDir.value);
            dst.uKeyColor.value.copy(src.uKeyColor.value);
            dst.uKeyIntensity.value = src.uKeyIntensity.value;
            dst.uRimDir.value.copy(src.uRimDir.value);
            dst.uRimColor.value.copy(src.uRimColor.value);
            dst.uRimIntensity.value = src.uRimIntensity.value;
            dst.uAmbientIntensity.value = src.uAmbientIntensity.value;
            dst.uClayKeyDir.value.copy(src.uClayKeyDir.value);
            dst.uClayKeyIntensity.value = src.uClayKeyIntensity.value;
            dst.uClayFillIntensity.value = src.uClayFillIntensity.value;
        }
    }

    _computeFitChecksum(pos) {
        const ids = [61, 291, 78, 308, 13, 14, 152, 18, 17, 234, 454];
        let sum = 0;
        for (let i = 0; i < ids.length; i++) {
            const o = ids[i] * 3;
            sum += pos[o] * (i + 1.3) + pos[o + 1] * (i + 2.7) + pos[o + 2] * (i + 4.1);
        }
        return sum;
    }

    fitToRestMesh(restPositions) {
        if (!restPositions || restPositions.length < 455 * 3) return false;
        this.restPositionsRef = restPositions;
        const checksum = this._computeFitChecksum(restPositions);
        if (Math.abs(checksum - this._lastFitChecksum) < 1e-7 && this.rest) {
            return false;
        }
        this._lastFitChecksum = checksum;

        const pt = (idx) => ({
            x: restPositions[idx * 3],
            y: restPositions[idx * 3 + 1],
            z: restPositions[idx * 3 + 2],
        });

        const p61 = pt(61);
        const p291 = pt(291);
        const p78 = pt(78);
        const p308 = pt(308);
        const p13 = pt(13);
        const p14 = pt(14);
        const p152 = pt(152);
        const p18 = pt(18);
        const p17 = pt(17);
        const p234 = pt(234);
        const p454 = pt(454);

        const mouthWidth = Math.max(Math.hypot(p291.x - p61.x, p291.y - p61.y), 0.04);
        const xMid = 0.5 * (p61.x + p291.x);
        const yOcclusal = 0.5 * (p13.y + p14.y);
        const zLipFront = Math.min(p13.z, p14.z);
        const zCorner = 0.5 * (p78.z + p308.z);

        const yEar = 0.5 * (p234.y + p454.y);
        const zEar = 0.5 * (p234.z + p454.z);
        const condyleHalfSpanX = Math.max(0.42 * Math.abs(p454.x - p234.x), 0.05);

        const oralSpanZ = Math.max(0.08 * mouthWidth, zLipFront - zEar);
        const cornerDropZ = Math.max(0.0, zLipFront - zCorner);

        const halfW = 0.45 * mouthWidth;
        const upperCrownH = 0.195 * mouthWidth;
        const lowerCrownH = 0.165 * mouthWidth;
        const gumH = 0.13 * mouthWidth;
        const thickness = 0.055 * mouthWidth;
        const upperRestZ = zLipFront - Math.min(0.024 * mouthWidth, 0.14 * oralSpanZ);
        const lowerRestZ = upperRestZ - Math.min(0.015 * mouthWidth, 0.08 * oralSpanZ);

        const pivotX = xMid;
        const pivotY = yEar - 0.12 * (yEar - yOcclusal);
        const pivotZ = zEar + 0.18 * (zLipFront - zEar);

        const rawDepth = Math.min(0.76 * oralSpanZ, cornerDropZ + 0.30 * mouthWidth);
        const maxSafeDepth = Math.max(0.10 * mouthWidth, (upperRestZ - pivotZ) * 0.82);
        const archDepth = Math.min(rawDepth, maxSafeDepth);

        this.rest = {
            mouthWidth,
            halfW,
            upperCrownH,
            lowerCrownH,
            gumH,
            thickness,
            archDepth,
            xMid,
            yOcclusal,
            upperRestZ,
            lowerRestZ,
            pivotX,
            pivotY,
            pivotZ,
            condyleHalfSpanX,
            oralSpanZ,
            mandibleRest: [
                p152.x, p152.y, p152.z,
                p18.x, p18.y, p18.z,
                p17.x, p17.y, p17.z,
            ],
        };

        updateSculptedTeethGeometry(this.upperTeethGeo, halfW, upperCrownH, archDepth, thickness, true);
        updateGumCollarGeometry(this.upperGumGeo, halfW, upperCrownH, gumH, archDepth, thickness, true);

        updateSculptedTeethGeometry(this.lowerTeethGeo, halfW * 0.94, lowerCrownH, archDepth * 0.94, thickness, false);
        updateGumCollarGeometry(this.lowerGumGeo, halfW * 0.94, lowerCrownH, gumH, archDepth * 0.94, thickness, false);

        const tongueHalfW = halfW * 0.94 * 0.78;
        const tongueLength = Math.max(archDepth * 1.05, 0.35 * mouthWidth);
        const tongueHeight = 0.16 * mouthWidth;
        const tongueFloorDrop = lowerCrownH * 0.95;
        updateTongueGeometry(this.tongueGeo, tongueHalfW, tongueLength, tongueHeight, tongueFloorDrop);

        const markerR = Math.max(0.004, 0.045 * mouthWidth);
        this.tmjLeftMarker.scale.setScalar(markerR);
        this.tmjRightMarker.scale.setScalar(markerR);
        this.tmjLeftMarker.position.set(-condyleHalfSpanX, 0, 0);
        this.tmjRightMarker.position.set(condyleHalfSpanX, 0, 0);

        const axisArr = this.tmjAxisGeo.getAttribute("position").array;
        axisArr[0] = -condyleHalfSpanX;
        axisArr[3] = condyleHalfSpanX;
        this.tmjAxisGeo.getAttribute("position").needsUpdate = true;

        this._applyStaticPlacement();

        if (this.group.visible && Math.abs(this._lastLoggedChecksum - checksum) >= 1e-7) {
            this._lastLoggedChecksum = checksum;
            console.log(
                `[YedpFace3D][TeethProto] Fitted | mouthW=${mouthWidth.toFixed(4)} | tmjPivot=(${pivotX.toFixed(4)}, ${pivotY.toFixed(4)}, ${pivotZ.toFixed(4)}) | upperRestZ=${upperRestZ.toFixed(4)}`
            );
        }
        return true;
    }

    _applyStaticPlacement() {
        if (!this.rest) return;
        const r = this.rest;
        const s = this.settings;
        const widthScale = clamp(Number(s.teethWidthScale ?? 1.0), 0.7, 1.6);
        const offsetY = Number(s.teethOffsetY || 0) * r.mouthWidth;
        const upperZ = r.upperRestZ + Number(s.teethUpperZ || 0) * r.mouthWidth;
        const lowerZ = r.lowerRestZ + Number(s.teethLowerZ || 0) * r.mouthWidth;

        this.upperArch.position.set(r.xMid, r.yOcclusal + offsetY, upperZ);
        this.upperArch.scale.set(widthScale, 1.0, 1.0);

        this.lowerArch.position.set(
            r.xMid - r.pivotX,
            (r.yOcclusal + offsetY) - r.pivotY,
            lowerZ - r.pivotZ
        );
        this.lowerArch.scale.set(widthScale, 1.0, 1.0);

        this.tongueBasePos = [
            r.xMid - r.pivotX,
            (r.yOcclusal + offsetY) - r.pivotY,
            (lowerZ - 0.12 * r.mouthWidth) - r.pivotZ,
        ];
        this.tongueMesh.position.set(this.tongueBasePos[0], this.tongueBasePos[1], this.tongueBasePos[2]);
        this.tongueMesh.scale.set(widthScale, 1.0, 1.0);
    }

    setSettings(settings) {
        if (!settings) return;
        this.settings.testTeeth = Boolean(settings.testTeeth);
        this.settings.testJawOpen = Number(settings.testJawOpen ?? 0.0);
        this.settings.teethUpperZ = Number(settings.teethUpperZ ?? 0.0);
        this.settings.teethLowerZ = Number(settings.teethLowerZ ?? 0.0);
        this.settings.teethOffsetY = Number(settings.teethOffsetY ?? 0.0);
        this.settings.teethWidthScale = Number(settings.teethWidthScale ?? 1.0);
        this._applyStaticPlacement();

        if (
            this.group.visible
            && this.rest
            && Math.abs(this._lastLoggedChecksum - this._lastFitChecksum) >= 1e-7
        ) {
            this._lastLoggedChecksum = this._lastFitChecksum;
            const r = this.rest;
            console.log(
                `[YedpFace3D][TeethProto] Fitted | mouthW=${r.mouthWidth.toFixed(4)} | tmjPivot=(${r.pivotX.toFixed(4)}, ${r.pivotY.toFixed(4)}, ${r.pivotZ.toFixed(4)}) | upperRestZ=${r.upperRestZ.toFixed(4)}`
            );
        }
    }

    _updateMouthBag(srcPos, open01) {
        if (!srcPos || srcPos.length < 416 * 3 || !this.rest) return;
        const r = this.rest;
        const arr = this.mouthBagGeo.getAttribute("position").array;
        const nRing = INNER_LIP_RING.length;

        const widthScale = clamp(Number(this.settings.teethWidthScale ?? 1.0), 0.7, 1.6);
        const offsetY = Number(this.settings.teethOffsetY || 0) * r.mouthWidth;

        // Ring 0: exact live inner lip positions pulled slightly inward (-Z) to avoid z-fighting
        let cx = 0;
        let cy = 0;
        for (let k = 0; k < nRing; k++) {
            const idx = INNER_LIP_RING[k] * 3;
            const lx = srcPos[idx];
            const ly = srcPos[idx + 1];
            const lz = srcPos[idx + 2] - 0.002;
            arr[k * 3] = lx;
            arr[k * 3 + 1] = ly;
            arr[k * 3 + 2] = lz;
            cx += lx;
            cy += ly;
        }
        cx /= nRing;
        cy /= nRing;

        // Pocket envelope must enclose both dental arches & gums even during wide jaw openings
        const pocketHalfW = Math.max(r.halfW * widthScale * 1.15, 0.48 * r.mouthWidth);
        const topY = r.yOcclusal + offsetY + r.upperCrownH + r.gumH * 1.08;
        const jawDropY = Math.sin(open01 * MAX_JAW_OPEN_RAD) * Math.max(r.lowerRestZ - r.pivotZ, 0.05);
        const botY = r.yOcclusal + offsetY - r.lowerCrownH - r.gumH * 1.08 - jawDropY;
        const midY = 0.5 * (topY + botY);
        const pocketHalfH = Math.max(0.5 * (topY - botY), 0.22 * r.mouthWidth);

        const zVestibule = r.upperRestZ - Math.max(0.45 * r.oralSpanZ, 0.55 * r.archDepth);
        const zDeep = r.upperRestZ - Math.max(0.84 * r.oralSpanZ, 1.15 * r.archDepth);
        const zCap = r.upperRestZ - Math.max(0.95 * r.oralSpanZ, 1.28 * r.archDepth);

        for (let k = 0; k < nRing; k++) {
            const ang = (k / nRing) * Math.PI * 2;
            // Start at left corner (k=0 -> theta=PI) going clockwise via top (PI/2) -> right (0) -> bottom (-PI/2)
            const theta = Math.PI - ang;
            const cosT = Math.cos(theta);
            const sinT = Math.sin(theta);

            // Ring 1 (vestibule pocket surrounding arches)
            const r1 = (nRing + k) * 3;
            arr[r1] = r.xMid + cosT * pocketHalfW;
            arr[r1 + 1] = midY + sinT * pocketHalfH;
            arr[r1 + 2] = zVestibule;

            // Ring 2 (deep cavity behind molars)
            const r2 = (2 * nRing + k) * 3;
            arr[r2] = r.xMid + cosT * (pocketHalfW * 0.92);
            arr[r2 + 1] = midY + sinT * (pocketHalfH * 1.06);
            arr[r2 + 2] = zDeep;
        }

        // Throat center cap
        const capOff = 3 * nRing * 3;
        arr[capOff] = r.xMid;
        arr[capOff + 1] = midY;
        arr[capOff + 2] = zCap;

        this.mouthBagGeo.getAttribute("position").needsUpdate = true;
        this.mouthBagGeo.computeVertexNormals();
        this.mouthBagGeo.computeBoundingSphere();
    }

    updatePose(activePositions = null, motionFrame = null) {
        if (!this.rest) return;
        const r = this.rest;
        const bs = motionFrame?.bs || null;

        let liveOpenRad = 0;
        let latX = 0;
        let protZ = 0;

        if (activePositions && activePositions.length >= 153 * 3) {
            const m = r.mandibleRest;
            const dx = 0.40 * (activePositions[152 * 3] - m[0])
                + 0.35 * (activePositions[18 * 3] - m[3])
                + 0.25 * (activePositions[17 * 3] - m[6]);
            const dy = 0.40 * (activePositions[152 * 3 + 1] - m[1])
                + 0.35 * (activePositions[18 * 3 + 1] - m[4])
                + 0.25 * (activePositions[17 * 3 + 1] - m[7]);
            const dz = 0.40 * (activePositions[152 * 3 + 2] - m[2])
                + 0.35 * (activePositions[18 * 3 + 2] - m[5])
                + 0.25 * (activePositions[17 * 3 + 2] - m[8]);

            const jawDrop = Math.max(0, -dy);
            const leverZ = Math.max(r.lowerRestZ - r.pivotZ, 0.04);
            liveOpenRad = Math.atan2(jawDrop, leverZ);

            latX = clamp(dx, -0.14 * r.mouthWidth, 0.14 * r.mouthWidth);
            protZ = clamp(Math.max(0, dz), 0, 0.08 * r.mouthWidth);
        } else if (bs) {
            const bsOpen = clamp(Number(bs.jawOpen || 0), 0, 1);
            liveOpenRad = bsOpen * MAX_JAW_OPEN_RAD;
            const jawR = clamp(Number(bs.jawRight || 0), 0, 1);
            const jawL = clamp(Number(bs.jawLeft || 0), 0, 1);
            latX = clamp((jawR - jawL) * 0.12 * r.mouthWidth, -0.14 * r.mouthWidth, 0.14 * r.mouthWidth);
        }

        const manualOpenRad = clamp(this.settings.testJawOpen, 0, 1) * MAX_JAW_OPEN_RAD;
        const openRad = Math.min(0.52, Math.max(liveOpenRad, manualOpenRad));
        const open01 = openRad / MAX_JAW_OPEN_RAD;

        const glideT = smoothstep(0.22, 1.0, open01);
        const glideZ = glideT * Math.min(0.07 * r.mouthWidth, 0.18 * r.oralSpanZ);
        const glideY = -glideT * 0.025 * r.mouthWidth;

        const bsForwardZ = bs ? clamp(Number(bs.jawForward || 0), 0, 1) * 0.06 * r.mouthWidth : 0;

        this.jawPivot.position.set(
            r.pivotX + latX,
            r.pivotY + glideY,
            r.pivotZ + glideZ + protZ + bsForwardZ
        );
        this.jawPivot.rotation.set(
            openRad,
            -(latX / Math.max(r.condyleHalfSpanX, 0.04)) * 0.12,
            0
        );

        const bagSource = (activePositions && activePositions.length >= 416 * 3)
            ? activePositions
            : this.restPositionsRef;
        this._updateMouthBag(bagSource, open01);

        if (this.tongueBasePos) {
            const tongueOut = bs ? clamp(Number(bs.tongueOut || 0), 0, 1) : 0;
            const tReact = tongueOut > 0.05 ? tongueOut : 0;
            this.tongueMesh.position.set(
                this.tongueBasePos[0],
                this.tongueBasePos[1] + tReact * 0.03 * r.mouthWidth,
                this.tongueBasePos[2] + tReact * 0.14 * r.mouthWidth
            );
        }
    }

    dispose() {
        this.upperTeethGeo.dispose();
        this.lowerTeethGeo.dispose();
        this.upperGumGeo.dispose();
        this.lowerGumGeo.dispose();
        this.mouthBagGeo.dispose();
        this.tongueGeo.dispose();
        this.tmjGeo.dispose();
        this.tmjAxisGeo.dispose();
        this.enamelMat.dispose();
        this.gumMat.dispose();
        this.mouthBagMat.dispose();
        this.tongueMat.dispose();
        this.tmjMat.dispose();
        this.tmjAxisMat.dispose();
    }
}
