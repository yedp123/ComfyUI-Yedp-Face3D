/**
 * Graph Laplacian diffused regional push/pull sculpting rig + global Face Depth
 * operating on the reconstructed base 478-vertex rest mesh before retargeting.
 */

const LEFT_TEMPLE_IDX = 234;
const RIGHT_TEMPLE_IDX = 454;

// #5 belongs to nose bridge; keep Nose Tip seeded purely on center apex/columella (#4/#1/#19/#94)
// so Graph Laplacian diffusion grades smoothly across wings and nostrils without ridge seams
const NOSE_BRIDGE_SEEDS = [168, 6, 197, 195, 5];
const NOSE_TIP_SEEDS = [4, 1, 19, 94];
const LIPS_SEEDS = [0, 11, 12, 13, 14, 15, 16, 17, 37, 267, 39, 269, 40, 270, 61, 291, 84, 314, 181, 405, 91, 321];
const CHIN_SEEDS = [152, 175, 199, 200, 148, 377, 176, 400, 149, 378];
const CHEEK_SEEDS = [205, 425, 50, 280, 123, 352, 187, 411, 101, 330];
const JAW_WIDTH_SEEDS = [58, 172, 136, 150, 132, 288, 397, 365, 379, 361];

function buildAdjacency(nVerts, indices) {
    const adj = Array.from({ length: nVerts }, () => new Set());
    if (indices && indices.length >= 3) {
        for (let i = 0; i + 2 < indices.length; i += 3) {
            const a = indices[i];
            const b = indices[i + 1];
            const c = indices[i + 2];
            if (a < nVerts && b < nVerts && c < nVerts) {
                adj[a].add(b).add(c);
                adj[b].add(a).add(c);
                adj[c].add(a).add(b);
            }
        }
    }
    return adj.map((s) => Array.from(s));
}

function buildDiffusedMask(nVerts, neighbors, seeds, iterations = 4) {
    const mask = new Float32Array(nVerts);
    const nxt = new Float32Array(nVerts);
    const seedSet = new Set(seeds);
    for (const s of seeds) {
        if (s < nVerts) mask[s] = 1.0;
    }

    for (let iter = 0; iter < iterations; iter++) {
        for (let i = 0; i < nVerts; i++) {
            if (seedSet.has(i)) {
                nxt[i] = 1.0;
                continue;
            }
            const nbrs = neighbors[i];
            if (!nbrs.length) {
                nxt[i] = mask[i];
                continue;
            }
            let sum = 0;
            for (let k = 0; k < nbrs.length; k++) sum += mask[nbrs[k]];
            nxt[i] = 0.35 * mask[i] + 0.65 * (sum / nbrs.length);
        }
        mask.set(nxt);
    }

    for (let i = 0; i < nVerts; i++) {
        const nbrs = neighbors[i];
        if (!nbrs.length) {
            nxt[i] = mask[i];
            continue;
        }
        let sum = 0;
        for (let k = 0; k < nbrs.length; k++) sum += mask[nbrs[k]];
        nxt[i] = 0.72 * mask[i] + 0.28 * (sum / nbrs.length);
    }
    return nxt;
}

export class SculptRig {
    constructor(basePositions, indices) {
        this.base = new Float32Array(basePositions);
        this.n = Math.floor(basePositions.length / 3);
        const neighbors = buildAdjacency(this.n, indices);

        // Anchor global Face Depth scaling to the ear/temple plane so temples stay aligned with ear guides
        const lz = this.n > LEFT_TEMPLE_IDX ? this.base[LEFT_TEMPLE_IDX * 3 + 2] : 0;
        const rz = this.n > RIGHT_TEMPLE_IDX ? this.base[RIGHT_TEMPLE_IDX * 3 + 2] : 0;
        this.earPlaneZ = 0.5 * (lz + rz);

        this.masks = {
            noseBridge: buildDiffusedMask(this.n, neighbors, NOSE_BRIDGE_SEEDS, 4),
            noseTip: buildDiffusedMask(this.n, neighbors, NOSE_TIP_SEEDS, 6),
            lipsDepth: buildDiffusedMask(this.n, neighbors, LIPS_SEEDS, 3),
            chin: buildDiffusedMask(this.n, neighbors, CHIN_SEEDS, 4),
            cheeks: buildDiffusedMask(this.n, neighbors, CHEEK_SEEDS, 5),
            jawWidth: buildDiffusedMask(this.n, neighbors, JAW_WIDTH_SEEDS, 4),
        };
    }

    /**
     * Return a newly sculpted rest-position Float32Array from sculpt slider values.
     */
    evaluate(sculptState) {
        const out = new Float32Array(this.base);
        if (!sculptState) return out;

        const faceDepth = Number(sculptState.faceDepth ?? 1.0);
        const nb = Number(sculptState.noseBridge || 0) * 0.045;
        const nt = Number(sculptState.noseTip || 0) * 0.055;
        const lp = Number(sculptState.lipsDepth || 0) * 0.045;
        const ch = Number(sculptState.chin || 0) * 0.055;
        const ck = Number(sculptState.cheeks || 0) * 0.045;
        const jw = Number(sculptState.jawWidth || 0) * 0.055;

        const m = this.masks;
        for (let i = 0; i < this.n; i++) {
            const o = i * 3;
            const bx = this.base[o];
            const bz = this.base[o + 2];
            const xSign = bx >= 0 ? 1.0 : -1.0;

            // Global Face Depth scales Z projection cleanly relative to the ear/temple plane
            const scaledZ = this.earPlaneZ + (bz - this.earPlaneZ) * faceDepth;

            const dz =
                nb * m.noseBridge[i] +
                nt * m.noseTip[i] +
                lp * m.lipsDepth[i] +
                ch * m.chin[i] +
                ck * m.cheeks[i];

            const dx =
                (jw * m.jawWidth[i] + ck * 0.35 * m.cheeks[i]) * xSign;

            // Gentle downward-forward pull on Nose Tip so tip projects naturally instead of hooking upward
            const dy =
                -nt * 0.14 * m.noseTip[i] -
                ch * 0.18 * m.chin[i];

            out[o] = bx + dx;
            out[o + 1] += dy;
            out[o + 2] = scaledZ + dz;
        }
        return out;
    }
}
