/**
 * 1-Level Phong Tessellation (curved triangle subdivision) modifier.
 * Precomputes the shared edge-midpoint topology once per reconstruction so
 * 852 triangles subdivide smoothly into 3,408 curved triangles at 60 FPS
 * with zero per-frame allocations.
 */
export class MeshSubdivider {
    constructor(nBaseVerts, baseIndices, baseUvs, smoothness = 0.65) {
        this.nBaseVerts = nBaseVerts;
        this.baseIndices = baseIndices;
        this.smoothness = smoothness;

        const edgeMap = new Map();
        const edgePairs = []; // [a, b, midVertIdx, ...]
        let nextVert = nBaseVerts;

        const getEdgeMid = (a, b) => {
            const lo = a < b ? a : b;
            const hi = a < b ? b : a;
            const key = `${lo}_${hi}`;
            let mid = edgeMap.get(key);
            if (mid === undefined) {
                mid = nextVert++;
                edgeMap.set(key, mid);
                edgePairs.push(lo, hi, mid);
            }
            return mid;
        };

        const subdivIndices = [];
        for (let i = 0; i + 2 < baseIndices.length; i += 3) {
            const a = baseIndices[i];
            const b = baseIndices[i + 1];
            const c = baseIndices[i + 2];
            const mab = getEdgeMid(a, b);
            const mbc = getEdgeMid(b, c);
            const mca = getEdgeMid(c, a);

            subdivIndices.push(a, mab, mca);
            subdivIndices.push(mab, b, mbc);
            subdivIndices.push(mca, mbc, c);
            subdivIndices.push(mab, mbc, mca);
        }

        this.totalVerts = nextVert;
        this.edgePairs = new Int32Array(edgePairs);
        this.subdivIndices = subdivIndices;

        // Precompute interpolated UVs and reusable scratch buffers
        this.subdivUvs = new Float32Array(this.totalVerts * 2);
        this.subdivUvs.set(baseUvs);
        for (let e = 0; e < this.edgePairs.length; e += 3) {
            const a = this.edgePairs[e];
            const b = this.edgePairs[e + 1];
            const m = this.edgePairs[e + 2];
            this.subdivUvs[m * 2] = 0.5 * (baseUvs[a * 2] + baseUvs[b * 2]);
            this.subdivUvs[m * 2 + 1] = 0.5 * (baseUvs[a * 2 + 1] + baseUvs[b * 2 + 1]);
        }

        this.baseNormals = new Float32Array(nBaseVerts * 3);
        this.subdivPositions = new Float32Array(this.totalVerts * 3);
    }

    /**
     * Compute curved Phong midpoint positions into `this.subdivPositions` and return it.
     */
    evaluate(basePositions) {
        const normals = this.baseNormals;
        normals.fill(0);

        const idx = this.baseIndices;
        for (let i = 0; i + 2 < idx.length; i += 3) {
            const ia = idx[i] * 3;
            const ib = idx[i + 1] * 3;
            const ic = idx[i + 2] * 3;

            const ax = basePositions[ia];
            const ay = basePositions[ia + 1];
            const az = basePositions[ia + 2];

            const ux = basePositions[ib] - ax;
            const uy = basePositions[ib + 1] - ay;
            const uz = basePositions[ib + 2] - az;

            const vx = basePositions[ic] - ax;
            const vy = basePositions[ic + 1] - ay;
            const vz = basePositions[ic + 2] - az;

            const nx = uy * vz - uz * vy;
            const ny = uz * vx - ux * vz;
            const nz = ux * vy - uy * vx;

            normals[ia] += nx; normals[ia + 1] += ny; normals[ia + 2] += nz;
            normals[ib] += nx; normals[ib + 1] += ny; normals[ib + 2] += nz;
            normals[ic] += nx; normals[ic + 1] += ny; normals[ic + 2] += nz;
        }

        for (let i = 0; i < normals.length; i += 3) {
            const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]);
            if (len > 1e-6) {
                const inv = 1.0 / len;
                normals[i] *= inv;
                normals[i + 1] *= inv;
                normals[i + 2] *= inv;
            } else {
                normals[i + 2] = 1.0;
            }
        }

        const out = this.subdivPositions;
        out.set(basePositions);

        const s = this.smoothness;
        const edges = this.edgePairs;
        for (let e = 0; e < edges.length; e += 3) {
            const ia = edges[e] * 3;
            const ib = edges[e + 1] * 3;
            const im = edges[e + 2] * 3;

            const pax = basePositions[ia];
            const pay = basePositions[ia + 1];
            const paz = basePositions[ia + 2];

            const pbx = basePositions[ib];
            const pby = basePositions[ib + 1];
            const pbz = basePositions[ib + 2];

            const mx = 0.5 * (pax + pbx);
            const my = 0.5 * (pay + pby);
            const mz = 0.5 * (paz + pbz);

            const nax = normals[ia];
            const nay = normals[ia + 1];
            const naz = normals[ia + 2];

            const nbx = normals[ib];
            const nby = normals[ib + 1];
            const nbz = normals[ib + 2];

            const dotA = (mx - pax) * nax + (my - pay) * nay + (mz - paz) * naz;
            const dotB = (mx - pbx) * nbx + (my - pby) * nby + (mz - pbz) * nbz;

            // Blend flat midpoint with average tangent-plane projection
            const offsetX = -0.5 * (dotA * nax + dotB * nbx);
            const offsetY = -0.5 * (dotA * nay + dotB * nby);
            const offsetZ = -0.5 * (dotA * naz + dotB * nbz);

            out[im] = mx + s * offsetX;
            out[im + 1] = my + s * offsetY;
            out[im + 2] = mz + s * offsetZ;
        }

        return out;
    }
}
