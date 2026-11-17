// Ordered 36-vertex MediaPipe FACE_OVAL boundary loop (#10 top forehead -> -X temple #234 -> #152 chin -> +X temple #454)
export const FACE_OVAL_LOOP = [
    10, 109, 67, 103, 54, 21, 162, 127, 234,
    93, 132, 58, 172, 136, 150, 149, 176, 148,
    152, 377, 400, 378, 379, 365, 397, 288, 361,
    323, 454, 356, 389, 251, 284, 332, 297, 338,
];

function findBoundaryLoops(indices) {
    const edgeCounts = new Map();
    for (let i = 0; i + 2 < indices.length; i += 3) {
        const tri = [indices[i], indices[i + 1], indices[i + 2]];
        for (let k = 0; k < 3; k++) {
            const a = tri[k];
            const b = tri[(k + 1) % 3];
            const key = a < b ? `${a}_${b}` : `${b}_${a}`;
            edgeCounts.set(key, (edgeCounts.get(key) || 0) + 1);
        }
    }

    const adj = new Map();
    for (const [key, cnt] of edgeCounts.entries()) {
        if (cnt !== 1) continue;
        const [aStr, bStr] = key.split("_");
        const a = Number(aStr);
        const b = Number(bStr);
        if (!adj.has(a)) adj.set(a, []);
        if (!adj.has(b)) adj.set(b, []);
        adj.get(a).push(b);
        adj.get(b).push(a);
    }

    const visited = new Set();
    const loops = [];
    for (const start of adj.keys()) {
        if (visited.has(start)) continue;
        const loop = [];
        let cur = start;
        let prev = -1;
        while (cur !== undefined && !visited.has(cur)) {
            visited.add(cur);
            loop.push(cur);
            const nbrs = adj.get(cur) || [];
            const next = nbrs.find((v) => v !== prev && !visited.has(v));
            prev = cur;
            cur = next;
        }
        loops.push(loop);
    }
    return loops;
}

function buildAdjacency(nVerts, indices) {
    const adj = Array.from({ length: nVerts }, () => new Set());
    for (let i = 0; i + 2 < indices.length; i += 3) {
        const a = indices[i];
        const b = indices[i + 1];
        const c = indices[i + 2];
        adj[a].add(b).add(c);
        adj[b].add(a).add(c);
        adj[c].add(a).add(b);
    }
    return adj.map((s) => Array.from(s));
}

function measureSocketFrame(positions, loopIds) {
    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (const id of loopIds) {
        cx += positions[id * 3];
        cy += positions[id * 3 + 1];
        cz += positions[id * 3 + 2];
    }
    const inv = 1.0 / loopIds.length;
    cx *= inv;
    cy *= inv;
    cz *= inv;

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;

    for (const id of loopIds) {
        const x = positions[id * 3];
        const y = positions[id * 3 + 1];
        const z = positions[id * 3 + 2];
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (z < minZ) minZ = z;
        if (z > maxZ) maxZ = z;
    }

    return {
        cx,
        cy,
        cz,
        spanX: Math.max(maxX - minX, 1e-4),
        spanY: Math.max(maxY - minY, 1e-4),
        spanZ: Math.max(maxZ - minZ, 1e-4),
    };
}

/**
 * Aligns, welds, and harmonically deforms the 36-vertex-socketed cranium+ears+neck shell
 * onto any 478-vertex reconstructed MediaPipe face mesh.
 */
export class HeadShellRig {
    constructor(shellPos, shellIndices) {
        this.rawShellPos = new Float32Array(shellPos);
        this.nShellVerts = Math.floor(shellPos.length / 3);
        const loops = findBoundaryLoops(shellIndices);

        const rawFaceLoop = loops.find((l) => l.length === FACE_OVAL_LOOP.length);
        if (!rawFaceLoop) {
            throw new Error(`Expected 36-vertex face socket loop in head_shell.glb (found ${loops.map((l) => l.length).join(", ")})`);
        }
        const neckLoop = loops.find((l) => l !== rawFaceLoop) || [];

        // Order socket loop starting at max-Y forehead vertex (#10) and progressing toward -X temple (#234)
        let topPos = 0;
        for (let i = 1; i < rawFaceLoop.length; i++) {
            if (shellPos[rawFaceLoop[i] * 3 + 1] > shellPos[rawFaceLoop[topPos] * 3 + 1]) {
                topPos = i;
            }
        }
        let ordered = [...rawFaceLoop.slice(topPos), ...rawFaceLoop.slice(0, topPos)];
        if (ordered.length > 8 && shellPos[ordered[8] * 3] > 0) {
            ordered = [ordered[0], ...ordered.slice(1).reverse()];
        }
        this.shellSocketLoop = ordered;
        this.shellSocketFrame = measureSocketFrame(this.rawShellPos, this.shellSocketLoop);

        // Map socket vertices onto MediaPipe vertices and assign contiguous interior indices [478..]
        const socketToMediaPipe = new Map();
        for (let k = 0; k < FACE_OVAL_LOOP.length; k++) {
            socketToMediaPipe.set(this.shellSocketLoop[k], FACE_OVAL_LOOP[k]);
        }

        const shellToMerged = new Int32Array(this.nShellVerts);
        const interiorShellIds = [];
        for (let i = 0; i < this.nShellVerts; i++) {
            const mpIdx = socketToMediaPipe.get(i);
            if (mpIdx !== undefined) {
                shellToMerged[i] = mpIdx;
            } else {
                shellToMerged[i] = 478 + interiorShellIds.length;
                interiorShellIds.push(i);
            }
        }
        this.shellToMerged = shellToMerged;
        this.interiorShellIds = interiorShellIds;
        this.nInterior = interiorShellIds.length;
        this.neckLoopMerged = neckLoop
            .map((sid) => shellToMerged[sid])
            .filter((mid) => mid >= 478);

        // Identify left (+X) and right (-X) ear clusters in baked shell coords for OpenPose anchors
        const earThresholdX = this.shellSocketFrame.spanX * 0.55;
        this.leftEarMerged = [];
        this.rightEarMerged = [];
        for (const sid of interiorShellIds) {
            const dx = this.rawShellPos[sid * 3] - this.shellSocketFrame.cx;
            if (dx >= earThresholdX) this.leftEarMerged.push(shellToMerged[sid]);
            else if (dx <= -earThresholdX) this.rightEarMerged.push(shellToMerged[sid]);
        }

        // Remap shell triangles using merged vertex indices so the 36-vertex seam is shared
        this.mergedShellIndices = new Array(shellIndices.length);
        for (let i = 0; i < shellIndices.length; i++) {
            this.mergedShellIndices[i] = shellToMerged[shellIndices[i]];
        }

        // Precompute ring-1 nearest socket vertex so border triangles sample skin edge UV safely
        const shellAdj = buildAdjacency(this.nShellVerts, shellIndices);
        this.shellAdj = shellAdj;
        this.nearestSocketMp = new Int32Array(this.nInterior).fill(-1);
        for (let k = 0; k < this.nInterior; k++) {
            const sid = interiorShellIds[k];
            for (const nb of shellAdj[sid]) {
                const mp = socketToMediaPipe.get(nb);
                if (mp !== undefined) {
                    this.nearestSocketMp[k] = mp;
                    break;
                }
            }
        }

        // Precompute discrete Laplace harmonic coordinates (nInterior x 36) from socket (1) to neck base (0)
        const isSocket = new Int8Array(this.nShellVerts).fill(-1);
        for (let k = 0; k < this.shellSocketLoop.length; k++) {
            isSocket[this.shellSocketLoop[k]] = k;
        }
        const isNeckBase = new Uint8Array(this.nShellVerts);
        for (const nid of neckLoop) {
            if (isSocket[nid] < 0) isNeckBase[nid] = 1;
        }

        const basis = new Float32Array(this.nShellVerts * 36);
        const tmp = new Float32Array(this.nShellVerts);
        for (let k = 0; k < 36; k++) {
            tmp.fill(0);
            tmp[this.shellSocketLoop[k]] = 1.0;
            for (let iter = 0; iter < 24; iter++) {
                for (let v = 0; v < this.nShellVerts; v++) {
                    if (isSocket[v] >= 0) {
                        tmp[v] = isSocket[v] === k ? 1.0 : 0.0;
                        continue;
                    }
                    if (isNeckBase[v]) {
                        tmp[v] = 0.0;
                        continue;
                    }
                    const nbrs = shellAdj[v];
                    if (!nbrs.length) continue;
                    let sum = 0;
                    for (let j = 0; j < nbrs.length; j++) sum += tmp[nbrs[j]];
                    tmp[v] = sum / nbrs.length;
                }
            }
            for (let v = 0; v < this.nShellVerts; v++) {
                basis[v * 36 + k] = tmp[v];
            }
        }

        // Pack compact immutable (nInterior x 36) harmonic basis matrix shared safely across instances
        this.harmonicWeights = new Float32Array(this.nInterior * 36);
        for (let idx = 0; idx < this.nInterior; idx++) {
            const sid = interiorShellIds[idx];
            this.harmonicWeights.set(basis.subarray(sid * 36, (sid + 1) * 36), idx * 36);
        }
    }

    /**
     * Weld shell geometry onto a reconstructed 478-vertex face mesh, affine-fitting skull
     * proportions and diffusing any residual 36-vertex seam offset across interior vertices.
     * Returns immutable mesh arrays plus per-instance `fitState` for zero-conflict runtime updates.
     */
    buildMergedRestMesh(facePositions, faceIndices, faceUvs) {
        const totalVerts = 478 + this.nInterior;
        const faceCount = Math.floor(facePositions.length / 3);
        const faceFrame = measureSocketFrame(facePositions, FACE_OVAL_LOOP);
        const sf = this.shellSocketFrame;

        const sx = faceFrame.spanX / sf.spanX;
        const sy = faceFrame.spanY / sf.spanY;
        const sz = 0.5 * (sx + sy);

        const fittedInteriorBase = new Float32Array(this.nInterior * 3);
        const refSocketPos = new Float32Array(36 * 3);
        const deltaScratch = new Float32Array(36 * 3);

        // 1. Compute affine-aligned shell vertex positions across all 816 shell vertices
        const alignedAll = new Float32Array(this.nShellVerts * 3);
        for (let i = 0; i < this.nShellVerts; i++) {
            alignedAll[i * 3] = faceFrame.cx + (this.rawShellPos[i * 3] - sf.cx) * sx;
            alignedAll[i * 3 + 1] = faceFrame.cy + (this.rawShellPos[i * 3 + 1] - sf.cy) * sy;
            alignedAll[i * 3 + 2] = faceFrame.cz + (this.rawShellPos[i * 3 + 2] - sf.cz) * sz;
        }

        // 2. Measure residual seam offset at the 36 socket vertices and record reference socket coords
        for (let k = 0; k < 36; k++) {
            const mpIdx = FACE_OVAL_LOOP[k];
            const shIdx = this.shellSocketLoop[k];
            const fx = facePositions[mpIdx * 3];
            const fy = facePositions[mpIdx * 3 + 1];
            const fz = facePositions[mpIdx * 3 + 2];
            refSocketPos[k * 3] = fx;
            refSocketPos[k * 3 + 1] = fy;
            refSocketPos[k * 3 + 2] = fz;
            deltaScratch[k * 3] = fx - alignedAll[shIdx * 3];
            deltaScratch[k * 3 + 1] = fy - alignedAll[shIdx * 3 + 1];
            deltaScratch[k * 3 + 2] = fz - alignedAll[shIdx * 3 + 2];
        }

        // 3. Harmonically deform shell interior vertices so the 36-vertex weld seam is exact & smooth
        for (let i = 0; i < this.nInterior; i++) {
            const sid = this.interiorShellIds[i];
            let dx = 0;
            let dy = 0;
            let dz = 0;
            const row = i * 36;
            for (let k = 0; k < 36; k++) {
                const w = this.harmonicWeights[row + k];
                if (w > 1e-5) {
                    dx += w * deltaScratch[k * 3];
                    dy += w * deltaScratch[k * 3 + 1];
                    dz += w * deltaScratch[k * 3 + 2];
                }
            }
            fittedInteriorBase[i * 3] = alignedAll[sid * 3] + dx;
            fittedInteriorBase[i * 3 + 1] = alignedAll[sid * 3 + 1] + dy;
            fittedInteriorBase[i * 3 + 2] = alignedAll[sid * 3 + 2] + dz;
        }

        const mergedPositions = new Float32Array(totalVerts * 3);
        mergedPositions.set(facePositions.subarray(0, faceCount * 3));
        mergedPositions.set(fittedInteriorBase, 478 * 3);

        // Assign nearest-border UV to ring-1 skull vertices so seam interpolation never hits background
        const mergedUvs = new Float32Array(totalVerts * 2);
        mergedUvs.set(faceUvs.subarray(0, faceCount * 2));
        for (let i = 0; i < this.nInterior; i++) {
            const mp = this.nearestSocketMp[i];
            if (mp >= 0) {
                mergedUvs[(478 + i) * 2] = faceUvs[mp * 2];
                mergedUvs[(478 + i) * 2 + 1] = faceUvs[mp * 2 + 1];
            } else {
                mergedUvs[(478 + i) * 2] = 0.5;
                mergedUvs[(478 + i) * 2 + 1] = 0.5;
            }
        }

        const mergedFaceMask = new Float32Array(totalVerts);
        mergedFaceMask.fill(1.0, 0, faceCount);

        const mergedIndices = [...faceIndices, ...this.mergedShellIndices];

        return {
            positions: mergedPositions,
            uvs: mergedUvs,
            faceMask: mergedFaceMask,
            indices: mergedIndices,
            faceIndicesCount: faceIndices.length,
            totalVerts,
            fitState: { fittedInteriorBase, refSocketPos, deltaScratch },
        };
    }

    /**
     * Propagate live sculpt/jaw/expression movement of the 36 FACE_OVAL boundary vertices
     * into the shell interior in one allocation-free pass.
     */
    deformInto(faceSubPositions, fitState, outMergedPositions) {
        outMergedPositions.set(faceSubPositions.subarray(0, 478 * 3));
        const { fittedInteriorBase, refSocketPos, deltaScratch } = fitState;

        for (let k = 0; k < 36; k++) {
            const mpIdx = FACE_OVAL_LOOP[k];
            deltaScratch[k * 3] = faceSubPositions[mpIdx * 3] - refSocketPos[k * 3];
            deltaScratch[k * 3 + 1] = faceSubPositions[mpIdx * 3 + 1] - refSocketPos[k * 3 + 1];
            deltaScratch[k * 3 + 2] = faceSubPositions[mpIdx * 3 + 2] - refSocketPos[k * 3 + 2];
        }

        for (let i = 0; i < this.nInterior; i++) {
            let dx = 0;
            let dy = 0;
            let dz = 0;
            const row = i * 36;
            for (let k = 0; k < 36; k++) {
                const w = this.harmonicWeights[row + k];
                if (w > 1e-5) {
                    dx += w * deltaScratch[k * 3];
                    dy += w * deltaScratch[k * 3 + 1];
                    dz += w * deltaScratch[k * 3 + 2];
                }
            }
            const o = (478 + i) * 3;
            outMergedPositions[o] = fittedInteriorBase[i * 3] + dx;
            outMergedPositions[o + 1] = fittedInteriorBase[i * 3 + 1] + dy;
            outMergedPositions[o + 2] = fittedInteriorBase[i * 3 + 2] + dz;
        }
    }
}
