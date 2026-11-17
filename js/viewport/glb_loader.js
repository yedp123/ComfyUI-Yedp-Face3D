import * as THREE from "../lib/three.module.js";

const GLB_MAGIC = 0x46546c67;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

const _pos = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3();
const _mat = new THREE.Matrix4();

function getNodeLocalMatrix(node) {
    if (node.matrix && node.matrix.length === 16) {
        return new THREE.Matrix4().fromArray(node.matrix);
    }
    _pos.fromArray(node.translation || [0, 0, 0]);
    _quat.fromArray(node.rotation || [0, 0, 0, 1]);
    _scale.fromArray(node.scale || [1, 1, 1]);
    return _mat.compose(_pos, _quat, _scale).clone();
}

function readAccessorArray(gltf, binBuffer, accessorIdx) {
    const acc = gltf.accessors[accessorIdx];
    const bv = gltf.bufferViews[acc.bufferView];
    const byteOffset = (bv.byteOffset || 0) + (acc.byteOffset || 0);
    const numComps = acc.type === "VEC3" ? 3 : acc.type === "VEC2" ? 2 : acc.type === "VEC4" ? 4 : 1;
    const count = acc.count * numComps;

    const ab = binBuffer.buffer;
    const baseOffset = binBuffer.byteOffset + byteOffset;

    if (acc.componentType === 5126) return new Float32Array(ab, baseOffset, count);
    if (acc.componentType === 5123) return new Uint16Array(ab, baseOffset, count);
    if (acc.componentType === 5125) return new Uint32Array(ab, baseOffset, count);
    if (acc.componentType === 5121) return new Uint8Array(ab, baseOffset, count);
    throw new Error(`Unsupported GLB accessor componentType: ${acc.componentType}`);
}

/**
 * Parse a glTF 2.0 binary (.glb) buffer, bake node transforms, and weld spatially
 * split vertices (e.g. UV seams) into a clean watertight manifold mesh.
 */
export function parseWeldedGlbMesh(arrayBuffer) {
    const view = new DataView(arrayBuffer);
    if (view.getUint32(0, true) !== GLB_MAGIC) {
        throw new Error("Invalid GLB file magic header.");
    }

    let offset = 12;
    let gltf = null;
    let binChunk = null;

    while (offset + 8 <= arrayBuffer.byteLength) {
        const chunkLen = view.getUint32(offset, true);
        const chunkType = view.getUint32(offset + 4, true);
        const chunkData = new Uint8Array(arrayBuffer, offset + 8, chunkLen);
        if (chunkType === CHUNK_JSON) {
            gltf = JSON.parse(new TextDecoder().decode(chunkData));
        } else if (chunkType === CHUNK_BIN) {
            binChunk = chunkData;
        }
        offset += 8 + chunkLen;
    }

    if (!gltf || !binChunk || !gltf.meshes?.length) {
        throw new Error("Missing JSON/BIN mesh data in GLB.");
    }

    // Collect transformed vertex positions & triangles across scene nodes
    const rawPositions = [];
    const rawIndices = [];
    const vTmp = new THREE.Vector3();

    const visitNode = (nodeIdx, parentMat) => {
        const node = gltf.nodes[nodeIdx];
        if (!node) return;
        const worldMat = parentMat.clone().multiply(getNodeLocalMatrix(node));

        if (node.mesh !== undefined) {
            const mesh = gltf.meshes[node.mesh];
            for (const prim of mesh.primitives || []) {
                if (prim.attributes?.POSITION === undefined) continue;
                const posArr = readAccessorArray(gltf, binChunk, prim.attributes.POSITION);
                const idxArr = prim.indices !== undefined ? readAccessorArray(gltf, binChunk, prim.indices) : null;
                const baseVert = rawPositions.length / 3;
                const nVerts = Math.floor(posArr.length / 3);

                for (let i = 0; i < nVerts; i++) {
                    vTmp.set(posArr[i * 3], posArr[i * 3 + 1], posArr[i * 3 + 2]).applyMatrix4(worldMat);
                    rawPositions.push(vTmp.x, vTmp.y, vTmp.z);
                }

                if (idxArr) {
                    for (let i = 0; i < idxArr.length; i++) rawIndices.push(baseVert + idxArr[i]);
                } else {
                    for (let i = 0; i < nVerts; i++) rawIndices.push(baseVert + i);
                }
            }
        }

        for (const childIdx of node.children || []) {
            visitNode(childIdx, worldMat);
        }
    };

    const sceneIdx = gltf.scene ?? 0;
    const rootNodes = gltf.scenes?.[sceneIdx]?.nodes || [0];
    const identity = new THREE.Matrix4();
    for (const rIdx of rootNodes) visitNode(rIdx, identity);

    // Weld spatially identical vertices across UV seams so Phong smoothing stays crack-free
    const posMap = new Map();
    const weldedPos = [];
    const nRaw = Math.floor(rawPositions.length / 3);
    const remap = new Int32Array(nRaw);

    for (let i = 0; i < nRaw; i++) {
        const x = rawPositions[i * 3];
        const y = rawPositions[i * 3 + 1];
        const z = rawPositions[i * 3 + 2];
        const key = `${Math.round(x * 1e4)}_${Math.round(y * 1e4)}_${Math.round(z * 1e4)}`;
        let wid = posMap.get(key);
        if (wid === undefined) {
            wid = weldedPos.length / 3;
            posMap.set(key, wid);
            weldedPos.push(x, y, z);
        }
        remap[i] = wid;
    }

    const weldedIndices = [];
    for (let i = 0; i + 2 < rawIndices.length; i += 3) {
        const a = remap[rawIndices[i]];
        const b = remap[rawIndices[i + 1]];
        const c = remap[rawIndices[i + 2]];
        if (a !== b && b !== c && c !== a) {
            weldedIndices.push(a, b, c);
        }
    }

    return {
        positions: new Float32Array(weldedPos),
        indices: weldedIndices,
    };
}

export async function fetchWeldedGlbMesh(url) {
    const res = await fetch(url);
    if (!res.ok) {
        throw new Error(`Failed to fetch GLB (${res.status}): ${url}`);
    }
    const buf = await res.arrayBuffer();
    return parseWeldedGlbMesh(buf);
}
