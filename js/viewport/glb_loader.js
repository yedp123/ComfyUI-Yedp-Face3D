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

const HEAD_JOINT_MATERIALS = new Set([
    "M_NOSE_JT",
    "M_L_EYE_JT",
    "M_R_EYE_JT",
    "M_L_EAR_JT",
    "M_R_EAR_JT",
    "M_NECK_JT",
]);

const HEAD_STICK_MATERIALS = new Set([
    "M_HEAD",
    "M_L_EYEBROW",
    "M_R_EYEBROW",
    "M_L_EARLINE",
    "M_R_EARLINE",
]);

function parseGlbChunks(arrayBuffer) {
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
    return { gltf, binChunk };
}

function buildCenteredSphereTemplate(posArr, idxArr, color) {
    const nVerts = Math.floor(posArr.length / 3);
    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (let i = 0; i < nVerts; i++) {
        cx += posArr[i * 3];
        cy += posArr[i * 3 + 1];
        cz += posArr[i * 3 + 2];
    }
    const inv = 1.0 / Math.max(nVerts, 1);
    cx *= inv;
    cy *= inv;
    cz *= inv;

    const centered = new Float32Array(nVerts * 3);
    for (let i = 0; i < nVerts; i++) {
        centered[i * 3] = posArr[i * 3] - cx;
        centered[i * 3 + 1] = posArr[i * 3 + 1] - cy;
        centered[i * 3 + 2] = posArr[i * 3 + 2] - cz;
    }
    return {
        positions: centered,
        indices: new Uint16Array(idxArr),
        color: [...color],
    };
}

function buildNormalizedStickTemplate(posArr, idxArr, color) {
    const nVerts = Math.floor(posArr.length / 3);
    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (let i = 0; i < nVerts; i++) {
        cx += posArr[i * 3];
        cy += posArr[i * 3 + 1];
        cz += posArr[i * 3 + 2];
    }
    const inv = 1.0 / Math.max(nVerts, 1);
    cx *= inv;
    cy *= inv;
    cz *= inv;

    // Identify the two opposite stick endpoints via diameter pair
    let idxA = 0;
    let bestD2 = -1;
    for (let i = 0; i < nVerts; i++) {
        const dx = posArr[i * 3] - cx;
        const dy = posArr[i * 3 + 1] - cy;
        const dz = posArr[i * 3 + 2] - cz;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > bestD2) {
            bestD2 = d2;
            idxA = i;
        }
    }
    let idxB = 0;
    bestD2 = -1;
    const ax0 = posArr[idxA * 3];
    const ay0 = posArr[idxA * 3 + 1];
    const az0 = posArr[idxA * 3 + 2];
    for (let i = 0; i < nVerts; i++) {
        const dx = posArr[i * 3] - ax0;
        const dy = posArr[i * 3 + 1] - ay0;
        const dz = posArr[i * 3 + 2] - az0;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > bestD2) {
            bestD2 = d2;
            idxB = i;
        }
    }

    const roughAxis = new THREE.Vector3(
        posArr[idxB * 3] - ax0,
        posArr[idxB * 3 + 1] - ay0,
        posArr[idxB * 3 + 2] - az0
    ).normalize();
    const midProj = roughAxis.x * cx + roughAxis.y * cy + roughAxis.z * cz;

    const endFlag = new Uint8Array(nVerts);
    const endA = new THREE.Vector3();
    const endB = new THREE.Vector3();
    let cntA = 0;
    let cntB = 0;
    for (let i = 0; i < nVerts; i++) {
        const proj = posArr[i * 3] * roughAxis.x + posArr[i * 3 + 1] * roughAxis.y + posArr[i * 3 + 2] * roughAxis.z;
        if (proj >= midProj) {
            endFlag[i] = 1;
            endB.x += posArr[i * 3];
            endB.y += posArr[i * 3 + 1];
            endB.z += posArr[i * 3 + 2];
            cntB++;
        } else {
            endFlag[i] = 0;
            endA.x += posArr[i * 3];
            endA.y += posArr[i * 3 + 1];
            endA.z += posArr[i * 3 + 2];
            cntA++;
        }
    }
    endA.multiplyScalar(1.0 / Math.max(cntA, 1));
    endB.multiplyScalar(1.0 / Math.max(cntB, 1));

    const axis = endB.clone().sub(endA).normalize();
    const refUp = Math.abs(axis.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
    const uVec = new THREE.Vector3().crossVectors(refUp, axis).normalize();
    const vVec = new THREE.Vector3().crossVectors(axis, uVec).normalize();

    const radialUV = new Float32Array(nVerts * 2);
    const pRel = new THREE.Vector3();
    for (let i = 0; i < nVerts; i++) {
        const base = endFlag[i] ? endB : endA;
        pRel.set(posArr[i * 3] - base.x, posArr[i * 3 + 1] - base.y, posArr[i * 3 + 2] - base.z);
        radialUV[i * 2] = pRel.dot(uVec);
        radialUV[i * 2 + 1] = pRel.dot(vVec);
    }

    return {
        radialUV,
        endFlag,
        indices: new Uint16Array(idxArr),
        color: [...color],
    };
}

/**
 * Extract only the unlit OpenPose Head + Neck skeleton (`Geo_OpenPose`) and 70-point
 * facial sphere template (`Geo_OpenPose_Face`) from `Yedp_Rig.glb`, skipping body meshes.
 */
export function parseOpenPoseRigGlb(arrayBuffer) {
    const { gltf, binChunk } = parseGlbChunks(arrayBuffer);
    const materials = gltf.materials || [];
    const getMatColor = (matIdx) => {
        const bc = materials[matIdx]?.pbrMetallicRoughness?.baseColorFactor;
        return bc && bc.length >= 3 ? [bc[0], bc[1], bc[2]] : [1, 1, 1];
    };

    // 1. Measure canonical face span between OP_Face_0 and OP_Face_16 from bind pose IBMs
    let rigFaceSpanX = 0.13844;
    const skin = gltf.skins?.[0];
    if (skin && skin.inverseBindMatrices !== undefined) {
        const ibmArr = readAccessorArray(gltf, binChunk, skin.inverseBindMatrices);
        const m4 = new THREE.Matrix4();
        let xLeft = null;
        let xRight = null;
        for (let j = 0; j < skin.joints.length; j++) {
            const nodeName = gltf.nodes[skin.joints[j]]?.name;
            if (nodeName === "OP_Face_0" || nodeName === "OP_Face_16") {
                m4.fromArray(ibmArr, j * 16).invert();
                const px = m4.elements[12];
                if (nodeName === "OP_Face_0") xLeft = px;
                else xRight = px;
            }
        }
        if (xLeft !== null && xRight !== null) {
            const span = Math.abs(xRight - xLeft);
            if (span > 0.02) rigFaceSpanX = span;
        }
    }

    // 2. Extract single 39-vertex sphere template from Geo_OpenPose_Face (Mesh 3 / OP_Face_*)
    let faceDotTemplate = null;
    const opFaceMesh = gltf.meshes.find((m) => m.name === "Mesh") || gltf.meshes[3];
    if (opFaceMesh?.primitives?.length) {
        const prim = opFaceMesh.primitives[0];
        const pos = readAccessorArray(gltf, binChunk, prim.attributes.POSITION);
        const idx = readAccessorArray(gltf, binChunk, prim.indices);
        const jts = readAccessorArray(gltf, binChunk, prim.attributes.JOINTS_0);
        const color = getMatColor(prim.material);

        const targetJoint = jts[0];
        const localRemap = new Int32Array(Math.floor(pos.length / 3)).fill(-1);
        const subPos = [];
        for (let i = 0; i < localRemap.length; i++) {
            if (jts[i * 4] === targetJoint) {
                localRemap[i] = subPos.length / 3;
                subPos.push(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
            }
        }
        const subIdx = [];
        for (let i = 0; i + 2 < idx.length; i += 3) {
            const a = localRemap[idx[i]];
            const b = localRemap[idx[i + 1]];
            const c = localRemap[idx[i + 2]];
            if (a >= 0 && b >= 0 && c >= 0) {
                subIdx.push(a, b, c);
            }
        }
        if (subPos.length > 0 && subIdx.length > 0) {
            faceDotTemplate = buildCenteredSphereTemplate(subPos, subIdx, color);
        }
    }

    // 3. Extract Head & Neck OpenPose spheres and sticks from Geo_OpenPose
    const jointTemplates = {};
    const stickTemplates = {};
    const opBodyMesh = gltf.meshes.find((m) => (m.primitives?.length || 0) > 10) || gltf.meshes[2];
    if (opBodyMesh?.primitives?.length) {
        for (const prim of opBodyMesh.primitives) {
            const matName = materials[prim.material]?.name;
            if (!matName) continue;
            const isJoint = HEAD_JOINT_MATERIALS.has(matName);
            const isStick = HEAD_STICK_MATERIALS.has(matName);
            if (!isJoint && !isStick) continue;

            const pos = readAccessorArray(gltf, binChunk, prim.attributes.POSITION);
            const idx = readAccessorArray(gltf, binChunk, prim.indices);
            const color = getMatColor(prim.material);

            if (isJoint) {
                jointTemplates[matName] = buildCenteredSphereTemplate(pos, idx, color);
            } else {
                stickTemplates[matName] = buildNormalizedStickTemplate(pos, idx, color);
            }
        }
    }

    if (!faceDotTemplate || !Object.keys(jointTemplates).length) {
        throw new Error("Missing OpenPose Head/Face primitives in Yedp_Rig.glb.");
    }

    return {
        rigFaceSpanX,
        faceDotTemplate,
        jointTemplates,
        stickTemplates,
    };
}

export async function fetchOpenPoseRigGlb(url) {
    const res = await fetch(url);
    if (!res.ok) {
        throw new Error(`Failed to fetch OpenPose rig GLB (${res.status}): ${url}`);
    }
    const buf = await res.arrayBuffer();
    return parseOpenPoseRigGlb(buf);
}
