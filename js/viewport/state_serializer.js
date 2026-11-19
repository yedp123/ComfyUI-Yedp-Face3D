const UPLOAD_SUBFOLDER = "yedp_face3d";
const IDB_NAME = "yedp_face3d_cache_v1";
const IDB_STORE = "landmarks";
const IDB_MAX_ENTRIES = 32;

export const REF_VIEW_KEYS = [
    "front",
    "three_quarter",
    "three_quarter_2",
    "side",
    "side_2",
];

export function createRefSlotMap(initialValue = null) {
    const map = {};
    for (const key of REF_VIEW_KEYS) {
        map[key] = initialValue;
    }
    return map;
}

export function normalizePersistedRefs(rawRefs = null) {
    const src = rawRefs && typeof rawRefs === "object" ? rawRefs : {};
    const out = {};
    for (const key of REF_VIEW_KEYS) {
        out[key] = src[key] || null;
    }
    return out;
}

let _dbPromise = null;

function openCacheDb() {
    if (typeof indexedDB === "undefined") return Promise.resolve(null);
    if (!_dbPromise) {
        _dbPromise = new Promise((resolve) => {
            try {
                const req = indexedDB.open(IDB_NAME, 1);
                req.onupgradeneeded = () => {
                    const db = req.result;
                    if (!db.objectStoreNames.contains(IDB_STORE)) {
                        db.createObjectStore(IDB_STORE, { keyPath: "key" });
                    }
                };
                req.onsuccess = () => resolve(req.result);
                req.onerror = () => resolve(null);
            } catch {
                resolve(null);
            }
        });
    }
    return _dbPromise;
}

export async function getCachedLandmarks(key) {
    if (!key) return null;
    const db = await openCacheDb();
    if (!db) return null;
    return new Promise((resolve) => {
        try {
            const tx = db.transaction(IDB_STORE, "readonly");
            const req = tx.objectStore(IDB_STORE).get(key);
            req.onsuccess = () => resolve(req.result?.data || null);
            req.onerror = () => resolve(null);
        } catch {
            resolve(null);
        }
    });
}

export async function setCachedLandmarks(key, data) {
    if (!key || !data) return;
    const db = await openCacheDb();
    if (!db) return;
    return new Promise((resolve) => {
        try {
            const tx = db.transaction(IDB_STORE, "readwrite");
            const store = tx.objectStore(IDB_STORE);
            store.put({ key, data, updatedAt: Date.now() });

            const countReq = store.count();
            countReq.onsuccess = () => {
                if (countReq.result > IDB_MAX_ENTRIES) {
                    const allReq = store.getAll();
                    allReq.onsuccess = () => {
                        const rows = (allReq.result || []).sort((a, b) => (a.updatedAt || 0) - (b.updatedAt || 0));
                        const excess = rows.length - IDB_MAX_ENTRIES;
                        for (let i = 0; i < excess; i++) {
                            store.delete(rows[i].key);
                        }
                    };
                }
            };
            tx.oncomplete = () => resolve();
            tx.onerror = () => resolve();
        } catch {
            resolve();
        }
    });
}

export function getRefCacheKey(refMeta) {
    if (!refMeta?.filename) return null;
    return `${refMeta.type || "input"}/${refMeta.subfolder || UPLOAD_SUBFOLDER}/${refMeta.filename}`;
}

export function buildRefViewUrl(refMeta, apiClient = null) {
    if (!refMeta?.filename) return null;
    const params = new URLSearchParams({
        filename: refMeta.filename,
        subfolder: refMeta.subfolder || UPLOAD_SUBFOLDER,
        type: refMeta.type || "input",
    });
    const relUrl = `/view?${params.toString()}`;
    return apiClient?.apiURL ? apiClient.apiURL(relUrl) : relUrl;
}

export async function uploadReferenceImage(file, viewKey, apiClient = null) {
    if (!file) return null;
    const body = new FormData();
    const fallbackName = `${viewKey}_${Date.now()}.png`;
    body.append("image", file, file.name || fallbackName);
    body.append("type", "input");
    body.append("subfolder", UPLOAD_SUBFOLDER);

    const fetchFn = apiClient?.fetchApi ? apiClient.fetchApi.bind(apiClient) : fetch;
    const res = await fetchFn("/upload/image", { method: "POST", body });
    if (!res.ok) {
        throw new Error(`Upload failed (HTTP ${res.status})`);
    }
    const json = await res.json();
    return {
        filename: json.name,
        subfolder: json.subfolder ?? UPLOAD_SUBFOLDER,
        type: json.type || "input",
    };
}

export function float32ToBase64(arr) {
    if (!arr || !arr.length) return null;
    const f32 = arr instanceof Float32Array ? arr : new Float32Array(arr);
    const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
    let bin = "";
    const step = 8192;
    for (let i = 0; i < bytes.length; i += step) {
        bin += String.fromCharCode(...bytes.subarray(i, Math.min(i + step, bytes.length)));
    }
    return btoa(bin);
}

export function base64ToFloat32(b64) {
    if (!b64) return null;
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) {
        bytes[i] = bin.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
}

export function packCameraPose(cameraPose) {
    if (!cameraPose) return null;
    return {
        projection: cameraPose.projection || "ortho",
        orthoHalfH: Number(cameraPose.orthoHalfH),
        orthoZoom: Number(cameraPose.orthoZoom ?? 1.0),
        fov: Number(cameraPose.fov ?? 36),
        position: cameraPose.position ? cameraPose.position.toArray() : [0, 0.05, 1.85],
        quaternion: cameraPose.quaternion ? cameraPose.quaternion.toArray() : [0, 0, 0, 1],
        target: cameraPose.target ? cameraPose.target.toArray() : [0, -0.02, 0],
    };
}

export function unpackCameraPose(raw, THREE) {
    if (!raw) return null;
    return {
        projection: raw.projection || "ortho",
        orthoHalfH: Number(raw.orthoHalfH || 0.6),
        orthoZoom: Number(raw.orthoZoom ?? 1.0),
        fov: Number(raw.fov ?? 36),
        position: new THREE.Vector3().fromArray(raw.position || [0, 0.05, 1.85]),
        quaternion: new THREE.Quaternion().fromArray(raw.quaternion || [0, 0, 0, 1]),
        target: new THREE.Vector3().fromArray(raw.target || [0, -0.02, 0]),
    };
}

export function packSnapshotPose(snapshotPose) {
    if (!snapshotPose?.motionFrame) return null;
    const mf = snapshotPose.motionFrame;
    if (!mf._deltasB64 && mf.deltas) {
        mf._deltasB64 = float32ToBase64(mf.deltas);
    }
    return {
        motionFrame: {
            quat: mf.quat ? [...mf.quat] : null,
            deltasB64: mf._deltasB64 || null,
            neutralAspect: mf.neutralAspect,
            mirror: Boolean(mf.mirror),
            poseAngleMag: Number(mf.poseAngleMag || 0),
            bs: mf.bs ? { ...mf.bs } : null,
        },
        cameraPose: packCameraPose(snapshotPose.cameraPose),
    };
}

export function unpackSnapshotPose(raw, THREE) {
    if (!raw?.motionFrame) return null;
    const mf = raw.motionFrame;
    const deltas = mf.deltasB64 ? base64ToFloat32(mf.deltasB64) : null;
    return {
        motionFrame: {
            quat: mf.quat ? [...mf.quat] : null,
            deltas,
            _deltasB64: mf.deltasB64 || null,
            neutralAspect: mf.neutralAspect,
            mirror: Boolean(mf.mirror),
            poseAngleMag: Number(mf.poseAngleMag || 0),
            bs: mf.bs ? { ...mf.bs } : null,
        },
        cameraPose: unpackCameraPose(raw.cameraPose, THREE),
    };
}
