import * as THREE from "../lib/three.module.js";

/**
 * Creates a stylized anatomical placeholder head + optional neck/ear guides for Phase 1
 * viewport, orbit, lighting, and ResizeObserver verification before MediaPipe mesh loading.
 */
export function createPlaceholderHead() {
    const root = new THREE.Group();
    const headMeshGroup = new THREE.Group();
    const guidesGroup = new THREE.Group();

    const skinMat = new THREE.MeshStandardMaterial({
        color: 0xd8b4a0,
        roughness: 0.58,
        metalness: 0.05,
        flatShading: false,
    });

    const accentMat = new THREE.MeshStandardMaterial({
        color: 0x334155,
        roughness: 0.45,
        metalness: 0.1,
    });

    const guideMat = new THREE.MeshStandardMaterial({
        color: 0x64748b,
        roughness: 0.65,
        metalness: 0.08,
        transparent: true,
        opacity: 0.72,
    });

    const earAnchorMat = new THREE.MeshStandardMaterial({
        color: 0x38bdf8,
        roughness: 0.4,
        metalness: 0.15,
        transparent: true,
        opacity: 0.85,
    });

    const geometries = [];
    const track = (geo) => {
        geometries.push(geo);
        return geo;
    };

    // Cranium & upper face volume
    const cranium = new THREE.Mesh(track(new THREE.SphereGeometry(0.46, 32, 24)), skinMat);
    cranium.scale.set(0.84, 1.04, 0.94);
    cranium.position.set(0, 0.18, -0.02);
    headMeshGroup.add(cranium);

    // Jaw / lower face taper
    const jaw = new THREE.Mesh(track(new THREE.CylinderGeometry(0.32, 0.19, 0.36, 24)), skinMat);
    jaw.scale.set(0.95, 1.0, 0.88);
    jaw.position.set(0, -0.14, 0.06);
    headMeshGroup.add(jaw);

    // Chin anchor
    const chin = new THREE.Mesh(track(new THREE.SphereGeometry(0.18, 20, 16)), skinMat);
    chin.scale.set(1.02, 0.78, 0.92);
    chin.position.set(0, -0.32, 0.14);
    headMeshGroup.add(chin);

    // Nose bridge & tip so front orientation is obvious during orbit tests
    const nose = new THREE.Mesh(track(new THREE.ConeGeometry(0.085, 0.28, 12)), skinMat);
    nose.rotation.x = Math.PI * 0.38;
    nose.position.set(0, 0.05, 0.44);
    headMeshGroup.add(nose);

    // Brow & eye landmarks
    const brow = new THREE.Mesh(track(new THREE.BoxGeometry(0.44, 0.055, 0.12)), skinMat);
    brow.position.set(0, 0.21, 0.36);
    headMeshGroup.add(brow);

    for (const side of [-1, 1]) {
        const eye = new THREE.Mesh(track(new THREE.SphereGeometry(0.048, 16, 12)), accentMat);
        eye.position.set(side * 0.15, 0.14, 0.37);
        headMeshGroup.add(eye);
    }

    // Subtle mouth landmark bar
    const mouth = new THREE.Mesh(track(new THREE.BoxGeometry(0.19, 0.032, 0.06)), accentMat);
    mouth.position.set(0, -0.14, 0.37);
    headMeshGroup.add(mouth);

    // Optional anatomical guides: minimal neck column + left/right ear anchor markers
    const neck = new THREE.Mesh(track(new THREE.CylinderGeometry(0.2, 0.23, 0.48, 24)), guideMat);
    neck.position.set(0, -0.52, -0.05);
    guidesGroup.add(neck);

    for (const side of [-1, 1]) {
        const ear = new THREE.Mesh(track(new THREE.TorusGeometry(0.085, 0.025, 12, 24)), earAnchorMat);
        ear.rotation.y = Math.PI / 2;
        ear.position.set(side * 0.41, 0.07, -0.04);
        guidesGroup.add(ear);
    }

    root.add(headMeshGroup, guidesGroup);

    const materials = [skinMat, accentMat, guideMat, earAnchorMat];

    return {
        group: root,
        setGuidesVisible(visible) {
            guidesGroup.visible = Boolean(visible);
        },
        setWireframe(enabled) {
            const flag = Boolean(enabled);
            for (const mat of materials) {
                mat.wireframe = flag;
            }
        },
        dispose() {
            for (const geo of geometries) geo.dispose();
            for (const mat of materials) mat.dispose();
        },
    };
}
