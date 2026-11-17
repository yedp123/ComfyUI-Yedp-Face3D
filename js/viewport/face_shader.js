import * as THREE from "../lib/three.module.js";

const VERTEX_SHADER = /* glsl */ `
    varying vec2 vUv;
    varying vec3 vWorldNormal;
    varying vec3 vViewNormal;
    varying float vViewDepth;

    void main() {
        vUv = uv;
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        vViewNormal = normalize(normalMatrix * normal);

        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vViewDepth = -mvPosition.z;
        gl_Position = projectionMatrix * mvPosition;
    }
`;

const FRAGMENT_SHADER = /* glsl */ `
    uniform sampler2D uFrontTex;
    uniform int uViewMode;
    uniform float uDepthNear;
    uniform float uDepthFar;

    uniform vec3 uKeyDir;
    uniform vec3 uKeyColor;
    uniform float uKeyIntensity;
    uniform vec3 uRimDir;
    uniform vec3 uRimColor;
    uniform float uRimIntensity;
    uniform float uAmbientIntensity;

    varying vec2 vUv;
    varying vec3 vWorldNormal;
    varying vec3 vViewNormal;
    varying float vViewDepth;

    void main() {
        if (uViewMode == 1) {
            // Camera-space ControlNet RGB surface normals
            vec3 nView = normalize(vViewNormal);
            gl_FragColor = vec4(nView * 0.5 + 0.5, 1.0);
            return;
        }

        if (uViewMode == 2) {
            // Tight-normalized Z-depth (1.0 = closest nose tip, 0.15 = back of head/ears)
            float span = max(uDepthFar - uDepthNear, 0.001);
            float d = 1.0 - clamp((vViewDepth - uDepthNear) / span, 0.0, 1.0);
            float mapped = mix(0.15, 1.0, d);
            gl_FragColor = vec4(vec3(mapped), 1.0);
            return;
        }

        vec3 albedo = texture2D(uFrontTex, vUv).rgb;
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

const GUIDE_DEPTH_VERTEX = /* glsl */ `
    varying float vViewDepth;
    void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vViewDepth = -mvPosition.z;
        gl_Position = projectionMatrix * mvPosition;
    }
`;

const GUIDE_DEPTH_FRAGMENT = /* glsl */ `
    uniform float uDepthNear;
    uniform float uDepthFar;
    varying float vViewDepth;
    void main() {
        float span = max(uDepthFar - uDepthNear, 0.001);
        float d = 1.0 - clamp((vViewDepth - uDepthNear) / span, 0.0, 1.0);
        gl_FragColor = vec4(vec3(mix(0.15, 1.0, d)), 1.0);
    }
`;

export function createPlaceholderTexture() {
    const data = new Uint8Array([216, 180, 160, 255]);
    const tex = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    return tex;
}

export function createFaceMaterial(fallbackTex) {
    return new THREE.ShaderMaterial({
        vertexShader: VERTEX_SHADER,
        fragmentShader: FRAGMENT_SHADER,
        side: THREE.DoubleSide,
        uniforms: {
            uFrontTex: { value: fallbackTex },
            uViewMode: { value: 0 },
            uDepthNear: { value: 1.4 },
            uDepthFar: { value: 2.3 },
            uKeyDir: { value: new THREE.Vector3(1.8, 1.6, 2.4).normalize() },
            uKeyColor: { value: new THREE.Color(0xfff5ea) },
            uKeyIntensity: { value: 0.72 },
            uRimDir: { value: new THREE.Vector3(-2.0, 1.1, -1.8).normalize() },
            uRimColor: { value: new THREE.Color(0x7dd3fc) },
            uRimIntensity: { value: 0.35 },
            uAmbientIntensity: { value: 0.55 },
        },
    });
}

export function createGuideDepthMaterial() {
    return new THREE.ShaderMaterial({
        vertexShader: GUIDE_DEPTH_VERTEX,
        fragmentShader: GUIDE_DEPTH_FRAGMENT,
        uniforms: {
            uDepthNear: { value: 1.4 },
            uDepthFar: { value: 2.3 },
        },
    });
}
