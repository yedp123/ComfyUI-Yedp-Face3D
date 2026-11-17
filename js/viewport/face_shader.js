import * as THREE from "../lib/three.module.js";

const VERTEX_SHADER = /* glsl */ `
    attribute float aFaceMask;

    varying vec2 vUv;
    varying float vFaceMask;
    varying vec3 vWorldNormal;
    varying vec3 vViewNormal;
    varying float vViewDepth;

    void main() {
        vUv = uv;
        vFaceMask = aFaceMask;
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        vViewNormal = normalize(normalMatrix * normal);

        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vViewDepth = -mvPosition.z;
        gl_Position = projectionMatrix * mvPosition;
    }
`;

const FRAGMENT_SHADER = /* glsl */ `
    uniform sampler2D uFrontTex;
    uniform vec3 uSkinColor;
    uniform float uHasHeadShell;
    uniform int uViewMode;
    uniform float uDepthNear;
    uniform float uDepthFar;
    uniform float uDepthInvert;
    uniform vec3 uNormalFlip;
    uniform float uAlphaMode;

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

    varying vec2 vUv;
    varying float vFaceMask;
    varying vec3 vWorldNormal;
    varying vec3 vViewNormal;
    varying float vViewDepth;

    void main() {
        float maskFactor = smoothstep(0.0, 0.85, vFaceMask);
        float outAlpha = uAlphaMode > 0.5 ? maskFactor : 1.0;

        if (uViewMode == 1) {
            // Pure unlit camera-space ControlNet RGB surface normals
            vec3 nView = normalize(vViewNormal) * uNormalFlip;
            gl_FragColor = vec4(nView * 0.5 + 0.5, outAlpha);
            return;
        }

        if (uViewMode == 2) {
            // Linear Z-depth mapped across [0.05..1.0] so foreground stays separate from 0.0 black bg
            float span = max(uDepthFar - uDepthNear, 0.001);
            float d = 1.0 - clamp((vViewDepth - uDepthNear) / span, 0.0, 1.0);
            if (uDepthInvert > 0.5) d = 1.0 - d;
            gl_FragColor = vec4(vec3(mix(0.05, 1.0, d)), outAlpha);
            return;
        }

        if (uViewMode == 3) {
            // Neutral gray Clay shaded render (soft key + hemisphere fill)
            vec3 nClay = normalize(vWorldNormal);
            float clayDiff = max(dot(nClay, normalize(uClayKeyDir)), 0.0);
            float clayHemi = 0.5 + 0.5 * nClay.y;
            vec3 clayColor = vec3(0.66) * (uClayFillIntensity * (0.7 + 0.3 * clayHemi) + clayDiff * uClayKeyIntensity);
            gl_FragColor = vec4(clamp(clayColor, 0.0, 1.0), outAlpha);
            #include <colorspace_fragment>
            return;
        }

        vec3 texAlbedo = texture2D(uFrontTex, vUv).rgb;
        vec3 albedo = uHasHeadShell > 0.5
            ? mix(uSkinColor, texAlbedo, maskFactor)
            : texAlbedo;
        vec3 n = normalize(vWorldNormal);
        float keyDiff = max(dot(n, normalize(uKeyDir)), 0.0);
        float rimDiff = max(dot(n, normalize(uRimDir)), 0.0);
        float hemi = 0.5 + 0.5 * n.y;

        vec3 lighting = vec3(uAmbientIntensity * (0.75 + 0.25 * hemi))
            + uKeyColor * (keyDiff * uKeyIntensity)
            + uRimColor * (rimDiff * uRimIntensity);

        gl_FragColor = vec4(albedo * lighting, outAlpha);
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
    uniform float uDepthInvert;
    varying float vViewDepth;
    void main() {
        float span = max(uDepthFar - uDepthNear, 0.001);
        float d = 1.0 - clamp((vViewDepth - uDepthNear) / span, 0.0, 1.0);
        if (uDepthInvert > 0.5) d = 1.0 - d;
        gl_FragColor = vec4(vec3(mix(0.05, 1.0, d)), 1.0);
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
        blending: THREE.NoBlending,
        uniforms: {
            uFrontTex: { value: fallbackTex },
            uSkinColor: { value: new THREE.Color(0xd8b4a0).convertSRGBToLinear() },
            uHasHeadShell: { value: 0.0 },
            uViewMode: { value: 0 },
            uDepthNear: { value: 1.4 },
            uDepthFar: { value: 2.3 },
            uDepthInvert: { value: 0.0 },
            uNormalFlip: { value: new THREE.Vector3(1, 1, 1) },
            uAlphaMode: { value: 0.0 },
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

export function createGuideDepthMaterial() {
    return new THREE.ShaderMaterial({
        vertexShader: GUIDE_DEPTH_VERTEX,
        fragmentShader: GUIDE_DEPTH_FRAGMENT,
        blending: THREE.NoBlending,
        uniforms: {
            uDepthNear: { value: 1.4 },
            uDepthFar: { value: 2.3 },
            uDepthInvert: { value: 0.0 },
        },
    });
}

const DEPTH_HULL_VERTEX = /* glsl */ `
    uniform vec3 uHullCenter;
    uniform float uHullScale;
    void main() {
        vec3 hullPos = mix(uHullCenter, position, uHullScale);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(hullPos, 1.0);
    }
`;

const DEPTH_HULL_FRAGMENT = /* glsl */ `
    void main() {
        gl_FragColor = vec4(0.0);
    }
`;

const OPENPOSE_VERTEX = /* glsl */ `
    attribute vec3 color;
    varying vec3 vColor;
    void main() {
        vColor = color;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

const OPENPOSE_FRAGMENT = /* glsl */ `
    varying vec3 vColor;
    void main() {
        gl_FragColor = vec4(vColor, 1.0);
    }
`;

export function createDepthHullMaterial() {
    return new THREE.ShaderMaterial({
        vertexShader: DEPTH_HULL_VERTEX,
        fragmentShader: DEPTH_HULL_FRAGMENT,
        side: THREE.DoubleSide,
        colorWrite: false,
        depthWrite: true,
        depthTest: true,
        blending: THREE.NoBlending,
        uniforms: {
            uHullCenter: { value: new THREE.Vector3(0, 0, -0.05) },
            uHullScale: { value: 0.76 },
        },
    });
}

export function createOpenPoseMaterial() {
    return new THREE.ShaderMaterial({
        vertexShader: OPENPOSE_VERTEX,
        fragmentShader: OPENPOSE_FRAGMENT,
        side: THREE.DoubleSide,
        depthTest: true,
        depthWrite: true,
        blending: THREE.NoBlending,
    });
}
