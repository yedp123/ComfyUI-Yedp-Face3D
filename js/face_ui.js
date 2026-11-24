import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { createViewportDOM } from "./viewport/viewport_styles.js";
import { FaceViewport } from "./viewport/face_viewport.js";

const TARGET_NODE = "YedpFace3D";
const MIN_VIEWPORT_HEIGHT = 360;

function matchesNodeId(eventNodeId, localNodeId) {
    if (eventNodeId == null || localNodeId == null) return false;
    const raw = String(eventNodeId);
    const local = String(localNodeId);
    return raw === local || raw.split(":").pop() === local;
}

function syncNodeWidgets(node, viewport) {
    const getWidget = (name) => node.widgets?.find((w) => w.name === name);

    const guidesWidget = getWidget("anatomical_guides");
    if (guidesWidget) {
        viewport.setGuidesVisible(guidesWidget.value ?? true);
    }

    const depthWidget = getWidget("depth_weight");
    if (depthWidget) {
        viewport.depthWeight = Number(depthWidget.value ?? 1.0);
    }

    const frameCountWidget = getWidget("frame_count");
    if (frameCountWidget) {
        viewport.setFrameCount(frameCountWidget.value ?? 24);
    }

    const widthWidget = getWidget("render_width");
    const heightWidget = getWidget("render_height");
    viewport.setOutputResolution(widthWidget?.value ?? 512, heightWidget?.value ?? 512);
}

function bindNodeWidgets(node, viewport) {
    const getWidget = (name) => node.widgets?.find((w) => w.name === name);
    syncNodeWidgets(node, viewport);

    const guidesWidget = getWidget("anatomical_guides");
    if (guidesWidget) {
        viewport.onGuidesChanged = (visible) => {
            guidesWidget.value = visible;
            node.setDirtyCanvas?.(true, false);
        };
        const origCallback = guidesWidget.callback;
        guidesWidget.callback = function (value) {
            origCallback?.apply(this, arguments);
            viewport.setGuidesVisible(value);
        };
    }

    const depthWidget = getWidget("depth_weight");
    if (depthWidget) {
        const origCallback = depthWidget.callback;
        depthWidget.callback = function (value) {
            origCallback?.apply(this, arguments);
            viewport.setDepthWeight(value);
        };
    }

    const frameCountWidget = getWidget("frame_count");
    if (frameCountWidget) {
        const origCallback = frameCountWidget.callback;
        frameCountWidget.callback = function (value) {
            origCallback?.apply(this, arguments);
            viewport.setFrameCount(value);
        };
    }

    const widthWidget = getWidget("render_width");
    const heightWidget = getWidget("render_height");
    const syncOutputDims = () => {
        viewport.setOutputResolution(widthWidget?.value ?? 512, heightWidget?.value ?? 512);
    };

    for (const dimWidget of [widthWidget, heightWidget]) {
        if (!dimWidget) continue;
        const origCallback = dimWidget.callback;
        dimWidget.callback = function () {
            origCallback?.apply(this, arguments);
            syncOutputDims();
        };
    }
}

app.registerExtension({
    name: "Yedp.Face3D",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            // Let ComfyUI initialize all Python-defined widgets first so Nodes 2.0
            // CSS grid layout places the DOM widget cleanly below them without gaps.
            const result = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

            let viewport = null;
            const ui = createViewportDOM({
                onLightsChange: (lightState) => viewport?.updateLights(lightState),
                onWeightsChange: (weightState) => viewport?.updateWeights(weightState),
                onSculptChange: (sculptState) => viewport?.updateSculpt(sculptState),
                onRenderChange: (renderState, changedKey) => viewport?.updateRenderSettings(renderState, changedKey),
                onFitHead: (renderState) => viewport?.fitHeadToView(renderState),
                onSelectRefFile: (viewKey, dataUrl, file) => viewport?.loadSingleReference(viewKey, dataUrl, file),
                onResetAll: () => viewport?.clearAllReferences(),
                onStatusError: (msg) => viewport?.setStatus(msg, true),
                onExportStatic: (opts) => viewport?.exportGlb("static", opts),
                onExportTexture: () => viewport?.exportUvTexturePng(),
                onExportBlendshapes: (opts) => viewport?.exportGlb("blendshapes", opts),
                onExportAnimation: (opts) => viewport?.exportGlb("animation", opts),
            });
            viewport = new FaceViewport(ui);
            this._yedpFaceViewport = viewport;

            bindNodeWidgets(this, viewport);

            const domWidget = this.addDOMWidget("face3d_viewport", "face3d_vp", ui.container, {
                serialize: false,
                hideOnZoom: false,
                getMinHeight: () => MIN_VIEWPORT_HEIGHT,
            });
            domWidget.serialize = false;

            const onRenderJob = (event) => {
                const detail = event?.detail;
                if (!detail || !matchesNodeId(detail.node_id, this.id)) return;
                viewport.executeRenderJob(detail);
            };
            api.addEventListener("yedp_face3d_render_job", onRenderJob);

            const cleanup = () => {
                api.removeEventListener("yedp_face3d_render_job", onRenderJob);
                viewport.dispose();
                if (this._yedpFaceViewport === viewport) {
                    this._yedpFaceViewport = null;
                }
            };

            const origWidgetRemove = domWidget.onRemove;
            domWidget.onRemove = function () {
                cleanup();
                return origWidgetRemove ? origWidgetRemove.apply(this, arguments) : undefined;
            };

            const origNodeRemoved = this.onRemoved;
            this.onRemoved = function () {
                cleanup();
                return origNodeRemoved ? origNodeRemoved.apply(this, arguments) : undefined;
            };

            this.setSize([780, 700]);
            return result;
        };

        const origOnSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            origOnSerialize?.apply(this, arguments);
            if (!this._yedpFaceViewport || !o || typeof o !== "object") return;
            o.properties = o.properties || {};
            o.properties.yedp_face3d_state = this._yedpFaceViewport.exportSerializableState();
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const res = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
            if (this._yedpFaceViewport) {
                syncNodeWidgets(this, this._yedpFaceViewport);
                const saved = info?.properties?.yedp_face3d_state ?? this.properties?.yedp_face3d_state;
                if (saved) {
                    this._yedpFaceViewport.restoreSerializableState(saved);
                }
            }
            return res;
        };
    },
});
