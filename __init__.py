from pathlib import Path
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .face_node import YedpFace3D
from .server_routes import register_static_routes

PACKAGE_DIR = Path(__file__).resolve().parent
WEB_DIRECTORY = "./web"


class YedpFace3DExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        register_static_routes(PACKAGE_DIR)

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [YedpFace3D]


async def comfy_entrypoint() -> YedpFace3DExtension:
    return YedpFace3DExtension()
