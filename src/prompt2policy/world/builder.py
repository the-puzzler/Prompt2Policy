from __future__ import annotations

import logging
from pathlib import Path
import xml.etree.ElementTree as ET

from .specs import CameraSpec, ObjectSpec, WorldSpec


LOGGER = logging.getLogger("prompt2policy.world_builder")


class WorldBuilder:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root

    def compose_scene(
        self,
        world_spec: WorldSpec,
        output_path: Path,
        default_camera: CameraSpec,
    ) -> Path:
        base_scene_path = self._resolve_path(world_spec.base_scene)
        tree = ET.parse(base_scene_path)
        root = tree.getroot()

        # Many Menagerie scenes rely on include-relative asset behavior that can break
        # when serialized to another directory. For now, keep include-based scenes
        # untouched and use task/world overrides logically in the environment layer.
        if root.find("include") is not None:
            LOGGER.warning(
                "Base scene %s uses <include>; using original scene path to preserve "
                "asset resolution. World overrides remain available via task metadata.",
                base_scene_path,
            )
            return base_scene_path

        self._absolutize_includes(root, base_scene_path.parent)
        self._ensure_compiler_paths(root, base_scene_path.parent)

        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")

        existing_camera_names = {camera.attrib.get("name", "") for camera in worldbody.findall("camera")}

        cameras = list(world_spec.cameras)
        if not cameras:
            cameras = [default_camera]
        for camera in cameras:
            if camera.name in existing_camera_names:
                continue
            worldbody.append(self._camera_element(camera))

        for obj in world_spec.objects:
            if self._find_body(worldbody, obj.name) is not None:
                continue
            worldbody.append(self._object_element(obj))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        return output_path

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (self.workspace_root / path).resolve()

    def _absolutize_includes(self, root: ET.Element, parent_dir: Path) -> None:
        for include_elem in root.findall("include"):
            file_attr = include_elem.attrib.get("file")
            if not file_attr:
                continue
            include_path = Path(file_attr)
            if include_path.is_absolute():
                continue
            include_elem.attrib["file"] = str((parent_dir / include_path).resolve())

    def _ensure_compiler_paths(self, root: ET.Element, base_scene_dir: Path) -> None:
        compiler_elem = root.find("compiler")
        if compiler_elem is None:
            compiler_elem = ET.Element("compiler")
            root.insert(0, compiler_elem)

        assets_dir = (base_scene_dir / "assets").resolve()
        default_dir = assets_dir if assets_dir.exists() else base_scene_dir.resolve()

        meshdir = compiler_elem.attrib.get("meshdir")
        if not meshdir:
            compiler_elem.attrib["meshdir"] = str(default_dir)
        elif not Path(meshdir).is_absolute():
            compiler_elem.attrib["meshdir"] = str((base_scene_dir / meshdir).resolve())

        texturedir = compiler_elem.attrib.get("texturedir")
        if not texturedir:
            compiler_elem.attrib["texturedir"] = str(default_dir)
        elif not Path(texturedir).is_absolute():
            compiler_elem.attrib["texturedir"] = str((base_scene_dir / texturedir).resolve())

    def _camera_element(self, camera: CameraSpec) -> ET.Element:
        attrs = {
            "name": camera.name,
            "pos": self._to_mj_string(camera.pos),
            "fovy": str(camera.fovy),
        }
        if camera.quat is not None:
            attrs["quat"] = self._to_mj_string(camera.quat)
        return ET.Element("camera", attrs)

    def _object_element(self, obj: ObjectSpec) -> ET.Element:
        body = ET.Element(
            "body",
            {
                "name": obj.name,
                "pos": self._to_mj_string(obj.pos),
            },
        )

        geom = ET.SubElement(
            body,
            "geom",
            {
                "name": f"{obj.name}_geom",
                "type": obj.shape,
                "size": self._to_mj_string(obj.size),
                "rgba": self._to_mj_string(obj.rgba),
            },
        )

        if obj.role == "goal":
            geom.attrib["contype"] = "0"
            geom.attrib["conaffinity"] = "0"

        return body

    def _find_body(self, worldbody: ET.Element, name: str) -> ET.Element | None:
        for body in worldbody.findall("body"):
            if body.attrib.get("name") == name:
                return body
        return None

    def _to_mj_string(self, values: list[float]) -> str:
        return " ".join(f"{value:g}" for value in values)
