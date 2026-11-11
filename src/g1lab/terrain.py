"""Procedural physical terrain, generated without external scene downloads."""

import numpy as np


def add_terrain(root, kind: str, difficulty: float):
    """Modify an MJCF tree before compilation; return whether it uses a heightfield."""
    import xml.etree.ElementTree as ET

    world = root.find("worldbody")
    assets = root.find("asset")
    ET.SubElement(
        assets,
        "texture",
        name="floor_texture",
        type="2d",
        builtin="checker",
        rgb1="0.20 0.25 0.30",
        rgb2="0.32 0.38 0.42",
        width="512",
        height="512",
    )
    ET.SubElement(
        assets,
        "material",
        name="floor_material",
        texture="floor_texture",
        texrepeat="6 6",
        texuniform="true",
        reflectance="0.1",
    )
    material = {"rgba": "0.30 0.39 0.43 1", "friction": "0.8 0.005 0.0001"}
    material["material"] = "floor_material"
    if kind == "slope":
        angle = np.deg2rad(10 * difficulty)
        ET.SubElement(
            world,
            "geom",
            name="ground",
            type="plane",
            size="20 20 0.1",
            quat=f"{np.cos(angle / 2)} 0 {-np.sin(angle / 2)} 0",
            **material,
        )
    elif kind == "rough":
        ET.SubElement(
            assets,
            "hfield",
            name="rough",
            nrow="129",
            ncol="129",
            size=f"12 12 {max(0.001, 0.06 * difficulty)} 0.1",
        )
        ET.SubElement(world, "geom", name="ground", type="hfield", hfield="rough", **material)
        return True
    else:
        ET.SubElement(world, "geom", name="ground", type="plane", size="20 20 0.1", **material)
        if kind == "stairs":
            height = 0.10 * difficulty
            for i in range(6):
                h = height * (i + 1)
                if h > 0:
                    ET.SubElement(
                        world,
                        "geom",
                        name=f"step_{i}",
                        type="box",
                        size=f"0.20 2 {h / 2}",
                        pos=f"{1.2 + i * 0.4} 0 {h / 2}",
                        **material,
                    )
            if height > 0:
                ET.SubElement(
                    world,
                    "geom",
                    name="landing",
                    type="box",
                    size=f"2 2 {3 * height}",
                    pos=f"5.4 0 {3 * height}",
                    **material,
                )
    return False
