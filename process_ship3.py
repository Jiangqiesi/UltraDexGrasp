import os
import json
import numpy as np
import trimesh
import coacd
from lxml import etree

SRC = "asset/drill.obj"
OUT = "asset/object_mesh/drill"


def ensure_dirs():
    for d in ["mesh", "urdf/meshes", "info"]:
        os.makedirs(os.path.join(OUT, d), exist_ok=True)


def step_normalize(src_path):
    mesh = trimesh.load(src_path, force="mesh", process=False)
    bbox_min, bbox_max = mesh.bounds
    mesh.vertices -= (bbox_min + bbox_max) / 2.0
    bbox_min, bbox_max = mesh.bounds
    diag = np.linalg.norm(bbox_max - bbox_min)
    scale = 2.0 / diag
    mesh.vertices *= scale
    out_path = os.path.join(OUT, "mesh", "normalized.obj")
    mesh.export(out_path)
    print(
        f"[normalize] {len(mesh.vertices)} verts, scale factor={scale:.6f} -> {out_path}"
    )
    return out_path


def step_coacd_decompose(normalized_path):
    mesh = trimesh.load(normalized_path, force="mesh", process=False)
    tm = coacd.Mesh(mesh.vertices.astype(np.float64), mesh.faces.astype(np.int32))
    parts = coacd.run_coacd(tm, threshold=0.05)
    print(f"[coacd] {len(parts)} convex pieces")

    combined = trimesh.util.concatenate(
        [trimesh.Trimesh(vertices=v, faces=f) for v, f in parts]
    )
    combined.export(os.path.join(OUT, "mesh", "coacd.obj"))

    piece_names = []
    for i, (v, f) in enumerate(parts):
        name = f"convex_piece_{i:03d}.obj"
        piece_names.append(name)
        piece = trimesh.Trimesh(vertices=v, faces=f)
        piece.export(os.path.join(OUT, "urdf", "meshes", name))

    print(f"[coacd] saved {len(piece_names)} pieces to urdf/meshes/")
    return piece_names


def step_export_urdf(piece_names):
    robot = etree.Element("robot", name="root")

    for name in piece_names:
        link = etree.SubElement(robot, "link", name=f"link_{name}")
        inertial = etree.SubElement(link, "inertial")
        etree.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        visual = etree.SubElement(link, "visual")
        etree.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
        geom_v = etree.SubElement(visual, "geometry")
        etree.SubElement(geom_v, "mesh", filename=f"meshes/{name}", scale="1.0 1.0 1.0")
        collision = etree.SubElement(link, "collision")
        etree.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        geom_c = etree.SubElement(collision, "geometry")
        etree.SubElement(geom_c, "mesh", filename=f"meshes/{name}", scale="1.0 1.0 1.0")

    for i in range(1, len(piece_names)):
        etree.SubElement(
            robot, "joint", name=f"link_{piece_names[i]}_joint", type="fixed"
        ).extend(
            [
                etree.Element("origin", xyz="0 0 0", rpy="0 0 0"),
                etree.Element("parent", link=f"link_{piece_names[0]}"),
                etree.Element("child", link=f"link_{piece_names[i]}"),
            ]
        )

    out_path = os.path.join(OUT, "urdf", "coacd.urdf")
    with open(out_path, "wb") as f:
        f.write(etree.tostring(robot, pretty_print=True, xml_declaration=False))
    print(f"[urdf] {len(piece_names)} links -> {out_path}")


def step_export_mjcf(piece_names):
    mujoco = etree.Element("mujoco", model="MuJoCo Model")
    etree.SubElement(mujoco, "compiler", angle="radian", meshdir=".")

    asset = etree.SubElement(mujoco, "asset")
    for name in piece_names:
        etree.SubElement(asset, "mesh", name=name, file=f"meshes/{name}")

    worldbody = etree.SubElement(mujoco, "worldbody")
    body = etree.SubElement(worldbody, "body", name="object")
    for i, name in enumerate(piece_names):
        etree.SubElement(
            body,
            "geom",
            name=f"object_visual_{i}",
            type="mesh",
            contype="0",
            conaffinity="0",
            density="0",
            mesh=name,
        )
        etree.SubElement(
            body, "geom", name=f"object_collision_{i}", type="mesh", mesh=name
        )

    out_path = os.path.join(OUT, "urdf", "coacd.xml")
    with open(out_path, "wb") as f:
        f.write(etree.tostring(mujoco, pretty_print=True, xml_declaration=False))
    print(f"[mjcf] {len(piece_names)} pieces -> {out_path}")


def step_simplify(normalized_path):
    mesh = trimesh.load(normalized_path, force="mesh", process=False)
    n_orig = len(mesh.vertices)
    target_faces = 4000
    reduction = 1.0 - (target_faces / len(mesh.faces))
    reduction = max(0.0, min(reduction, 1.0))
    simplified = mesh.simplify_quadric_decimation(reduction)
    out_path = os.path.join(OUT, "mesh", "simplified.obj")
    simplified.export(out_path)
    print(
        f"[simplify] {n_orig} -> {len(simplified.vertices)} verts, {len(simplified.faces)} faces -> {out_path}"
    )
    return out_path


def step_basic_info(simplified_path):
    mesh = trimesh.load(simplified_path, force="mesh", process=False)
    gravity_center = mesh.center_mass.tolist()
    obb = mesh.bounding_box_oriented.primitive.extents.tolist()
    info = {
        "gravity_center": gravity_center,
        "obb": obb,
        "scale": 1.0,
        "density": 1.0,
        "mass": float(mesh.mass),
    }
    out_path = os.path.join(OUT, "info", "simplified.json")
    with open(out_path, "w") as f:
        json.dump(info, f, indent=1)
    print(f"[info] mass={info['mass']:.6f}, obb={obb} -> {out_path}")


if __name__ == "__main__":
    ensure_dirs()
    normalized = step_normalize(SRC)
    piece_names = step_coacd_decompose(normalized)
    step_export_urdf(piece_names)
    step_export_mjcf(piece_names)
    simplified = step_simplify(normalized)
    step_basic_info(simplified)
    print(f"\nDone. Output: {OUT}/")
