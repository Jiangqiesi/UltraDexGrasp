import os
import sys
import json
import argparse
import numpy as np
import trimesh
import coacd
from lxml import etree


def ensure_dirs(out_dir):
    for d in ["mesh", "urdf/meshes", "info"]:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)


def step_convert_units(src_path, out_dir, scale=1000.0):
    mesh = trimesh.load(src_path, force="mesh", process=False)
    mesh.vertices *= scale
    out_path = os.path.join(out_dir, "mesh", "raw.obj")
    mesh.export(out_path)
    print(f"[convert] {len(mesh.vertices)} verts, x{scale} (m->mm) -> {out_path}")
    return out_path


def step_normalize(src_path, out_dir):
    mesh = trimesh.load(src_path, force="mesh", process=False)
    bbox_min, bbox_max = mesh.bounds
    mesh.vertices -= (bbox_min + bbox_max) / 2.0
    bbox_min, bbox_max = mesh.bounds
    diag = np.linalg.norm(bbox_max - bbox_min)
    scale = 2.0 / diag
    mesh.vertices *= scale
    out_path = os.path.join(out_dir, "mesh", "normalized.obj")
    mesh.export(out_path)
    print(f"[normalize] {len(mesh.vertices)} verts, scale={scale:.6f} -> {out_path}")
    return out_path


def step_coacd_decompose(normalized_path, out_dir):
    mesh = trimesh.load(normalized_path, force="mesh", process=False)
    tm = coacd.Mesh(mesh.vertices.astype(np.float64), mesh.faces.astype(np.int32))
    parts = coacd.run_coacd(tm, threshold=0.05)
    print(f"[coacd] {len(parts)} convex pieces")

    combined = trimesh.util.concatenate(
        [trimesh.Trimesh(vertices=v, faces=f) for v, f in parts]
    )
    combined.export(os.path.join(out_dir, "mesh", "coacd.obj"))

    piece_names = []
    for i, (v, f) in enumerate(parts):
        name = f"convex_piece_{i:03d}.obj"
        piece_names.append(name)
        piece = trimesh.Trimesh(vertices=v, faces=f)
        piece.export(os.path.join(out_dir, "urdf", "meshes", name))

    print(f"[coacd] saved {len(piece_names)} pieces to urdf/meshes/")
    return piece_names


def step_export_urdf(piece_names, out_dir):
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

    out_path = os.path.join(out_dir, "urdf", "coacd.urdf")
    with open(out_path, "wb") as f:
        f.write(etree.tostring(robot, pretty_print=True, xml_declaration=False))
    print(f"[urdf] {len(piece_names)} links -> {out_path}")


def step_export_mjcf(piece_names, out_dir):
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

    out_path = os.path.join(out_dir, "urdf", "coacd.xml")
    with open(out_path, "wb") as f:
        f.write(etree.tostring(mujoco, pretty_print=True, xml_declaration=False))
    print(f"[mjcf] {len(piece_names)} pieces -> {out_path}")


def step_simplify(normalized_path, out_dir, target_faces=4000):
    mesh = trimesh.load(normalized_path, force="mesh", process=False)
    n_orig = len(mesh.vertices)
    reduction = 1.0 - (target_faces / len(mesh.faces))
    reduction = max(0.0, min(reduction, 1.0))
    simplified = mesh.simplify_quadric_decimation(reduction)
    out_path = os.path.join(out_dir, "mesh", "simplified.obj")
    simplified.export(out_path)
    print(
        f"[simplify] {n_orig} -> {len(simplified.vertices)} verts, {len(simplified.faces)} faces -> {out_path}"
    )
    return out_path


def step_basic_info(simplified_path, out_dir):
    mesh = trimesh.load(simplified_path, force="mesh", process=False)
    info = {
        "gravity_center": mesh.center_mass.tolist(),
        "obb": mesh.bounding_box_oriented.primitive.extents.tolist(),
        "scale": 1.0,
        "density": 1.0,
        "mass": float(mesh.mass),
    }
    out_path = os.path.join(out_dir, "info", "simplified.json")
    with open(out_path, "w") as f:
        json.dump(info, f, indent=1)
    print(f"[info] mass={info['mass']:.6f}, obb={info['obb']} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input .obj file path")
    parser.add_argument("--name", help="output name (default: stem of input file)")
    parser.add_argument(
        "--unit-scale",
        type=float,
        default=1000.0,
        help="unit conversion factor applied before normalization (default: 1000, m->mm)",
    )
    parser.add_argument("--target-faces", type=int, default=4000)
    args = parser.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.input))[0]
    out_dir = os.path.join("asset", "object_mesh", name)

    print(f"Input: {args.input}")
    print(f"Output: {out_dir}/")
    print(f"Unit scale: x{args.unit_scale}")
    print()

    ensure_dirs(out_dir)
    raw = step_convert_units(args.input, out_dir, scale=args.unit_scale)
    normalized = step_normalize(raw, out_dir)
    piece_names = step_coacd_decompose(normalized, out_dir)
    step_export_urdf(piece_names, out_dir)
    step_export_mjcf(piece_names, out_dir)
    simplified = step_simplify(normalized, out_dir, target_faces=args.target_faces)
    step_basic_info(simplified, out_dir)
    print(f"\nDone. Output: {out_dir}/")


if __name__ == "__main__":
    main()
