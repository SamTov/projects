"""Render the prepared scenes with ZnVis' headless Mitsuba path.

Run with an environment that has ZnVis >= 2 (git+https://github.com/zincware/ZnVis).
Usage: python render_znvis.py <scene_dir> <out_dir> [resolution] [spp]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import znvis as vis  # sets the Mitsuba variant (CUDA if a GPU is visible)
import mitsuba as mi

# Paper palette (validated categorical order + single-hue blue ramp).
GREY = np.array([0.50, 0.50, 0.48])
INK = np.array([0.20, 0.20, 0.19])
SERIES = {"crystal": "#2a78d6", "liquid": "#eb6834", "vapour": "#1baf7a"}
SIGMA = 3.405


def hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


BLUE_RAMP = [hex_rgb(h) for h in ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")]


def material(rgb, roughness=0.35):
    rgb = [float(x) for x in rgb]
    bsdf = mi.load_dict({"type": "principled", "base_color": {"type": "rgb", "value": rgb},
                         "roughness": roughness, "specular": 0.4})
    return vis.Material(colour=np.array(rgb), mitsuba_bsdf=bsdf)


def scene_dict(res):
    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 6},
        "ambient": {"type": "constant", "radiance": {"type": "rgb", "value": 0.78}},
        "key": {"type": "directional", "direction": [-0.4, -0.6, -1.0],
                "irradiance": {"type": "rgb", "value": 1.1}},
        "sensor": {"type": "perspective", "fov": 32, "near_clip": 0.1, "far_clip": 5000,
                   "thefilm": {"type": "hdrfilm", "width": res, "height": res},
                   "thesampler": {"type": "multijitter", "sample_count": 64}},
    }


def spheres(name, pos, radius, rgb):
    mesh = vis.Sphere(radius=radius, resolution=24, material=material(rgb))
    return vis.Particle(name=name, mesh=mesh, position=pos[None, :, :])


def cylinders(name, mid, vec, radius, rgb):
    length = float(np.median(np.linalg.norm(vec, axis=1)))
    mesh = vis.Cylinder(radius=radius, height=length, resolution=16, material=material(rgb),
                        base_direction=np.array([0.0, 0.0, 1.0]))
    unit = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    return vis.Particle(name=name, mesh=mesh, position=mid[None, :, :], director=unit[None, :, :])


def render(particles, box, out_png: Path, res: int, spp: int, distance=2.3,
           view=(0.85, 0.55, 0.65)):
    centre = box / 2.0
    view = np.asarray(view, dtype=float)
    eye = centre + distance * view / np.linalg.norm(view) * box.max()
    cam = vis.cameras.StaticCamera(center=centre, eye=eye, up=np.array([0.0, 0.0, 1.0]))
    tmp = out_png.parent / f"_{out_png.stem}"
    viz = vis.HeadlessVisualizer(
        particles=particles, output_folder=tmp, renderer=vis.rendering.Mitsuba(scene_dict=scene_dict(res), update_camera=True),
        renderer_resolution=[res, res], renderer_spp=spp, camera=cam, do_create_video=False,
        number_of_steps=1)
    viz.render_visualization(frame_indices=[0], skip_existing_frames=False)
    frame = next(tmp.rglob("frame_000000.png"))
    shutil.move(str(frame), out_png)
    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", out_png)


def main() -> int:
    scenes, out = Path(sys.argv[1]), Path(sys.argv[2])
    res = int(sys.argv[3]) if len(sys.argv) > 3 else 1400
    spp = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    out.mkdir(parents=True, exist_ok=True)

    s = np.load(scenes / "bonded_liquid.npz")
    pos, box, w, bonded = s["positions"], s["box"], s["weight"], s["bonded"]
    parts = [spheres("free", pos[~bonded], 0.24 * SIGMA, GREY)]
    # Colour bonded atoms by their weight in the sub-bulk eigenvectors.
    edges = np.quantile(w[bonded], [0, 0.25, 0.5, 0.75, 1.0])
    for i in range(4):
        sel = bonded & (w >= edges[i]) & (w <= edges[i + 1] + 1e-12)
        if sel.any():
            parts.append(spheres(f"bonded_{i}", pos[sel], 0.38 * SIGMA, BLUE_RAMP[i]))
    parts.append(cylinders("bonds", s["bond_mid"], s["bond_vec"], 0.10 * SIGMA, INK))
    render(parts, box, out / "render_bonded_liquid.png", res, spp)

    for phase in ("crystal", "liquid", "vapour"):
        s = np.load(scenes / f"lj_{phase}.npz")
        # Look down a cube axis of the crystal so its lattice planes show.
        view = (1.0, 0.22, 0.16) if phase == "crystal" else (0.85, 0.55, 0.65)
        render([spheres(phase, s["positions"], 0.5 * SIGMA, hex_rgb(SERIES[phase]))],
               s["box"], out / f"render_lj_{phase}.png", res, spp, view=view,
               distance=2.9 if phase == "crystal" else 2.3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
