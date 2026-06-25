import os
import re
import numpy as np
import pandas as pd
import taichi as ti

# 1. Initialize Taichi with Apple Silicon Metal Backend
ti.init(arch=ti.metal)

def high_perf_lammps_lexer(file_path):
    print(f"[ZnVis] Ingesting {file_path} via fast Pandas C-engine...")
    
    # Fast byte scan to find line offsets for the data blocks
    num_atoms = None
    header_lines = 0
    
    with open(file_path, 'rb') as f:
        for idx, line in enumerate(f):
            if b'atoms' in line and num_atoms is None:
                num_atoms = int(line.split()[0])
            if b'Atoms' in line:
                # We want to skip everything up to and including the "Atoms" header line
                header_lines = idx + 1
                break
                
    if num_atoms is None or header_lines == 0:
        raise ValueError("Invalid LAMMPS structure: Failed to extract global headers.")

    # Read the data block instantly using the multi-threaded C engine.
    # This automatically bypasses inconsistent spacing, tabs, and trailing footers.
    df = pd.read_csv(
        file_path, 
        skiprows=header_lines, 
        nrows=num_atoms, 
        sep=r'\s+', 
        header=None,
        engine='c'
    )
    
    print(f"[ZnVis] Parsed data frame matrix shape: {df.shape}")
    raw_data = df.to_numpy(dtype=np.float32)

    # Based on standard write_data layout: 
    # Column 0: ID | Column 1: Type | Columns 2,3,4: X, Y, Z
    positions = raw_data[:, 2:5]
    types = raw_data[:, 1].astype(np.int32)

    # Filter out empty or broken rows if any exist
    valid_mask = ~np.isnan(positions).any(axis=1)
    positions = positions[valid_mask]
    types = types[valid_mask]

    # Robust spatial normalization mapping to a visible view window
    min_bound = positions.min(axis=0)
    max_bound = positions.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    max_range = np.max(max_bound - min_bound)
    
    if max_range == 0: 
        max_range = 1.0
        
    # Standard scale normalization factor to center it nicely
    normalized_positions = (positions - center) * (10.0 / max_range)

    print(f"[ZnVis] Normalization Box - Min: {min_bound}, Max: {max_bound}")
    return len(positions), normalized_positions, types


# --- Data Pipeline Setup ---
target_file = "shallow.data"

if os.path.exists(target_file):
    num_atoms, np_pos, np_types = high_perf_lammps_lexer(target_file)
else:
    print(f"[Warning] '{target_file}' not found. Using mock fallback matrix.")
    num_atoms = 1_000_000
    np_pos = (np.random.rand(num_atoms, 3).astype(np.float32) - 0.5) * 10.0
    np_types = np.random.randint(1, 3, size=num_atoms, dtype=np.int32)


# --- Taichi GPU Allocations ---
atom_positions = ti.Vector.field(3, dtype=ti.f32, shape=num_atoms)
# 4 channels (RGBA) for alpha transparency blending
atom_colors = ti.Vector.field(4, dtype=ti.f32, shape=num_atoms)
atom_radii = ti.field(dtype=ti.f32, shape=num_atoms)

# Direct transfer via Metal Unified Memory
atom_positions.from_numpy(np_pos)

@ti.kernel
def compute_species_attributes(types: ti.types.ndarray()):
    """
    Type 1 -> Carbon (Small, Dark Slate Grey, 20% Opacity)
    Type 2 -> Tin / Sn (Large, Solid Crimson Red, 100% Opacity)
    """
    for i in atom_colors:
        species = types[i]
        
        if species == 1:
            # Type 1: Carbon Layout (20% transparent)
            atom_colors[i] = ti.Vector([0.25, 0.28, 0.32, 0.2])  
            atom_radii[i] = 0.35                                  
        elif species == 2:
            # Type 2: Tin / Sn Layout (100% Solid)
            atom_colors[i] = ti.Vector([0.85, 0.15, 0.15, 1.0])  
            atom_radii[i] = 1.1                                  
        else:
            atom_colors[i] = ti.Vector([0.5, 0.5, 0.5, 0.5])
            atom_radii[i] = 0.5

compute_species_attributes(np_types)


# --- Modern GGUI Window Initialization ---
window = ti.ui.Window("ZnVis Engine - Carbon & Sn Matrix", (1280, 720), vsync=False)
canvas = window.get_canvas()
scene = window.get_scene()

# Clean, bright white backdrop canvas
canvas.set_background_color((1.0, 1.0, 1.0))

camera = ti.ui.Camera()
camera.position(0.0, 0.0, 15.0)
camera.lookat(0.0, 0.0, 0.0)
camera.up(0.0, 1.0, 0.0)

print("[ZnVis] Engine running smoothly. Left-click and drag to rotate, scroll to zoom.")

# --- Viewport Render Loop ---
while window.running:
    camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.LMB)
    scene.set_camera(camera)
    
    # Balanced light setup for clear shadows and geometry definitions
    scene.ambient_light((0.6, 0.6, 0.6))
    scene.point_light(pos=(15.0, 20.0, 15.0), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(-15.0, -20.0, -15.0), color=(0.2, 0.2, 0.2))
    
    # Draw particle arrays natively via Metal shaders
    scene.particles(
        atom_positions, 
        per_vertex_color=atom_colors, 
        per_vertex_radius=atom_radii, 
        radius=0.08  
    )
    
    canvas.scene(scene)
    window.show()
