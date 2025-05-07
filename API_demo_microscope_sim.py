"""
batch_run_microscope_sim.py
===========================
Head‑less test of microscope_sim.py.

• Creates a circular stimulation mask 80 px in diameter at (x=380, y=140).
• Advances the simulation for 120 frames (~2 s).
• Saves the first 10 frames as PNG images.
• Prints average frame time.

No window is opened; works on machines without a display by forcing the
SDL 'dummy' video driver.
"""
import os, time, pathlib

# --------------------------------------------------------------------- #
# Force head‑less SDL before importing pygame or microscope_sim         #
# --------------------------------------------------------------------- #
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import microscope_sim as sim

def make_circular_mask(cx: int, cy: int, radius: int) -> np.ndarray:
    """Return boolean mask with a filled circle of given radius."""
    y, x = np.ogrid[:sim.HEIGHT, :sim.WIDTH]
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

def main() -> None:
    # ----- create a template mask ------------------------------------ #
    stim_mask = make_circular_mask(cx=380, cy=140, radius=40)

    # ----- run the simulation ---------------------------------------- #
    n_frames   = 120
    save_first = 10
    out_dir    = pathlib.Path("headless_frames")
    out_dir.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    for i in range(n_frames):
        frame = sim.get_frame(stim_mask)          # pygame.Surface
        if i < save_first:
            pygame.image.save(frame, out_dir / f"frame_{i:03d}.png")

    elapsed = time.perf_counter() - t0
    print(f"Generated {n_frames} frames in {elapsed:.2f} s "
          f"(avg {1000*elapsed/n_frames:.2f} ms/frame).")
    print(f"First {save_first} PNGs saved in '{out_dir}/'.")

    # ----- save the stim mask as a PNG image ----------------------------- #
    stim_mask_img = pygame.Surface((sim.WIDTH, sim.HEIGHT), pygame.SRCALPHA)
    stim_mask_img.fill((0, 0, 0, 0))
    stim_mask_surface = pygame.surfarray.make_surface(stim_mask.astype(np.uint8) * 255)
    stim_mask_img.blit(stim_mask_surface, (0, 0))
    pygame.image.save(stim_mask_img, out_dir / "stim_mask.png")
    print(f"Stimulus mask saved as '{out_dir}/stim_mask.png'.")


if __name__ == "__main__":
    main()