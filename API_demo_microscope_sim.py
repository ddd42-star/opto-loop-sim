import os, time, pathlib

# --------------------------------------------------------------------- #
# Force head‑less SDL before importing pygame or microscope_sim         #
# --------------------------------------------------------------------- #
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import microscope_sim as sim

def make_circular_mask(cx: int, cy: int, radius: int, width: int, height: int) -> np.ndarray:
    """Return boolean mask with a filled circle of given radius."""
    y, x = np.ogrid[:height, :width]
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

def main() -> None:
    microscope = sim.MicroscopeSim(overlay_mask=True)
    # ----- create a template mask ------------------------------------ #
    stim_mask = make_circular_mask(cx=380, cy=140, radius=40, width=microscope.width, height=microscope.height)

    # ----- run the simulation ---------------------------------------- #
    n_frames   = 120
    save_first = 10
    out_dir    = pathlib.Path("headless_frames")
    out_dir.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    for i in range(n_frames):
        frame = microscope.get_frame(stim_mask)

    elapsed = time.perf_counter() - t0
    print(f"Generated {n_frames} frames in {elapsed:.2f}\u00a0s "
          f"(avg {1000*elapsed/n_frames:.2f}\u00a0ms/frame).")

    stim_mask_img = pygame.Surface((microscope.width, microscope.height), pygame.SRCALPHA)
    stim_mask_img.fill((0, 0, 0, 0))
    stim_mask_surface = pygame.surfarray.make_surface(stim_mask.astype(np.uint8) * 255)
    stim_mask_img.blit(stim_mask_surface, (0, 0))
    print(f"Stimulus mask shape: {stim_mask_img.get_size()}")

if __name__ == "__main__":
    main()