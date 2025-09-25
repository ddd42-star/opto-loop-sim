"""
microscope_sim.py
Optogenetic cell-motility simulator (vertex-based stimulation).

Rule change
-----------
- A stimulation mask is a 2-D boolean array (HEIGHTxWIDTH).
- For each cell, every vertex that overlaps a True pixel protrudes
  directly outward (radius += PROTRUSION_GAIN · base_r).

Public API
----------
get_frame(mask: np.ndarray) -> pygame.Surface
"""
from __future__ import annotations
import itertools, math, random, time
from typing import List, Tuple, Optional, Any
import numpy as np
import pygame
from PIL import Image, ImageEnhance, ImageFilter

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def wrap(pos: Tuple[float, float], width: int, height: int) -> Tuple[float, float]:
    return (pos[0] % width, pos[1] % height)

def polygon_area(pts) -> float:
    pts = np.asarray(pts)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1))

# --------------------------------------------------------------------------- #
# Cell                                                                        #
# --------------------------------------------------------------------------- #
class Cell:
    def __init__(self, width, height, base_radius, vertices, friction, brownian_d, curvature_relax, radial_relax, protrusion_gain, impulse, ruffle_std, rng, z_position):
        self.width = width
        self.height = height
        self.base_r = base_radius * (0.85 + 0.3 * rng.random())
        self.r = np.full(vertices, self.base_r)
        self.angles = np.linspace(0, 2 * math.pi, vertices, endpoint=False)
        self.area0 = math.pi * self.base_r ** 2
        self.vel = np.zeros(2)
        self.vertices = vertices
        self.friction = friction
        self.brownian_d = brownian_d
        self.curvature_relax = curvature_relax
        self.radial_relax = radial_relax
        self.protrusion_gain = protrusion_gain
        self.impulse = impulse
        self.ruffle_std = ruffle_std
        self.rng = rng
        self.center = np.array([rng.uniform(0, width), rng.uniform(0, height)], float)
        self.nucleus_fluorescence = 0.0
        self.membrane_fluorescence = np.zeros(self.vertices)
        self.nucleus_membrane_fluorescence = 0.0
        self._glow_surf = pygame.Surface((512, 512), pygame.SRCALPHA)
        self.z_position: float = z_position
        #self.z_velocity: float = 0.0

    # Add nucleus membrane fluorescence update in stimulate
    def update_nucleus_membrane_fluorescence(self, fluorescence_mode: int) -> None:
        """
        Update pixels to highlight the differents part of the cell
        """
        # Reset all fluorescence
        self.nucleus_fluorescence = 0.0
        self.nucleus_membrane_fluorescence = 0.0
        self.membrane_fluorescence[:] = 0

        if fluorescence_mode == 1:
            self.nucleus_fluorescence = 1.0
        elif fluorescence_mode == 2:
            self.nucleus_membrane_fluorescence = 1.0
        elif fluorescence_mode == 3:
            self.membrane_fluorescence[:] = 1.0
        elif fluorescence_mode == 4:
            self.nucleus_fluorescence = 1.0
            self.nucleus_membrane_fluorescence = 1.0
            self.membrane_fluorescence[:] = 1.0
    def update_fluorescence(self, mode: int) -> None:
        """
        Update fluorescence
        """
        # Reset fluorescence
        self.nucleus_fluorescence = 0.0
        self.membrane_fluorescence[:] = 0.0

        if mode == 0: # no fluorescence
            self.nucleus_fluorescence = 0.0
            self.membrane_fluorescence[:] = 0.0
        elif mode == 1: # nucleus fluorescence
            self.nucleus_fluorescence = 1.0
        elif mode == 2: # memebrane fluorescence
            self.membrane_fluorescence[:] = 1.0

    # ---------------- stimulation ----------------
    def stimulate(self, mask: np.ndarray) -> None:
        """
        For every vertex that falls on a True pixel in *mask*,
        increase its radius outward.  No periodic wrapping: the
        mask acts like a real camera sensor.
        """
        # raw vertex coordinates (float)
        vx = self.center[0] + np.cos(self.angles) * self.r
        vy = self.center[1] + np.sin(self.angles) * self.r

        # keep only vertices inside the screen
        inside = (vx >= 0) & (vx < self.width) & (vy >= 0) & (vy < self.height)
        if not inside.any():
            return

        ix = vx[inside].astype(int)
        iy = vy[inside].astype(int)
        hit = mask[iy, ix]          # Boolean array, len = num_inside

        if not hit.any():
            return

        # enlarge only the hit vertices
        idx = np.nonzero(inside)[0][hit]   # indices in self.r that need protrusion
        # 1) protrude stimulated vertices outward
        self.r[idx] += self.protrusion_gain * self.base_r
        self.r = np.clip(self.r, 0.4 * self.base_r, 2.2 * self.base_r)
        self._conserve_area()

         # 2) give the whole cell a kick toward the stimulated side
        #    direction = vector from centre to mean of hit vertices
        vx_hit = vx[idx]
        vy_hit = vy[idx]
        tgt = np.array([vx_hit.mean(), vy_hit.mean()])
        vec = tgt - self.center
        n = np.linalg.norm(vec)
        if n:
            self.vel += (vec / n) * self.impulse

    # ---------------- physics update -------------
    def update(self, dt: float) -> None:
        amp = self.rng.normalvariate(0, math.sqrt(2 * self.brownian_d * dt))
        ang = self.rng.uniform(0, 2 * math.pi)
        self.vel += amp * np.array([math.cos(ang), math.sin(ang)])
        self.center += self.vel * dt
        self.center = np.array(wrap(self.center, self.width, self.height))
        self.vel *= max(0.0, 1.0 - self.friction * dt)

        self.r += self.rng.normalvariate(0, self.ruffle_std * self.base_r) * np.cos(
            self.rng.randint(1, 3) * self.angles + self.rng.uniform(0, 2 * math.pi)
        )
        lap = np.roll(self.r, -1) + np.roll(self.r, 1) - 2 * self.r
        self.r += self.curvature_relax * lap + self.radial_relax * (self.base_r - self.r)
        self.r = np.clip(self.r, 0.4 * self.base_r, 2.2 * self.base_r)
        self._conserve_area()

    # ---------------- collision ------------------
    def collide(self, other) -> None:
        dvx = other.center[0] - self.center[0]
        dvy = other.center[1] - self.center[1]
        dvx -= self.width * round(dvx / self.width)
        dvy -= self.height * round(dvy / self.height)
        dvec = np.array([dvx, dvy])
        dist = np.linalg.norm(dvec)
        if dist == 0:
            return
        overlap = max(self.r) + max(other.r) - dist
        if overlap <= 0:
            return
        n = dvec / dist
        shift = 0.5 * (overlap + 0.5) * n
        self.center -= shift
        other.center += shift
        self.center = np.array(wrap(self.center, self.width, self.height))
        other.center = np.array(wrap(other.center, other.width, other.height))
        self.vel[:] = 0
        other.vel[:] = 0

    # ---------------- rendering ------------------
    def draw(self, surf: pygame.Surface, fluorescence_mode: int, camera_offset=(0,0)) -> None:
        # Early exit if cell is outside viewport
        viewport_x = camera_offset[0]
        viewport_y = camera_offset[1]
        viewport_w = 512
        viewport_h = 512

        # check if cell is roughly in viewport (with some for cell radius)
        margin = self.base_r * 3
        if (self.center[0] < viewport_x - margin or
            self.center[0] > viewport_x + viewport_w + margin or 
            self.center[1] < viewport_y - margin or 
            self.center[1] > viewport_y + viewport_h + margin):
            return # skip drawing the cell
        
        rel = self._rel_pts()
        layers = 6

        if np.any(camera_offset):
            offsets = [(0,0)]
        else:
            offsets = [(ox, oy) for ox in (-self.width, 0, self.width) for oy in (-self.height, 0, self.height)]


        #for ox in (-self.width, 0, self.width):
        #    for oy in (-self.height, 0, self.height):
        for ox, oy in offsets:
            pts = [(x + ox + self.center[0] - camera_offset[0], y + oy + self.center[1] - camera_offset[1]) for x, y in rel]
            if not any(0 <= px <= viewport_w and 0 <= py <= viewport_h for px, py in pts):
                continue

            # Base cell shading
            for i in range(layers, 0, -1):
                s = i / layers
                shade = 80 + int(100 * s)
                scaled = [
                    (self.center[0] + ox - camera_offset[0] + (px - (self.center[0] + ox - camera_offset[0])) * s,
                        self.center[1] + oy - camera_offset[1] + (py - (self.center[1] + oy - camera_offset[1])) * s)
                    for px, py in pts
                ]
                pygame.draw.polygon(surf, (shade, shade, 255), scaled)

            # Always draw outline
            pygame.draw.polygon(surf, (0, 0, 0), pts, 2)

            # temp surface
            self._glow_surf.fill((0,0,0,0))
            #glow_surf = pygame.Surface((512, 512), pygame.SRCALPHA)

            # --- Conditional fluorescence ---
            if fluorescence_mode == 1:  # nucleus only
                if self.nucleus_fluorescence > 0:
                    glow_radius = int(0.55 * self.base_r)
                    pygame.draw.circle(self._glow_surf, (0, 255, 255, 200),
                                        (int(self.center[0] + ox - camera_offset[0]), int(self.center[1] + oy - camera_offset[1])), glow_radius, width=6)

            elif fluorescence_mode == 2:  # nucleus membrane only
                if hasattr(self, 'nucleus_membrane_fluorescence') and self.nucleus_membrane_fluorescence > 0:
                    glow_radius = int(0.5 * self.base_r)
                    pygame.draw.circle(self._glow_surf, (255, 0, 255, 220),
                                        (int(self.center[0] + ox - camera_offset[0]), int(self.center[1] + oy - camera_offset[1])), glow_radius, width=6)

            elif fluorescence_mode == 3:  # cell membrane only
                for i, (x, y) in enumerate(pts):
                    intensity = self.membrane_fluorescence[i]
                    if intensity > 0:
                        pygame.draw.circle(self._glow_surf, (255, 165, 0, 220), (int(x), int(y)), 4)

            elif fluorescence_mode == 4:  # all on
                # nucleus
                if self.nucleus_fluorescence > 0:
                    glow_radius = int(0.55 * self.base_r)
                    pygame.draw.circle(self._glow_surf, (0, 255, 255, 220),
                                        (int(self.center[0] + ox - camera_offset[0]), int(self.center[1] + oy - camera_offset[1])), glow_radius, width=6)
                # nucleus membrane
                if hasattr(self, 'nucleus_membrane_fluorescence') and self.nucleus_membrane_fluorescence > 0:
                    glow_radius = int(0.5 * self.base_r)
                    pygame.draw.circle(self._glow_surf, (255, 0, 255, 220),
                                        (int(self.center[0] + ox - camera_offset[0]), int(self.center[1] + oy - camera_offset[1])), glow_radius, width=6)
                # membrane
                for i, (x, y) in enumerate(pts):
                    intensity = self.membrane_fluorescence[i]
                    if intensity > 0:
                        pygame.draw.circle(self._glow_surf, (255, 165, 0, 220), (int(x), int(y)), 4)

            # Finally add glow layer on top
            surf.blit(self._glow_surf, (0, 0), special_flags=pygame.BLEND_ADD)

    def draw_opto(self, surf: pygame.Surface, camera_offset=(0,0)) -> None:
                # Early exit if cell is outside viewport
        viewport_x = camera_offset[0]
        viewport_y = camera_offset[1]
        viewport_w = 512
        viewport_h = 512

        # check if cell is roughly in viewport (with some for cell radius)
        margin = self.base_r * 3
        if (self.center[0] < viewport_x - margin or
            self.center[0] > viewport_x + viewport_w + margin or 
            self.center[1] < viewport_y - margin or 
            self.center[1] > viewport_y + viewport_h + margin):
            return # skip drawing the cell
        
        rel = self._rel_pts()
        layers = 6

        if np.any(camera_offset):
            offsets = [(0,0)]
        else:
            offsets = [(ox, oy) for ox in (-self.width, 0, self.width) for oy in (-self.height, 0, self.height)]

        #for ox in (-self.width, 0, self.width):
        #    for oy in (-self.height, 0, self.height):
        for ox, oy in offsets:
            pts = [(x + ox + self.center[0] - camera_offset[0], y + oy + self.center[1] - camera_offset[1]) for x, y in rel]
            if not any(0 <= px <= viewport_w and 0 <= py <= viewport_h for px, py in pts):
                continue
            for i in range(layers, 0, -1):
                s = i / layers
                shade = 80 + int(100 * s)
                scaled = [
                    (
                        self.center[0] + ox - camera_offset[0] + (px - (self.center[0] + ox - camera_offset[0])) * s,
                        self.center[1] + oy - camera_offset[1] + (py - (self.center[1] + oy - camera_offset[1])) * s,
                    )
                    for px, py in pts
                ]
                pygame.draw.polygon(surf, (shade, shade, 255), scaled)
            pygame.draw.polygon(surf, (0, 0, 0), pts, 1)
            pygame.draw.circle(
                surf,
                (60, 60, 150),
                (int(self.center[0] + ox - camera_offset[0]), int(self.center[1] + oy - camera_offset[1])),
                int(0.4 * self.base_r),
            )
    def draw_cells(self, surf: pygame.Surface, mode: int, camera_offset: tuple[int,int]) -> None:
        # Early exit if cell is outside viewport
        viewport_x = camera_offset[0]
        viewport_y = camera_offset[1]
        viewport_w = 512
        viewport_h = 512

        # check if cell is roughly in viewport (with some for cell radius)
        margin = self.base_r * 3
        if (self.center[0] < viewport_x - margin or
                self.center[0] > viewport_x + viewport_w + margin or
                self.center[1] < viewport_y - margin or
                self.center[1] > viewport_y + viewport_h + margin):
            return  # skip drawing the cell

        rel = self._rel_pts()
        layers = 6

        if np.any(camera_offset):
            offsets = [(0, 0)]
        else:
            offsets = [(ox, oy) for ox in (-self.width, 0, self.width) for oy in (-self.height, 0, self.height)]

        for ox, oy in offsets:
            pts = [(x + ox + self.center[0] - camera_offset[0], y + oy + self.center[1] - camera_offset[1]) for x, y in
                   rel]
            if not any(0 <= px <= viewport_w and 0 <= py <= viewport_h for px, py in pts):
                continue

            if mode == 1:
                if self.nucleus_fluorescence > 0:
                    glow_radius = int(0.55 * self.base_r)
                    pygame.draw.circle(surf, (136, 8, 8),
                                       (int(self.center[0] + ox - camera_offset[0]),
                                        int(self.center[1] + oy - camera_offset[1])), glow_radius)
                # Finally add glow layer on top
                #surf.blit(self._glow_surf, (0, 0), special_flags=pygame.BLEND_ADD)
            elif mode == 2:
                # self._glow_surf.fill((0, 0, 0, 0))
                # for i, (x, y) in enumerate(pts):
                #     intensity = self.membrane_fluorescence[i]
                #     if intensity > 0:
                #         pygame.draw.circle(self._glow_surf, (255, 165, 0, 220), (int(x), int(y)), 4)
                # # Finally add glow layer on top
                # surf.blit(self._glow_surf, (0, 0), special_flags=pygame.BLEND_ADD)
                # Membrane channel: fill cell with membrane color, nucleus area with dark red
                membrane_color = (136, 8, 8)  # Blood Red
                #nucleus_color = (120, 0, 0)  # Dark red

                # Fill the whole cell
                pygame.draw.polygon(surf, membrane_color, pts)

                # Draw dark red nucleus in the center
                # nucleus_radius = int(0.55 * self.base_r)
                # pygame.draw.circle(
                #     surf,
                #     nucleus_color,
                #     (int(self.center[0] + ox - camera_offset[0]), int(self.center[1] + oy - camera_offset[1])),
                #     nucleus_radius,
                # )
            elif mode == 0:
                # Base cell shading
                for i in range(layers, 0, -1):
                    s = i / layers
                    shade = 80 + int(100 * s)
                    scaled = [
                        (self.center[0] + ox - camera_offset[0] + (px - (self.center[0] + ox - camera_offset[0])) * s,
                         self.center[1] + oy - camera_offset[1] + (py - (self.center[1] + oy - camera_offset[1])) * s)
                        for px, py in pts
                    ]
                    pygame.draw.polygon(surf, (shade, shade, 255), scaled)

                # Always draw outline
                pygame.draw.polygon(surf, (0, 0, 0), pts, 1)
                pygame.draw.circle(
                    surf,
                    (60, 60, 150),
                    (int(self.center[0] + ox - camera_offset[0]), int(self.center[1] + oy - camera_offset[1])),
                    int(0.4 * self.base_r),
                )

        # Add this method to the Cell class
    def draw_cells_scaled(self, surf: pygame.Surface, mode: int, camera_offset: tuple[int, int],
                              scale_factor: float = 1.0) -> None:
            """
            Draw cell with optional scaling factor for focus effects
            """
            # Early exit if cell is outside viewport
            viewport_x = camera_offset[0]
            viewport_y = camera_offset[1]
            viewport_w = surf.get_width()
            viewport_h = surf.get_height()

            # check if cell is roughly in viewport (with some margin for cell radius)
            margin = self.base_r * 3 * scale_factor
            if (self.center[0] < viewport_x - margin or
                    self.center[0] > viewport_x + viewport_w + margin or
                    self.center[1] < viewport_y - margin or
                    self.center[1] > viewport_y + viewport_h + margin):
                return  # skip drawing the cell

            # Scale the relative points
            rel = self._rel_pts_scaled(scale_factor)
            layers = 6

            if np.any(camera_offset):
                offsets = [(0, 0)]
            else:
                offsets = [(ox, oy) for ox in (-self.width, 0, self.width) for oy in (-self.height, 0, self.height)]

            for ox, oy in offsets:
                pts = [(x + ox + self.center[0] - camera_offset[0], y + oy + self.center[1] - camera_offset[1]) for x, y
                       in rel]
                if not any(0 <= px <= viewport_w and 0 <= py <= viewport_h for px, py in pts):
                    continue

                if mode == 1:
                    if self.nucleus_fluorescence > 0:
                        glow_radius = int(0.55 * self.base_r * scale_factor)
                        pygame.draw.circle(surf, (136, 8, 8),
                                           (int(self.center[0] + ox - camera_offset[0]),
                                            int(self.center[1] + oy - camera_offset[1])), glow_radius)
                elif mode == 2:
                    membrane_color = (136, 8, 8)  # Blood Red
                    pygame.draw.polygon(surf, membrane_color, pts)
                elif mode == 0:
                    # Base cell shading
                    for i in range(layers, 0, -1):
                        s = i / layers
                        shade = 80 + int(100 * s)
                        scaled = [
                            (self.center[0] + ox - camera_offset[0] + (
                                        px - (self.center[0] + ox - camera_offset[0])) * s,
                             self.center[1] + oy - camera_offset[1] + (
                                         py - (self.center[1] + oy - camera_offset[1])) * s)
                            for px, py in pts
                        ]
                        pygame.draw.polygon(surf, (shade, shade, 255), scaled)

                    # Always draw outline
                    pygame.draw.polygon(surf, (0, 0, 0), pts, 1)
                    pygame.draw.circle(
                        surf,
                        (60, 60, 150),
                        (int(self.center[0] + ox - camera_offset[0]), int(self.center[1] + oy - camera_offset[1])),
                        int(0.4 * self.base_r * scale_factor),
                    )

    def _rel_pts_scaled(self, scale_factor: float = 1.0):
        """Get relative points with optional scaling"""
        scaled_r = self.r * scale_factor
        return list(zip(np.cos(self.angles) * scaled_r, np.sin(self.angles) * scaled_r))



        # ---------------- helpers --------------------
    def _rel_pts(self):
        return list(zip(np.cos(self.angles) * self.r, np.sin(self.angles) * self.r))

    def _conserve_area(self):
        pts = np.array([(x + self.center[0], y + self.center[1]) for x, y in self._rel_pts()])
        area = abs(polygon_area(pts))
        if area:
            self.r *= math.sqrt(self.area0 / area)


class MicroscopeSim:
    def __init__(
        self,
        width=1500,# default 512
        height=1500,# default 512
        nb_cells=120, # default 30
        vertices=24,
        base_radius=20,
        friction=3.0,
        brownian_d=80,
        curvature_relax=0.06,
        radial_relax=0.02,
        protrusion_gain=0.05,
        impulse=10,
        ruffle_std=0.06,
        contrast=0.55,
        blur_rad=3,
        noise_std=10,
        brightness=0.78,
        overlay_mask: bool = False,
        rng_seed=0,
        focal_plane=0.0,
        depth_of_field=50.0,
        z_range=500.0
    ):
        self.width = width
        self.height = height
        self.nb_cells = nb_cells
        self.vertices = vertices
        self.base_radius = base_radius
        self.friction = friction
        self.brownian_d = brownian_d
        self.curvature_relax = curvature_relax
        self.radial_relax = radial_relax
        self.protrusion_gain = protrusion_gain
        self.impulse = impulse
        self.ruffle_std = ruffle_std
        self.contrast = contrast
        self.blur_rad = blur_rad
        self.noise_std = noise_std
        self.brightness = brightness
        self.overlay_mask = overlay_mask
        self.rng = random.Random(rng_seed)
        self._last_time = time.perf_counter()
        self.viewport_width = 512
        self.viewport_height = 512
        self._cell_layer = pygame.Surface((self.viewport_width, self.viewport_height))# render area
        self.camera_offset : tuple[float,float] = (0.0,0.0)
        self.state_devices: dict[str, dict[str, str]] = {}
        self.mode : int | None = None
        self.n_channel: int | None = None
        self.focal_plane=focal_plane
        self.depth_of_field=depth_of_field
        self.z_range=z_range
        self._cells: List[Cell] = [
            Cell(
                self.width, self.height, self.base_radius, self.vertices, self.friction, self.brownian_d,
                self.curvature_relax, self.radial_relax, self.protrusion_gain, self.impulse, self.ruffle_std, self.rng,
                z_position=0
            ) for _ in range(self.nb_cells)
        ]

    def reset(self):
        self._cells = [
            Cell(
                self.width, self.height, self.base_radius, self.vertices, self.friction, self.brownian_d,
                self.curvature_relax, self.radial_relax, self.protrusion_gain, self.impulse, self.ruffle_std, self.rng,
                z_position=0
            ) for _ in range(self.nb_cells)
        ]
        self._last_time = time.perf_counter()
        self._cell_layer = pygame.Surface((self.viewport_width, self.viewport_height))

    def _microscope_filter_opto(self, surface: pygame.Surface) -> pygame.Surface:
        raw = pygame.image.tobytes(surface, "RGB")
        pil = Image.frombytes("RGB", (self.viewport_width, self.viewport_height), raw)
        pil = pil.convert("L")
        pil = ImageEnhance.Contrast(pil).enhance(self.contrast)
        pil = pil.filter(ImageFilter.GaussianBlur(self.blur_rad))
        arr = np.asarray(pil, dtype=np.float32)
        arr += np.random.normal(0, self.noise_std, arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="L")
        pil = ImageEnhance.Brightness(pil).enhance(self.brightness)
        rgb = pil.convert("RGB")
        return pygame.image.frombytes(rgb.tobytes(), rgb.size, "RGB")


    def _microscope_filter(self, surface: pygame.Surface, fluorescence_mode: int) -> pygame.Surface:
        raw = pygame.image.tobytes(surface, "RGB")
        pil = Image.frombytes("RGB", (self.viewport_width, self.viewport_height), raw)

        if fluorescence_mode == 0:
            # Apply grayscale filter only when fluorescence is off
            pil = pil.convert("L").convert("RGB")
            pil = ImageEnhance.Contrast(pil).enhance(self.contrast)
            pil = ImageEnhance.Brightness(pil).enhance(self.brightness)
            pil = pil.filter(ImageFilter.GaussianBlur(self.blur_rad))
        else:
            # Apply mild enhancements without grayscale
            pil = ImageEnhance.Contrast(pil).enhance(self.contrast)
            pil = ImageEnhance.Brightness(pil).enhance(self.brightness)
            pil = pil.filter(ImageFilter.GaussianBlur(self.blur_rad))
            
        return pygame.image.frombytes(pil.tobytes(), pil.size, "RGB")
    def _microscope_filter_new(self, surface: pygame.Surface, mode: int) -> pygame.Surface:
        raw = pygame.image.tobytes(surface, "RGB")
        pil = Image.frombytes("RGB", (self.viewport_width, self.viewport_height), raw)

        if mode == 2:
            pil = pil.convert("L").convert("RGB")

        pil = ImageEnhance.Contrast(pil).enhance(self.contrast)
        pil = ImageEnhance.Brightness(pil).enhance(self.brightness)
        pil = pil.filter(ImageFilter.GaussianBlur(self.blur_rad))

        return pygame.image.frombytes(pil.tobytes(), pil.size, "RGB")

    def apply_intensity(self, surface: pygame.Surface, intensity: float, exposure: float) -> pygame.Surface:
        """
        Apply the intensity to the frame
        """
        # convert to array
        arr = pygame.surfarray.array3d(surface)
        mult_arr = np.multiply(arr.astype(float), intensity)
        mult_arr = np.multiply(mult_arr.astype(float), exposure)
        # clip values to stay in the right range
        mult_arr = np.clip(mult_arr, 0, 255).astype(np.uint8)
        # convert array to pygame surface
        frame = pygame.surfarray.make_surface(mult_arr)

        return frame

    def _calculate_focus_factor(self, cell_z: float) -> tuple[float, float, float]:
        """
        Calculate opacity, blur factor, and size factor based on distance from focal plane
        Returns: (opacity_factor, blur_factor, size_factor)
        """
        z_distance = abs(cell_z - self.focal_plane)

        # Objects further from focal plane become dimmer and more blurred
        opacity_factor = max(0.1, 1.0 - (z_distance / self.depth_of_field))
        blur_factor = 1.0 + (z_distance / (self.depth_of_field * 0.5))

        # Add size factor - cells appear larger when in focus, smaller when out of focus
        # This simulates the optical magnification effect of focusing
        if z_distance < self.depth_of_field * 0.5:
            # In focus - normal to slightly larger size
            size_factor = 1.0 + 0.5 * (1.0 - z_distance / (self.depth_of_field * 0.5))
        else:
            # Out of focus - progressively smaller
            size_factor = max(0.3, 1.0 - 0.7 * (z_distance - self.depth_of_field * 0.5) / self.depth_of_field)

        return opacity_factor, blur_factor, size_factor

    def _render_cell_with_focus(self, cell: Cell, surf: pygame.Surface, mode: int,
                                camera_offset: tuple[float, float]) -> None:
        """
        Render cell with focus effects, including size scaling, blur approximation,
        and fluorescence glow. Optimized for pygame rendering.
        """
        opacity_factor, blur_factor, size_factor = self._calculate_focus_factor(cell.z_position)

        # Skip cells too far out of focus
        if opacity_factor < 0.05:
            return

        # Temporary surface for the cell
        temp_size = int(self.viewport_width * 1.1)  # slightly larger for scaling
        temp_surf = pygame.Surface((temp_size, temp_size), pygame.SRCALPHA)

        # Offset for temporary surface
        scale_offset_x = camera_offset[0] - (temp_size - self.viewport_width) // 2
        scale_offset_y = camera_offset[1] - (temp_size - self.viewport_height) // 2

        # Temporarily scale cell radius
        original_base_r = cell.base_r
        original_r = cell.r.copy()
        cell.base_r *= size_factor
        cell.r *= size_factor

        # Draw cell on temp surface
        cell.draw_cells_scaled(temp_surf, mode, (scale_offset_x, scale_offset_y), scale_factor=size_factor)

        # Restore original size
        cell.base_r = original_base_r
        cell.r = original_r

        # Apply opacity
        temp_surf.set_alpha(int(255 * opacity_factor))

        # Approximate blur with additive offset copies
        blur_radius = max(1, int((blur_factor - 1.0) * 3))
        for dx in range(-blur_radius, blur_radius + 1):
            for dy in range(-blur_radius, blur_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                offset_surf = temp_surf.copy()
                offset_surf.set_alpha(int(255 * opacity_factor / ((blur_radius * 2) + 1)))
                surf.blit(offset_surf,
                          (-((temp_size - self.viewport_width) // 2) + dx,
                           -((temp_size - self.viewport_height) // 2) + dy),
                          special_flags=pygame.BLEND_ADD)

        # Draw main cell on top
        surf.blit(temp_surf, (-((temp_size - self.viewport_width) // 2),
                              -((temp_size - self.viewport_height) // 2)))

    def set_focal_plane(self, z_position: float) -> None:
        """
        Set the focal plane Z position (simulates Z-stage movement)
        """
        self.focal_plane = z_position

    def move_focus_relative(self, delta_z: float) -> None:
        """
        Move focal plane by relative amount
        """
        self.set_focal_plane(self.focal_plane + delta_z)

    def get_focal_plane(self) -> float:
        """
        Get current focal plane position
        """
        return self.focal_plane

    def get_visible_cells(self):
        """
        Return only cells visible in current viewport
        """
        visible_cells = []
        vx, vy = self.camera_offset
        margin = 100

        for cell in self._cells:
            if vx - margin <= cell.center[0] <= vx + 512 + margin and vy - margin <= cell.center[1] <= vy + 512 + margin:
                visible_cells.append(cell)

        return visible_cells

    def get_frame(self, mask: np.ndarray) -> pygame.Surface:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now

        if mask.dtype != np.bool_:
            mask = mask.astype(bool)

        if mask.any():
            for c in self._cells:
                c.stimulate(mask)
        
        for c in self._cells:
            c.update(dt)
        for a, b in itertools.combinations(self._cells, 2):
            a.collide(b)

        self._cell_layer.fill((235, 235, 235))
        # only update visible cells
        visible_cells = self.get_visible_cells()
        for c in visible_cells:
            c.draw_opto(self._cell_layer, self.camera_offset)

        frame = self._microscope_filter_opto(self._cell_layer.copy())

        if self.overlay_mask:
            mask_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            blue_rgb = (50, 140, 255)
            alpha = 128
            mask_surf.fill((0, 0, 0, 0))
            arr_rgb = pygame.surfarray.pixels3d(mask_surf)
            arr_alpha = pygame.surfarray.pixels_alpha(mask_surf)
            mask_T = mask.T
            arr_rgb[mask_T] = blue_rgb
            arr_alpha[mask_T] = alpha
            del arr_rgb, arr_alpha
            frame.blit(mask_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        return frame

    def get_frame_random_walk(self, fluorescence_mode: int) -> pygame.Surface:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now

        
        for c in self._cells:
            c.update(dt)
        for a, b in itertools.combinations(self._cells, 2):
            a.collide(b)

        self._cell_layer.fill((235, 235, 235))
        # Only update visible cells
        visible_cells = self.get_visible_cells()
        for c in visible_cells:
            c.draw(self._cell_layer, fluorescence_mode, self.camera_offset)

        frame = self._microscope_filter(self._cell_layer.copy(), fluorescence_mode)
        return frame


    def get_frame_random_gray(self) -> pygame.Surface:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        
        for c in self._cells:
            c.update(dt)
        for a, b in itertools.combinations(self._cells, 2):
            a.collide(b)
            
        self._cell_layer.fill((0,0,0))
        # only update visible cells
        visible_cells = self.get_visible_cells()
        for c in visible_cells:
            c.draw_opto(self._cell_layer, self.camera_offset)

        # force grayscale
        return self._microscope_filter(self._cell_layer, fluorescence_mode=0)

    def get_frame_random_fluoro(self, fluorescence_mode) -> pygame.Surface:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        
        for c in self._cells:
            c.update(dt)
            c.update_nucleus_membrane_fluorescence(fluorescence_mode)
        for a, b in itertools.combinations(self._cells, 2):
            a.collide(b)

            
        self._cell_layer.fill((0,0,0))
        # Only update visible cells
        visible_cells = self.get_visible_cells()
        for c in visible_cells:
            c.draw(self._cell_layer, fluorescence_mode, self.camera_offset)
        # preserve color, apply contrast/blur
        return self._microscope_filter(self._cell_layer, fluorescence_mode=fluorescence_mode)

    def snap_frame(self, mask: Optional[np.ndarray], intensity: float, exposure: float) -> np.ndarray:
        """
        This function represents the main logic that is applied to the virtual microscope
        """
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now

        # mask is not None -> optogenetics effects
        if mask is not None:
            if mask.dtype != np.bool_:
                mask = mask.astype(bool)

            if mask.any():
                for c in self._cells:
                    c.stimulate(mask)

        # checks if is a fluorescence image or not
        print(self.state_devices)
        if "Filter Wheel" not in self.state_devices.keys() or "LED" not in self.state_devices.keys():
            raise RuntimeError('Filter Wheel and LED device must be provided')
        filter_wheel_channel_label = self.state_devices["Filter Wheel"]["label"]
        led_channel_label = self.state_devices["LED"]["label"]
        # checks which mode to use
        if filter_wheel_channel_label == "mScarlet3(569/582)" and led_channel_label == "ORANGE":
            # show nucleus
            self.mode = 1
        elif filter_wheel_channel_label == "miRFP670(642/670)" and led_channel_label == "RED":
            # show membrane
            self.mode = 2
        else:
            self.mode = 0


        # Scale movement for smoother animation
        xy_speed_scale = 0.3
        z_speed_scale = 0.2
        # change based on fluorescence
        for c in self._cells:
            c.update(dt * xy_speed_scale) # shared
            #c.z_position += c.z_velocity * dt * z_speed_scale
            c._z_position = 0.0
            c.update_fluorescence(mode=self.mode)
        for a, b in itertools.combinations(self._cells, 2):
            a.collide(b) # shared

        self._cell_layer.fill((235, 235, 235))

        # Only update visible cells
        visible_cells = self.get_visible_cells()
        for c in visible_cells:
            self._render_cell_with_focus(c,self._cell_layer,self.mode, self.camera_offset)

        frame = self._microscope_filter_new(self._cell_layer.copy(), mode=self.mode)

        # convert to array
        arr = pygame.surfarray.array3d(frame)
        mult_arr = np.multiply(arr.astype(float), intensity)
        mult_arr = np.multiply(mult_arr.astype(float), exposure)
        # clip values to stay in the right range
        mult_arr = np.clip(mult_arr, 0, 255).astype(np.uint8)

        arr = mult_arr[..., 0].astype(np.uint8)
        return arr.T