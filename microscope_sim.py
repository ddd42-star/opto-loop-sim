"""
microscope_sim.py
Optogenetic cell‑motility simulator (vertex‑based stimulation).

Rule change
-----------
• A stimulation mask is a 2‑D boolean array (HEIGHT×WIDTH).
• For each cell, every vertex that overlaps a True pixel protrudes
  directly outward (radius += PROTRUSION_GAIN · base_r).

Public API
----------
get_frame(mask: np.ndarray) -> pygame.Surface
"""
from __future__ import annotations
import itertools, math, random, time
from typing import List, Tuple

import numpy as np
import pygame
from PIL import Image, ImageEnhance, ImageFilter

# --------------------------------------------------------------------------- #
# Parameters                                                                  #
# --------------------------------------------------------------------------- #
WIDTH, HEIGHT   = 800, 800
N_CELLS         = 20
VERTICES        = 24
BASE_RADIUS     = 40
FRICTION        = 8.0
BROWNIAN_D      = 30
CURVATURE_RELAX = 0.12
RADIAL_RELAX    = 0.02
PROTRUSION_GAIN = 0.05          # Δr as fraction of base_r
IMPULSE         = 10        # px s‑1 kick toward stimulated side
RUFFLE_STD      = 0.06

# faux‑microscope filter constants
_CONTRAST   = 0.55
_BLUR_RAD   = 3
_NOISE_STD  = 9
_BRIGHTNESS = 0.78

RNG = random.Random(0)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def wrap(pos: Tuple[float, float]) -> Tuple[float, float]:
    return (pos[0] % WIDTH, pos[1] % HEIGHT)

def polygon_area(pts) -> float:
    pts = np.asarray(pts)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1))

def microscope_filter(surface: pygame.Surface) -> pygame.Surface:
    raw = pygame.image.tostring(surface, "RGB")
    pil = Image.frombytes("RGB", (WIDTH, HEIGHT), raw)
    pil = pil.convert("L")
    pil = ImageEnhance.Contrast(pil).enhance(_CONTRAST)
    pil = pil.filter(ImageFilter.GaussianBlur(_BLUR_RAD))
    arr = np.asarray(pil, dtype=np.float32)
    arr += np.random.normal(0, _NOISE_STD, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="L")
    pil = ImageEnhance.Brightness(pil).enhance(_BRIGHTNESS)
    rgb = pil.convert("RGB")
    return pygame.image.fromstring(rgb.tobytes(), rgb.size, "RGB")

# --------------------------------------------------------------------------- #
# Cell                                                                        #
# --------------------------------------------------------------------------- #
class Cell:
    def __init__(self):
        self.center = np.array([RNG.uniform(0, WIDTH), RNG.uniform(0, HEIGHT)], float)
        self.base_r = BASE_RADIUS * (0.85 + 0.3 * RNG.random())
        self.r = np.full(VERTICES, self.base_r)
        self.angles = np.linspace(0, 2 * math.pi, VERTICES, endpoint=False)
        self.area0 = math.pi * self.base_r ** 2
        self.vel = np.zeros(2)

    # ---------------- stimulation ----------------
    def stimulate(self, mask: np.ndarray):
        """
        For every vertex that falls on a True pixel in *mask*,
        increase its radius outward.  No periodic wrapping: the
        mask acts like a real camera sensor.
        """
        # raw vertex coordinates (float)
        vx = self.center[0] + np.cos(self.angles) * self.r
        vy = self.center[1] + np.sin(self.angles) * self.r

        # keep only vertices inside the screen
        inside = (vx >= 0) & (vx < WIDTH) & (vy >= 0) & (vy < HEIGHT)
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
        self.r[idx] += PROTRUSION_GAIN * self.base_r
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
            self.vel += (vec / n) * IMPULSE

    # ---------------- physics update -------------
    def update(self, dt: float):
        amp = RNG.normalvariate(0, math.sqrt(2 * BROWNIAN_D * dt))
        ang = RNG.uniform(0, 2 * math.pi)
        self.vel += amp * np.array([math.cos(ang), math.sin(ang)])
        self.center += self.vel * dt
        self.center = np.array(wrap(self.center))
        self.vel *= max(0.0, 1.0 - FRICTION * dt)

        self.r += RNG.normalvariate(0, RUFFLE_STD * self.base_r) * np.cos(
            RNG.randint(1, 3) * self.angles + RNG.uniform(0, 2 * math.pi)
        )
        lap = np.roll(self.r, -1) + np.roll(self.r, 1) - 2 * self.r
        self.r += CURVATURE_RELAX * lap + RADIAL_RELAX * (self.base_r - self.r)
        self.r = np.clip(self.r, 0.4 * self.base_r, 2.2 * self.base_r)
        self._conserve_area()

    # ---------------- collision ------------------
    def collide(self, other):
        dvx = other.center[0] - self.center[0]
        dvy = other.center[1] - self.center[1]
        dvx -= WIDTH * round(dvx / WIDTH)
        dvy -= HEIGHT * round(dvy / HEIGHT)
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
        self.center = np.array(wrap(self.center))
        other.center = np.array(wrap(other.center))
        self.vel[:] = 0
        other.vel[:] = 0

    # ---------------- rendering ------------------
    def draw(self, surf: pygame.Surface):
        rel = self._rel_pts()
        layers = 6
        for ox in (-WIDTH, 0, WIDTH):
            for oy in (-HEIGHT, 0, HEIGHT):
                pts = [(x + ox + self.center[0], y + oy + self.center[1]) for x, y in rel]
                if not any(0 <= px <= WIDTH and 0 <= py <= HEIGHT for px, py in pts):
                    continue
                for i in range(layers, 0, -1):
                    s = i / layers
                    shade = 80 + int(100 * s)
                    scaled = [
                        (
                            self.center[0] + ox + (px - (self.center[0] + ox)) * s,
                            self.center[1] + oy + (py - (self.center[1] + oy)) * s,
                        )
                        for px, py in pts
                    ]
                    pygame.draw.polygon(surf, (shade, shade, 255), scaled)
                pygame.draw.polygon(surf, (0, 0, 0), pts, 1)
                pygame.draw.circle(
                    surf,
                    (60, 60, 150),
                    (int(self.center[0] + ox), int(self.center[1] + oy)),
                    int(0.4 * self.base_r),
                )

    # ---------------- helpers --------------------
    def _rel_pts(self):
        return list(zip(np.cos(self.angles) * self.r, np.sin(self.angles) * self.r))

    def _conserve_area(self):
        pts = np.array([(x + self.center[0], y + self.center[1]) for x, y in self._rel_pts()])
        area = abs(polygon_area(pts))
        if area:
            self.r *= math.sqrt(self.area0 / area)

# --------------------------------------------------------------------------- #
# Module‑level state                                                          #
# --------------------------------------------------------------------------- #
pygame.init()
_cells: List[Cell] = [Cell() for _ in range(N_CELLS)]
_last_time = time.perf_counter()
_cell_layer = pygame.Surface((WIDTH, HEIGHT))

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def get_frame(mask: np.ndarray) -> pygame.Surface:
    """
    Advance the simulation one time‑step and return the faux‑microscope image.
    *mask* must be a boolean or {0,1} array of shape (HEIGHT, WIDTH).
    """
    global _last_time
    now = time.perf_counter()
    dt = now - _last_time
    _last_time = now

    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    # stimulate cells whose vertices overlap the mask
    if mask.any():
        for c in _cells:
            c.stimulate(mask)

    # physics
    for c in _cells:
        c.update(dt)
    for a, b in itertools.combinations(_cells, 2):
        a.collide(b)

    # render
    _cell_layer.fill((235, 235, 235))
    for c in _cells:
        c.draw(_cell_layer)
    return microscope_filter(_cell_layer.copy())