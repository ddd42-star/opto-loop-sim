"""game.py — Optogenetic cell motility simulator

Drag the mouse to paint a semi‑transparent blue light mask. Cell edges inside
that mask protrude; cells migrate overdamped at low Reynolds numbers, collide
inelastically, and wrap around periodic boundaries. The rendered cell layer is
passed through a faux‑microscopy filter (contrast ↓, blur ↑, Gaussian noise ↑,
brightness ↓) before being displayed. The stimulation mask is drawn afterward
so it remains crisp.

Dependencies
============
python -m pip install pygame numpy pillow

Run
===
python cell_simulation.py
"""
from __future__ import annotations
import math, random, time, itertools
from typing import List, Tuple
import pygame, numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
WIDTH, HEIGHT   = 512, 512
N_CELLS         = 10
VERTICES        = 24
BASE_RADIUS     = 40
IMPULSE         = 200        # px s‑1 impulse from light
FRICTION        = 8.0        # viscous damping s‑1
BROWNIAN_D      = 30         # translational diffusion px² s‑1
CURVATURE_RELAX = 0.12       # edge smoothing rate
RADIAL_RELAX    = 0.02       # radial tension rate
PROTRUSION_GAIN = 0.55       # magnitude of light‑induced protrusion
RUFFLE_STD      = 0.06       # membrane noise (fraction of base_r)
LIGHT_DURATION  = 0.5        # seconds a brush stroke persists
BRUSH_RADIUS    = 10         # px half‑width
BRUSH_ALPHA     = 128        # 50 % opacity
FPS             = 60
RNG             = random.Random(0)

# faux‑microscope filter constants
_CONTRAST   = 0.55
_BLUR_RAD   = 3
_NOISE_STD  = 9
_BRIGHTNESS = 0.78

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def wrap(pos: Tuple[float, float]) -> Tuple[float, float]:
    """Periodic wrapping for a point."""
    return (pos[0] % WIDTH, pos[1] % HEIGHT)

def polygon_area(pts) -> float:
    pts = np.asarray(pts)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1))

# ---------------------------------------------------------------------------
# Microscope filter
# ---------------------------------------------------------------------------

def microscope_filter(surface: pygame.Surface) -> pygame.Surface:
    """Return a new pygame surface after camera‑like filtering."""
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

# ---------------------------------------------------------------------------
# Cell class
# ---------------------------------------------------------------------------
class Cell:
    def __init__(self):
        self.center = np.array([RNG.uniform(0, WIDTH), RNG.uniform(0, HEIGHT)], float)
        self.base_r = BASE_RADIUS * (0.85 + 0.3 * RNG.random())
        self.r = np.full(VERTICES, self.base_r)
        self.angles = np.linspace(0, 2 * math.pi, VERTICES, endpoint=False)
        self.area0 = math.pi * self.base_r ** 2
        self.vel = np.zeros(2)

    # ---------------- stimulation ----------------
    def stimulate(self, mask: List[np.ndarray]):
        if not mask:
            return
        pts = [self._unwrap(p) for p in mask]
        tgt = np.mean(pts, axis=0)
        vec = tgt - self.center
        n = np.linalg.norm(vec)
        if n == 0:
            return
        du = vec / n
        self.vel = du * IMPULSE
        self._protrude(du)

    # ---------------- physics update -------------
    def update(self, dt: float):
        # Brownian kick
        amp = RNG.normalvariate(0, math.sqrt(2 * BROWNIAN_D * dt))
        ang = RNG.uniform(0, 2 * math.pi)
        self.vel += amp * np.array([math.cos(ang), math.sin(ang)])
        # integrate
        self.center += self.vel * dt
        self.center = np.array(wrap(self.center))
        self.vel *= max(0.0, 1.0 - FRICTION * dt)
        # membrane noise + relaxation
        self.r += RNG.normalvariate(0, RUFFLE_STD * self.base_r) * np.cos(RNG.randint(1, 3) * self.angles + RNG.uniform(0, 2 * math.pi))
        lap = np.roll(self.r, -1) + np.roll(self.r, 1) - 2 * self.r
        self.r += CURVATURE_RELAX * lap + RADIAL_RELAX * (self.base_r - self.r)
        self.r = np.clip(self.r, 0.4 * self.base_r, 2.2 * self.base_r)
        self._conserve_area()

    # ---------------- collision (inelastic) ------
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
                    scaled = [(
                        self.center[0] + ox + (px - (self.center[0] + ox)) * s,
                        self.center[1] + oy + (py - (self.center[1] + oy)) * s,
                    ) for px, py in pts]
                    pygame.draw.polygon(surf, (shade, shade, 255), scaled)
                pygame.draw.polygon(surf, (0, 0, 0), pts, 1)
                pygame.draw.circle(surf, (60, 60, 150), (int(self.center[0] + ox), int(self.center[1] + oy)), int(0.4 * self.base_r))

    # ---------------- helpers --------------------
    def _rel_pts(self):
        return list(zip(np.cos(self.angles) * self.r, np.sin(self.angles) * self.r))

    def _protrude(self, du):
        theta = math.atan2(du[1], du[0])
        self.r += PROTRUSION_GAIN * self.base_r * np.cos(np.unwrap(self.angles - theta))
        self.r = np.clip(self.r, 0.4 * self.base_r, 2.2 * self.base_r)
        self._conserve_area()

    def _conserve_area(self):
        pts = np.array([(x + self.center[0], y + self.center[1]) for x, y in self._rel_pts()])
        area = abs(polygon_area(pts))
        if area:
            self.r *= math.sqrt(self.area0 / area)

    def _unwrap(self, p: np.ndarray) -> np.ndarray:
        return min([p + np.array([dx, dy]) for dx in (-WIDTH, 0, WIDTH) for dy in (-HEIGHT, 0, HEIGHT)], key=lambda q: np.linalg.norm(q - self.center))

    def contains(self, pts):
        return any(np.linalg.norm(self._unwrap(p) - self.center) <= 1.6 * max(self.r) for p in pts)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Cell motility – microscope view")
    clock = pygame.time.Clock()

    cells = [Cell() for _ in range(N_CELLS)]
    masks: List[Tuple[List[np.ndarray], float]] = []  # (stroke pts, expiry)
    drawing = False
    current: List[np.ndarray] = []

    brush = pygame.Surface((2 * BRUSH_RADIUS, 2 * BRUSH_RADIUS), pygame.SRCALPHA)
    pygame.draw.circle(brush, (50, 140, 255, BRUSH_ALPHA), (BRUSH_RADIUS, BRUSH_RADIUS), BRUSH_RADIUS)

    cell_layer = pygame.Surface((WIDTH, HEIGHT))

    running = True
    try:
        while running:
            dt = clock.tick(FPS) / 1000.0
            now = time.perf_counter()

            # ---------------- event handling ----------------
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    drawing = True
                    current = [np.array(e.pos, float)]
                elif e.type == pygame.MOUSEMOTION and drawing:
                    current.append(np.array(e.pos, float))
                elif e.type == pygame.MOUSEBUTTONUP and e.button == 1 and drawing:
                    drawing = False
                    if current:
                        expiry = now + LIGHT_DURATION
                        masks.append((current.copy(), expiry))
                        for c in cells:
                            if c.contains(current):
                                c.stimulate(current)
                        current.clear()

            masks = [(pts, t) for pts, t in masks if t > now]

            # ---------------- physics updates ---------------
            for c in cells:
                c.update(dt)
            for a, b in itertools.combinations(cells, 2):
                a.collide(b)

            # ---------------- rendering ---------------------
            cell_layer.fill((235, 235, 235))
            for c in cells:
                c.draw(cell_layer)
            filtered = microscope_filter(cell_layer)
            screen.blit(filtered, (0, 0))

            # draw masks (crisp)
            for pts, _ in masks:
                for p in pts:
                    pos = wrap(p)
                    screen.blit(brush, (pos[0] - BRUSH_RADIUS, pos[1] - BRUSH_RADIUS))
            if drawing and current:
                for p in current:
                    pos = wrap(p)
                    screen.blit(brush, (pos[0] - BRUSH_RADIUS, pos[1] - BRUSH_RADIUS))

            pygame.display.flip()
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()