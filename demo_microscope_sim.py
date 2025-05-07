"""
demo_microscope_sim.py
Minimal GUI for microscope_sim.py.
Drag with the left mouse button to shine blue light.
"""
import pygame
import numpy as np
import microscope_sim as sim

FPS          = 60
BRUSH_RADIUS = 30
BRUSH_ALPHA  = 128
FADE_STEP    = 50

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((sim.WIDTH, sim.HEIGHT))
    pygame.display.set_caption("Microscope simulation demo")
    clock = pygame.time.Clock()

    mask_surf = pygame.Surface((sim.WIDTH, sim.HEIGHT), pygame.SRCALPHA)
    brush = pygame.Surface((2 * BRUSH_RADIUS, 2 * BRUSH_RADIUS), pygame.SRCALPHA)
    pygame.draw.circle(brush, (50, 140, 255, BRUSH_ALPHA), (BRUSH_RADIUS, BRUSH_RADIUS), BRUSH_RADIUS)

    drawing = False
    running = True
    while running:
        # events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                drawing = True
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                drawing = False
            elif e.type == pygame.MOUSEMOTION and drawing:
                mask_surf.blit(brush, (e.pos[0] - BRUSH_RADIUS, e.pos[1] - BRUSH_RADIUS))

        # mask → boolean array
#        mask_bool = pygame.surfarray.array_alpha(mask_surf) > 0
        mask_bool = (pygame.surfarray.array_alpha(mask_surf).T > 0)

        # simulation step
        frame = sim.get_frame(mask_bool)
        screen.blit(frame, (0, 0))
        screen.blit(mask_surf, (0, 0))

        # fade mask
        mask_surf.fill((0, 0, 0, FADE_STEP), special_flags=pygame.BLEND_RGBA_SUB)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()