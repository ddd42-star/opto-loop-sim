import pygame
import microscope_sim as sim

FPS          = 60
BRUSH_RADIUS = 30
BRUSH_ALPHA  = 128
FADE_STEP    = 50


def main() -> None:
    pygame.init()
    microscope = sim.MicroscopeSim(overlay_mask=True, fluorescence_filter=True)  # Use the new class
    screen = pygame.display.set_mode((microscope.width, microscope.height))
    pygame.display.set_caption("Microscope simulation demo")
    clock = pygame.time.Clock()

    mask_surf = pygame.Surface((microscope.width, microscope.height), pygame.SRCALPHA)
    brush = pygame.Surface((2 * BRUSH_RADIUS, 2 * BRUSH_RADIUS), pygame.SRCALPHA)
    pygame.draw.circle(brush, (50, 140, 255, BRUSH_ALPHA), (BRUSH_RADIUS, BRUSH_RADIUS), BRUSH_RADIUS)

    drawing = False
    running = True
    # fluorescent mode: 0 = off, 1 = nucleus only, 2 = membrane only, 3 = both
    fluorescence_mode = 0
    font = pygame.font.SysFont(None, 16)

    while running:
        # events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_0:
                    fluorescence_mode = 0
                elif e.key == pygame.K_1:
                    fluorescence_mode = 1
                elif e.key == pygame.K_2:
                    fluorescence_mode = 2
                elif e.key == pygame.K_3:
                    fluorescence_mode = 3
                elif e.key == pygame.K_4:
                    fluorescence_mode = 4
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                 drawing = True
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                 drawing = False
            elif e.type == pygame.MOUSEMOTION and drawing:
                 mask_surf.blit(brush, (e.pos[0] - BRUSH_RADIUS, e.pos[1] - BRUSH_RADIUS))

        # mask → boolean array
        mask_bool = (pygame.surfarray.array_alpha(mask_surf).T > 0)
        #print(mask_bool.shape, mask_bool.dtype)
        #print(mask_bool)

        # simulation step
        #frame = microscope.get_frame(mask_bool)  # Use the instance method
        print(mask_bool.any())
        if mask_bool.any():
            frame = microscope.get_frame(mask_bool, fluorescence_mode)
        else:
            frame = microscope.get_frame_random_walk(fluorescence_mode)
        screen.blit(frame, (0, 0))
        screen.blit(mask_surf, (0, 0))

        # fade mask
        mask_surf.fill((0, 0, 0, FADE_STEP), special_flags=pygame.BLEND_RGBA_SUB)

        # Draw instructions and current mode
        mode_texts = {
            0: "Fluorescence OFF",
            1: "Nucleus fluorescence ONLY",
            2: "Nucleus Memebrane fluorescence ONLY",
            3: "Membrane fluorescence ONLY",
            4: "All fluorescence ON"
        }
        text_surf = font.render(f"Press 0-3 to change fluorescence mode: {mode_texts[fluorescence_mode]}", True, (255, 255, 255))
        screen.blit(text_surf, (10,10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()

