import pygame
import pygame_gui
import microscope_sim as sim
from enum import Enum, auto

FPS          = 60
BRUSH_RADIUS = 30
BRUSH_ALPHA  = 128
FADE_STEP    = 50

class SimMode(Enum):
    OPTO = auto()
    RANDOM_GRAY = auto()
    RANDOM_FLUORO = auto()


def main() -> None:
    pygame.init()
    microscope = sim.MicroscopeSim(overlay_mask=True, fluorescence_filter=True)  # Use the new class
    panel_width = 200
    screen = pygame.display.set_mode((microscope.width + panel_width, microscope.height))
    pygame.display.set_caption("Microscope simulation demo")
    clock = pygame.time.Clock()

    mask_surf = pygame.Surface((microscope.width, microscope.height), pygame.SRCALPHA)
    brush = pygame.Surface((2 * BRUSH_RADIUS, 2 * BRUSH_RADIUS), pygame.SRCALPHA)
    pygame.draw.circle(brush, (50, 140, 255, BRUSH_ALPHA), (BRUSH_RADIUS, BRUSH_RADIUS), BRUSH_RADIUS)

    drawing = False
    running = True
    sim_mode = SimMode.OPTO
    # fluorescent mode: 0 = off, 1 = nucleus only,2 = nucleus membrane,  3 = membrane only, 4 = both
    fluorescence_mode = 0
    #font = pygame.font.SysFont(None, 16)
    # GUI manager
    manager = pygame_gui.UIManager((microscope.width + panel_width, microscope.height))

    # Control buttons
    opto_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(microscope.width + 20, 20, 160, 30),
                                               text="OPTO Mode", manager=manager)
    gray_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(microscope.width + 20, 60, 160, 30),
                                               text="Gray Mode", manager=manager)
    fluoro_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(microscope.width + 20, 100, 160, 30),
                                                 text="Fluoro Mode", manager=manager)

    # Fluorescence selector dropdown
    fluorescence_dropdown = pygame_gui.elements.UIDropDownMenu(
        options_list=["OFF", "Nucleus", "Nucleus Membrane", "Membrane", "All"],
        starting_option="OFF",
        relative_rect=pygame.Rect(microscope.width + 20, 160, 160, 30),
        manager=manager
    )

    while running:
        # events
        time_delta = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                 drawing = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                 drawing = False
            elif event.type == pygame.MOUSEMOTION and drawing:
                 mask_surf.blit(brush, (event.pos[0] - BRUSH_RADIUS, event.pos[1] - BRUSH_RADIUS))


            # Pass event to GUI
            manager.process_events(event)

            # Handle button presses
            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == opto_button:
                        sim_mode = SimMode.OPTO
                    elif event.ui_element == gray_button:
                        sim_mode = SimMode.RANDOM_GRAY
                    elif event.ui_element == fluoro_button:
                        sim_mode = SimMode.RANDOM_FLUORO
                elif event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                    if event.ui_element == fluorescence_dropdown:
                        choice = event.text
                        mapping = {"OFF": 0, "Nucleus": 1, "Nucleus Membrane": 2, "Membrane": 3, "All": 4}
                        fluorescence_mode = mapping[choice]
        manager.update(time_delta)



        # # mask → boolean array
        mask_bool = (pygame.surfarray.array_alpha(mask_surf).T > 0)
        if sim_mode is SimMode.OPTO:
            frame = microscope.get_frame(mask_bool)
        elif sim_mode is SimMode.RANDOM_GRAY:
            frame = microscope.get_frame_random_gray()
        else:
            frame = microscope.get_frame_random_fluoro(fluorescence_mode)

        screen.fill((30, 30, 30))
        screen.blit(frame, (0, 0))
        screen.blit(mask_surf, (0,0))

        # Fade mask slowly
        mask_surf.fill((0, 0, 0, FADE_STEP), special_flags=pygame.BLEND_RGBA_SUB)

        # Draw GUI
        manager.draw_ui(screen)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

