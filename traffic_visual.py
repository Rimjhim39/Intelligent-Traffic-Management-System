
import pygame
import random
from stable_baselines3 import DQN

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Intelligent Traffic Management")

clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 22)
small_font = pygame.font.SysFont("Arial", 17)

# Road layout
ROAD = (65, 65, 65)
GREEN = (20, 190, 70)
RED = (230, 45, 45)
WHITE = (255, 255, 255)
YELLOW = (245, 200, 30)
BLACK = (20, 20, 20)

# Load the trained DQN model
model = DQN.load("traffic_dqn")

# Traffic queues
ns_queue = 12
ew_queue = 8
phase = 0
step = 0
running = True

def draw_text(text, x, y, color=WHITE, selected_font=font):
    surface = selected_font.render(text, True, color)
    screen.blit(surface, (x, y))

def draw_light(x, y, active):
    pygame.draw.circle(
        screen, GREEN if active else RED, (x, y), 13
    )

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Ask trained AI for its next decision.
    # State format must match the training environment.
    state = [
        ns_queue / 50,
        ew_queue / 50,
        float(phase)
    ]

    action, _ = model.predict(state, deterministic=True)
    phase = int(action)

    # Simulate new vehicles and movement.
    ns_queue = min(50, ns_queue + random.randint(0, 2))
    ew_queue = min(50, ew_queue + random.randint(0, 2))

    if phase == 0:
        ns_queue = max(0, ns_queue - 3)
    else:
        ew_queue = max(0, ew_queue - 3)

    step += 1

    # Draw background and intersection
    screen.fill((40, 120, 55))
    pygame.draw.rect(screen, ROAD, (300, 0, 200, HEIGHT))
    pygame.draw.rect(screen, ROAD, (0, 200, WIDTH, 200))

    # Lane markings
    for y in range(0, HEIGHT, 40):
        pygame.draw.rect(screen, YELLOW, (397, y, 6, 20))
    for x in range(0, WIDTH, 40):
        pygame.draw.rect(screen, YELLOW, (x, 297, 20, 6))

    # Signal indicators
    draw_light(280, 175, phase == 0)
    draw_light(520, 425, phase == 0)
    draw_light(520, 175, phase == 1)
    draw_light(280, 425, phase == 1)

    # Draw queued vehicles as simple car rectangles
    for i in range(min(ns_queue, 15)):
        pygame.draw.rect(
            screen, (60, 150, 255),
            (340, 20 + i * 30, 30, 18), border_radius=4
        )

    for i in range(min(ew_queue, 15)):
        pygame.draw.rect(
            screen, (255, 145, 50),
            (20 + i * 32, 240, 23, 15), border_radius=4
        )

    # Information panel
    pygame.draw.rect(screen, BLACK, (10, 10, 270, 145))
    draw_text("AI Traffic Controller", 20, 18)
    draw_text(f"North-South queue: {ns_queue}", 20, 50,
              selected_font=small_font)
    draw_text(f"East-West queue: {ew_queue}", 20, 77,
              selected_font=small_font)
    draw_text(
        "GREEN: " + ("North-South" if phase == 0 else "East-West"),
        20, 104, GREEN, small_font
    )
    draw_text(f"Simulation step: {step}", 20, 127,
              selected_font=small_font)

    pygame.display.flip()
    clock.tick(2)

pygame.quit()
