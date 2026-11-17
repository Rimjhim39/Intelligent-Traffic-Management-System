
import random
import pygame
from stable_baselines3 import DQN

pygame.init()

WIDTH, HEIGHT = 900, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Traffic Control Simulator")
clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 22, bold=True)
small = pygame.font.SysFont("Arial", 16)

GREEN = (35, 230, 130)
RED = (245, 65, 80)
WHITE = (245, 245, 245)
ROAD = (48, 55, 67)
GRASS = (24, 82, 67)
BLUE = (60, 160, 255)
ORANGE = (255, 165, 65)

# Load your existing trained model
model = DQN.load("traffic_dqn")

# Select "DQN" or "TRAFFIC_AWARE"
CONTROLLER = "DQN"

# Cars store their top-left position
ns_cars = []
ew_cars = []

ns_queue = 0
ew_queue = 0
phase = 0
frame = 0
total_served = 0

def text(message, x, y, color=WHITE, selected_font=font):
    screen.blit(selected_font.render(message, True, color), (x, y))

def draw_car(x, y, color, vertical=False):
    if vertical:
        rect = pygame.Rect(x, y, 28, 38)
    else:
        rect = pygame.Rect(x, y, 38, 28)

    pygame.draw.rect(screen, color, rect, border_radius=6)

    if vertical:
        pygame.draw.rect(
            screen, (190, 225, 245),
            (x + 5, y + 5, 18, 9), border_radius=3
        )
    else:
        pygame.draw.rect(
            screen, (190, 225, 245),
            (x + 5, y + 5, 10, 18), border_radius=3
        )

def draw_light(x, y, active):
    pygame.draw.rect(
        screen, (15, 20, 28),
        (x - 17, y - 17, 34, 34), border_radius=17
    )
    pygame.draw.circle(
        screen, GREEN if active else RED, (x, y), 10
    )

def count_waiting(cars, direction):
    if direction == "NS":
        return sum(180 <= y < 430 for x, y in cars)
    return sum(260 <= x < 570 for x, y in cars)

running = True

while running:
    clock.tick(30)
    frame += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                CONTROLLER = "DQN"
            elif event.key == pygame.K_2:
                CONTROLLER = "TRAFFIC_AWARE"

    # Add vehicles randomly
    if random.random() < 0.055:
        if not ns_cars or ns_cars[0][1] > 45:
            ns_cars.insert(0, [360, -40])

    if random.random() < 0.055:
        if not ew_cars or ew_cars[0][0] > 45:
            ew_cars.insert(0, [-40, 300])

    # Estimate queues near the intersection
    ns_queue = count_waiting(ns_cars, "NS")
    ew_queue = count_waiting(ew_cars, "EW")

    # Choose signal
    if CONTROLLER == "DQN":
        observation = [
            min(ns_queue, 50) / 50,
            min(ew_queue, 50) / 50,
            float(phase)
        ]

        action, _ = model.predict(
            observation, deterministic=True
        )
        action = int(action)
    else:
        if ns_queue > ew_queue:
            action = 0
        elif ew_queue > ns_queue:
            action = 1
        else:
            action = phase

    # Change phase only after the previous cycle finishes.
    # This simple demo uses a fixed minimum green duration.
    if frame % 90 == 0:
        phase = action
    elif frame < 90:
        phase = action

    # Move north-south cars
    new_ns = []
    for x, y in ns_cars:
        speed = 3

        # Stop at red while approaching the intersection
        if phase != 0 and 140 <= y < 430:
            y = min(y + speed, 180)
        else:
            y += speed

        if y < HEIGHT:
            new_ns.append([x, y])
        else:
            total_served += 1

    ns_cars = new_ns

    # Move east-west cars
    new_ew = []
    for x, y in ew_cars:
        speed = 3

        if phase != 1 and 210 <= x < 570:
            x = min(x + speed, 260)
        else:
            x += speed

        if x < WIDTH:
            new_ew.append([x, y])
        else:
            total_served += 1

    ew_cars = new_ew

    # Draw scene
    screen.fill(GRASS)

    pygame.draw.rect(screen, ROAD, (300, 0, 200, HEIGHT))
    pygame.draw.rect(screen, ROAD, (0, 220, WIDTH, 200))

    for y in range(0, HEIGHT, 45):
        pygame.draw.line(
            screen, (205, 205, 205),
            (400, y), (400, y + 22), 2
        )

    for x in range(0, WIDTH, 45):
        pygame.draw.line(
            screen, (205, 205, 205),
            (x, 320), (x + 22, 320), 2
        )

    # Stop lines
    pygame.draw.line(screen, WHITE, (300, 215), (500, 215), 4)
    pygame.draw.line(screen, WHITE, (295, 220), (295, 420), 4)

    for x, y in ns_cars:
        draw_car(x, y, BLUE, True)

    for x, y in ew_cars:
        draw_car(x, y, ORANGE)

    # Signal displays
    draw_light(270, 185, phase == 0)
    draw_light(530, 435, phase == 0)
    draw_light(530, 185, phase == 1)
    draw_light(270, 435, phase == 1)

    # Dashboard panel
    pygame.draw.rect(
        screen, (12, 19, 32),
        (15, 15, 285, 155), border_radius=12
    )

    text("AI TRAFFIC CONTROL", 28, 25)
    text(f"Controller: {CONTROLLER}", 28, 57, GREEN, small)
    text(f"NS waiting: {ns_queue}", 28, 82, WHITE, small)
    text(f"EW waiting: {ew_queue}", 28, 104, WHITE, small)
    text(f"Vehicles passed: {total_served}", 28, 126, WHITE, small)
    text("1: DQN   2: Traffic-aware", 28, 148, WHITE, small)

    pygame.display.flip()

pygame.quit()
