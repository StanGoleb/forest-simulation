import pygame
import pygame_gui
import random
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageOps
import numpy as np
import colorsys
import math
import time

pygame.init()


# PARAMETRY DO ZMIENIANIA

SZYBKOŚĆ_ROŚLINOŻERCÓW = 1
SZYBKOŚĆ_MIĘSOŻERCÓW = 1
SZYBKOŚĆ_WSZYSTKOŻERCÓW = 0.8
SZYBKOŚĆ_DETRYTUSOŻERCÓW = 0.45 

ZMYSŁ_ROŚLINOŻERCÓW = 200
ZMYSŁ_MIĘSOŻERCÓW = 150
ZMYSŁ_WSZYSTKOŻERCÓW = 150
ZMYSŁ_DETRYTUSOŻERCÓW = 400


ILOŚĆ_ROŚLINOŻERCÓW = 6
ILOŚĆ_MIĘSOŻERCÓW = 1
ILOŚĆ_WSZYSTKOŻERCÓW = 1
ILOŚĆ_DETRYTUSOŻERCÓW = 5

ROZRODCZOŚĆ_ROŚLINOŻERCÓW = 0.25
ROZRODCZOŚĆ_MIĘSOŻERCÓW = 0.1
ROZRODCZOŚĆ_WSZYSTKOŻERCÓW = 0.06
ROZRODCZOŚĆ_DETRYTUSOŻERCÓW = 0.2

MINIMALNA_LICZBA_ŻOŁĘDZI = 5


# COLORS
BACKGROUND_COLOR = (255, 255, 255)  # White
EDGE_COLOR = (255, 0, 0)  # Red
NEAR_DEATH_COLOR = (255, 0, 0)  # Red for near-death state
DEAD_COLOR = (169, 169, 169)  # Dark Grey for dead organisms

# Speed multiplier
SPEED_MULTIPLIER = 2  # Adjust this value to speed up or slow down the simulation

# SCREEN
screen_info = pygame.display.Info()
FULL_SCREEN_WIDTH, FULL_SCREEN_HEIGHT = screen_info.current_w, screen_info.current_h

# SIMULATION 
SCREEN_WIDTH, SCREEN_HEIGHT = int(FULL_SCREEN_WIDTH * 0.6), int(FULL_SCREEN_HEIGHT * 0.7)
EDGE_WIDTH = 2
GUI_HEIGHT = 200  # Height for the GUI area

# POPULATION PLOT
PLOT_WIDTH = SCREEN_WIDTH // 2  # Half of the simulation width for the plot

TOTAL_WIDTH = SCREEN_WIDTH + PLOT_WIDTH + EDGE_WIDTH * 2

# Scaling factors for speed and sense
SPEED_SCALE = (SCREEN_WIDTH / 640) * SPEED_MULTIPLIER
SENSE_SCALE = SCREEN_HEIGHT / 480
SURVIVAL_SCALE = (SCREEN_WIDTH * SCREEN_HEIGHT) / 240

# CARNIVORE
CARNIVORE_PREY_DETECTION_DISTANCE = ZMYSŁ_MIĘSOŻERCÓW * SENSE_SCALE
CARNIVORE_SIZE = int(16 * 2 * 1.3)
CARNIVORE_RANDOM_SPEED_MEAN = SZYBKOŚĆ_MIĘSOŻERCÓW * SPEED_SCALE
CARNIVORE_CHASE_SPEED_MEAN = CARNIVORE_RANDOM_SPEED_MEAN * 2
CARNIVORE_REPRODUCTION_CHANCE = ROZRODCZOŚĆ_MIĘSOŻERCÓW
CARNIVORE_TERRITORY_REQUIREMENT = 2000
CARNIVORE_HUNGER_COLOR = 60 

# HERBIVORE
HERBIVORE_FRUIT_DETECTION_DISTANCE = ZMYSŁ_ROŚLINOŻERCÓW * SENSE_SCALE
HERBIVORE_RUN_AWAY_DISTANCE = HERBIVORE_FRUIT_DETECTION_DISTANCE / 4
HERBIVORE_SIZE = 16

HERBIVORE_RANDOM_SPEED_MEAN = SZYBKOŚĆ_ROŚLINOŻERCÓW * SPEED_SCALE
HERBIVORE_RUN_AWAY_SPEED_MEAN = HERBIVORE_RANDOM_SPEED_MEAN * 2
HERBIVORE_REPRODUCTION_CHANCE = ROZRODCZOŚĆ_ROŚLINOŻERCÓW
HERBIVORE_HUNGER_COLOR = 240  # Hue for hunger state (Blue)

# OMNIVORE
OMNIVORE_SIZE = int(16 * 2 * 1.5)
OMNIVORE_RANDOM_SPEED_MEAN = SZYBKOŚĆ_WSZYSTKOŻERCÓW * SPEED_SCALE
OMNIVORE_CHASE_SPEED_MEAN = 2 * OMNIVORE_RANDOM_SPEED_MEAN
OMNIVORE_PREY_DETECTION_DISTANCE = ZMYSŁ_WSZYSTKOŻERCÓW * SENSE_SCALE
OMNIVORE_MEAT_FULLNESS_MULTIPLIER = 2
OMNIVORE_ANY_FULLNESS_MULTIPLIER = 0.5
OMNIVORE_REPRODUCTION_CHANCE = ROZRODCZOŚĆ_WSZYSTKOŻERCÓW
OMNIVORE_HUNGER_COLOR = 30

# FRUIT
MIN_FRUITS = MINIMALNA_LICZBA_ŻOŁĘDZI
FRUIT_SIZE = 8
FRUIT_REPRODUCTION_RATE = 3
FRUIT_REPRODUCTION_CHANCE = 0.005
FRUIT_DENSITY_THRESHOLD = 0.001
NEW_FRUIT_CHANCE = 0.008
MAX_FRUITS = 50

# DECOMPOSER
DECOMPOSER_SIZE = 16
DECOMPOSER_SPEED = SZYBKOŚĆ_DETRYTUSOŻERCÓW * SPEED_SCALE
DECOMPOSER_SENSE = ZMYSŁ_DETRYTUSOŻERCÓW * SENSE_SCALE
DECOMPOSER_EATING_DURATION = 500 // SPEED_MULTIPLIER  # ticks to eat a dead organism
DECOMPOSER_REPRODUCTION_CHANCE = ROZRODCZOŚĆ_DETRYTUSOŻERCÓW  # reproduction chance
DECOMPOSER_HUNGER_COLOR = 120  # Hue for hunger state (Green)

# POPULATION SIZES
NUM_CARNIVORES = ILOŚĆ_MIĘSOŻERCÓW
NUM_HERBIVORES = ILOŚĆ_ROŚLINOŻERCÓW
NUM_FRUITS = 15
NUM_OMNIVORES = ILOŚĆ_WSZYSTKOŻERCÓW
NUM_DECOMPOSERS = ILOŚĆ_DETRYTUSOŻERCÓW
TRAIT_SPREAD = 0.25

theme =  {
    "default_theme": {
        "colours": {
            "normal_text": "#000000",
            "white_text": "#FFFFFF"
        }
    },
    "button": {
        "colours": {
            "normal_text": "#FFFFFF"
        }
    }
}

# Load images
def load_image(filename):
    image = Image.open(filename).convert('RGBA')
    image = image.resize((32, 32), Image.LANCZOS)
    return image

def resize_image(image, scale_factor):
    width, height = image.size
    return image.resize((int(width * scale_factor), int(height * scale_factor)), Image.LANCZOS)

def load_background(filename):
    background = Image.open(filename).convert('RGB')
    background = background.resize((SCREEN_WIDTH, SCREEN_HEIGHT), Image.LANCZOS)
    return background

def pil_to_surface(image):
    mode = image.mode
    size = image.size
    data = image.tobytes()
    return pygame.image.frombuffer(data, size, mode)

# Load and resize images
background_image = load_background('background.jpg')  # Replace 'background.jpg' with your background image file
background_surface = pil_to_surface(background_image)
background_surface.set_alpha(100)  # Set the alpha value to 100 (range is 0 to 255)

carnivore_image = resize_image(load_image('carnivore.png'), 2)  # Make carnivore image 2x larger
herbivore_image = load_image('herbivore.png')
omnivore_image = resize_image(load_image('omnivore.png'), 2)  # Make omnivore image 2x larger
decomposer_image = load_image('decomposer.png')
fruit_image = load_image('fruit.png')

# Utility functions for colorizing and rotating images
def rgb_to_hsv_vectorized(r, g, b):
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

def hsv_to_rgb_vectorized(h, s, v):
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

def shift_hue(arr, hout):
    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = np.vectorize(rgb_to_hsv_vectorized)(r, g, b)
    h = hout
    r, g, b = np.vectorize(hsv_to_rgb_vectorized)(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

def colorize(image, hue):
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, hue / 720.0).astype('uint8'), 'RGBA')  # Reduced the hue shift strength further
    return new_img

def pil_to_surface(image):
    mode = image.mode
    size = image.size
    data = image.tobytes()
    return pygame.image.frombuffer(data, size, mode)

# Define the calculate_survival_time and calculate_full_duration functions
def calculate_survival_time(speed, sense):
    int(SURVIVAL_SCALE / (speed + sense * 0.01))
    return int(SURVIVAL_SCALE / (speed + sense * 0.01))

def calculate_full_duration(speed, sense):
    return int(0.2 * SURVIVAL_SCALE / (speed + sense * 0.01))

class Entity:
    def __init__(self, x, y, size, image, hunger_color, initial_speed, sense):
        self.x = x
        self.y = y
        self.size = size
        self.original_image = image
        self.image = image
        self.hunger_color = hunger_color
        self.initial_speed = initial_speed
        self.sense = sense
        self.dx = random.choice([initial_speed, -initial_speed])
        self.dy = random.choice([initial_speed, -initial_speed])
        self.target_x = None
        self.target_y = None
        self.full_ticks = 0
        self.near_death = False
        self.blink_ticks = 0  # Counter for blinking effect
        self.mirrored = False

    def draw(self, screen):
        if self.dx > 0:  # Moving right
            self.image = self.original_image
            self.mirrored = False
        elif self.dx < 0:  # Moving left
            self.image = self.original_image.transpose(Image.FLIP_LEFT_RIGHT)
            self.mirrored = True
        elif self.dy < 0:  # Moving up
            if not self.mirrored:
                self.image = self.original_image.transpose(Image.FLIP_LEFT_RIGHT)
                self.mirrored = True
        elif self.dy > 0:  # Moving down
            self.image = self.original_image
            self.mirrored = False

        surface = pil_to_surface(self.image)
        rect = surface.get_rect(center=(int(self.x), int(self.y)))

        if not self.is_full() and not isinstance(self, Decomposer):
            if (self.blink_ticks // 10) % 2 == 0:
                self.image = colorize(self.original_image, self.hunger_color)  # Hunger color
            else:
                self.image = self.original_image
            self.blink_ticks += 1
        elif self.near_death and not isinstance(self, Decomposer):
            if (self.blink_ticks // 10) % 2 == 0:
                self.image = colorize(self.original_image, 0)  # Red color for near death
            else:
                self.image = self.original_image
            self.blink_ticks += 1
        else:
            self.image = self.original_image
            self.blink_ticks = 0

        screen.blit(surface, rect.topleft)

    def bounce_off_edges(self):
        if self.x - self.size <= EDGE_WIDTH or self.x + self.size >= SCREEN_WIDTH - EDGE_WIDTH:
            self.dx = -self.dx
            self.x = max(min(self.x, SCREEN_WIDTH - EDGE_WIDTH - self.size), EDGE_WIDTH + self.size)
        if self.y - self.size <= EDGE_WIDTH or self.y + self.size >= SCREEN_HEIGHT - EDGE_WIDTH:
            self.dy = -self.dy
            self.y = max(min(self.y, SCREEN_HEIGHT - EDGE_WIDTH - self.size), EDGE_WIDTH + self.size)

    def move_towards(self, target_x, target_y, speed):
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < speed:  # If we're close enough to the target, snap to it
            self.x = target_x
            self.y = target_y
        else:
            dx = dx / dist * speed
            dy = dy / dist * speed
            self.x += dx
            self.y += dy
        self.dx = dx
        self.dy = dy
        self.bounce_off_edges()

    def is_full(self):
        if self.full_ticks > 0:
            self.full_ticks -= 1
            return True
        return False

    def set_full(self):
        self.full_ticks = self.full_duration

    def update_near_death(self):
        if self.time_since_last_meal > 0.9 * self.survival_time:
            self.near_death = True

    def increment_hunger(self):
        self.time_since_last_meal += 1
        self.update_near_death()
        if self.time_since_last_meal > self.survival_time:
            return True
        return False

    def reset_hunger(self):
        self.time_since_last_meal = 0
        self.near_death = False


# Define organism classes
class Carnivore(Entity):
    def __init__(self, x, y, size, image, hunger_color, chase_speed, random_speed, sense):
        super().__init__(x, y, size, image, hunger_color, random_speed, sense)
        self.chase_speed = chase_speed
        self.random_speed = random_speed
        self.survival_time = calculate_survival_time(chase_speed, sense)
        self.full_duration = calculate_full_duration(chase_speed, sense)
        self.time_since_last_meal = 0

    def move_randomly(self):
        if self.target_x is None or self.target_y is None or (self.x == self.target_x and self.y == self.target_y):
            self.set_random_target()
        self.move_towards(self.target_x, self.target_y, self.random_speed)

    def set_random_target(self):
        self.target_x = random.randint(EDGE_WIDTH + self.size, SCREEN_WIDTH - EDGE_WIDTH - self.size)
        self.target_y = random.randint(EDGE_WIDTH + self.size, SCREEN_HEIGHT - EDGE_WIDTH - self.size)

    def chase(self, target_x, target_y):
        self.move_towards(target_x, target_y, self.chase_speed)

    def is_full(self):
        if self.full_ticks > 0:
            self.full_ticks -= 1
            return True
        return False

    def set_full(self):
        self.full_ticks = self.full_duration

    def update_near_death(self):
        if self.time_since_last_meal > 0.9 * self.survival_time:
            self.near_death = True

    def increment_hunger(self):
        self.time_since_last_meal += 1
        self.update_near_death()
        if self.time_since_last_meal > self.survival_time:
            return True
        return False

    def reset_hunger(self):
        self.time_since_last_meal = 0
        self.near_death = False

    def try_reproduce(self):
        if random.random() < CARNIVORE_REPRODUCTION_CHANCE and len(carnivores) < carnivore_cap:
            x, y = place_near_parent(self.x, self.y, CARNIVORE_SIZE)
            carnivores.append(Carnivore(x, y, CARNIVORE_SIZE, carnivore_image, CARNIVORE_HUNGER_COLOR,
                                        self.chase_speed,
                                        self.random_speed,
                                        self.sense))

class Herbivore(Entity):
    def __init__(self, x, y, size, image, hunger_color, speed, sense):
        super().__init__(x, y, size, image, hunger_color, speed, sense)
        self.survival_time = calculate_survival_time(speed, sense)
        self.full_duration = calculate_full_duration(speed, sense)
        self.time_since_last_meal = 0

    def move_randomly(self):
        if self.target_x is None or self.target_y is None or (self.x == self.target_x and self.y == self.target_y):
            self.set_random_target()
        self.move_towards(self.target_x, self.target_y, self.initial_speed)

    def set_random_target(self):
        self.target_x = random.randint(EDGE_WIDTH + self.size, SCREEN_WIDTH - EDGE_WIDTH - self.size)
        self.target_y = random.randint(EDGE_WIDTH + self.size, SCREEN_HEIGHT - EDGE_WIDTH - self.size)

    def move_away_from(self, predator_x, predator_y, run_away_speed):
        dx = self.x - predator_x
        dy = self.y - predator_y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist != 0:
            dx = dx / dist * run_away_speed
            dy = dy / dist * run_away_speed
            self.x += dx
            self.y += dy
            self.bounce_off_edges()

    def is_full(self):
        if self.full_ticks > 0:
            self.full_ticks -= 1
            return True
        return False

    def set_full(self):
        self.full_ticks = self.full_duration

    def update_near_death(self):
        if self.time_since_last_meal > 0.9 * self.survival_time:
            self.near_death = True

    def increment_hunger(self):
        self.time_since_last_meal += 1
        self.update_near_death()
        if self.time_since_last_meal > self.survival_time:
            return True
        return False

    def reset_hunger(self):
        self.time_since_last_meal = 0
        self.near_death = False

class Omnivore(Entity):
    def __init__(self, x, y, size, image, hunger_color, chase_speed, random_speed, sense):
        super().__init__(x, y, size, image, hunger_color, random_speed, sense)
        self.chase_speed = chase_speed
        self.random_speed = random_speed
        self.survival_time = calculate_survival_time(chase_speed, sense)
        self.full_duration_meat = calculate_full_duration(chase_speed, sense) * OMNIVORE_MEAT_FULLNESS_MULTIPLIER
        self.full_duration_any = calculate_full_duration(chase_speed, sense) * OMNIVORE_ANY_FULLNESS_MULTIPLIER
        self.time_since_last_meal = 0

    def move_randomly(self):
        if self.target_x is None or self.target_y is None or (self.x == self.target_x and self.y == self.target_y):
            self.set_random_target()
        self.move_towards(self.target_x, self.target_y, self.random_speed)

    def set_random_target(self):
        self.target_x = random.randint(EDGE_WIDTH + self.size, SCREEN_WIDTH - EDGE_WIDTH - self.size)
        self.target_y = random.randint(EDGE_WIDTH + self.size, SCREEN_HEIGHT - EDGE_WIDTH - self.size)

    def chase(self, target_x, target_y):
        self.move_towards(target_x, target_y, self.chase_speed)

    def is_full(self):
        if self.full_ticks > 0:
            self.full_ticks -= 1
            return True
        return False

    def set_full_meat(self):
        self.full_ticks = self.full_duration_meat

    def set_full_any(self):
        self.full_ticks = self.full_duration_any

    def update_near_death(self):
        if self.time_since_last_meal > 0.9 * self.survival_time:
            self.near_death = True

    def increment_hunger(self):
        self.time_since_last_meal += 1
        self.update_near_death()
        if self.time_since_last_meal > self.survival_time:
            return True
        return False

    def reset_hunger(self):
        self.time_since_last_meal = 0
        self.near_death = False

class DeadOrganism(Entity):
    def __init__(self, x, y, size, original_image):
        # Split the image into RGB and alpha channels
        r, g, b, alpha = original_image.split()
        
        # Convert the RGB channels to greyscale
        greyscale = ImageOps.grayscale(original_image).convert("RGB")
        
        # Combine the greyscale RGB with the original alpha channel
        greyscale_image = Image.merge("RGBA", (greyscale.split()[0], greyscale.split()[1], greyscale.split()[2], alpha))
        
        super().__init__(x, y, size, greyscale_image, 0, 0, 0)  # No movement, just greyscale
        self.image = greyscale_image  # Greyscale image
    
    def is_full(self):
        return True

class Decomposer(Entity):
    def __init__(self, x, y, size, image, hunger_color, speed, sense):
        super().__init__(x, y, size, image, hunger_color, speed, sense)
        self.time_since_last_meal = 0
        self.eating_duration = 0

    def move_randomly(self):
        if self.target_x is None or self.target_y is None or (self.x == self.target_x and self.y == self.target_y):
            self.set_random_target()
        self.move_towards(self.target_x, self.target_y, self.initial_speed)

    def set_random_target(self):
        self.target_x = random.randint(EDGE_WIDTH + self.size, SCREEN_WIDTH - EDGE_WIDTH - self.size)
        self.target_y = random.randint(EDGE_WIDTH + self.size, SCREEN_HEIGHT - EDGE_WIDTH - self.size)

    def move_towards_dead(self, target_x, target_y):
        self.move_towards(target_x, target_y, self.initial_speed)

    def reset_hunger(self):
        self.time_since_last_meal = 0

    def is_eating(self):
        if self.eating_duration > 0:
            self.eating_duration -= 1
            return True
        return False
    
    def is_full(self):
        return False

    def start_eating(self):
        self.eating_duration = DECOMPOSER_EATING_DURATION

class Fruit(Entity):
    def __init__(self, x, y, size, image, reproduction_rate, reproduction_chance):
        super().__init__(x, y, size, image, 0, 0, 0)
        self.reproduction_rate = reproduction_rate
        self.reproduction_chance = reproduction_chance
        self.ticks = 0

    def reproduce(self, fruits):
        self.ticks += 1
        if self.ticks >= self.reproduction_rate and len(fruits) < MAX_FRUITS:
            self.ticks = 0
            if random.random() < self.reproduction_chance:
                new_fruit_x = self.x + random.randint(-10, 10)
                new_fruit_y = self.y + random.randint(-10, 10)
                new_fruit_x = max(EDGE_WIDTH + self.size, min(SCREEN_WIDTH - EDGE_WIDTH - self.size, new_fruit_x))
                new_fruit_y = max(EDGE_WIDTH + self.size, min(SCREEN_HEIGHT - EDGE_WIDTH - self.size, new_fruit_y))
                fruits.append(Fruit(new_fruit_x, new_fruit_y, self.size, self.image, self.reproduction_rate, self.reproduction_chance))
    
    def is_full(self):
        return True

def place_herbivore():
    x = random.randint(EDGE_WIDTH + HERBIVORE_SIZE, SCREEN_WIDTH - EDGE_WIDTH - HERBIVORE_SIZE)
    y = random.randint(EDGE_WIDTH + HERBIVORE_SIZE, SCREEN_HEIGHT - EDGE_WIDTH - HERBIVORE_SIZE)
    return x, y

def place_fruit():
    x = random.randint(EDGE_WIDTH + FRUIT_SIZE, SCREEN_WIDTH - EDGE_WIDTH - FRUIT_SIZE)
    y = random.randint(EDGE_WIDTH + FRUIT_SIZE, SCREEN_HEIGHT - EDGE_WIDTH - FRUIT_SIZE)
    return x, y

def place_omnivore():
    x = random.randint(EDGE_WIDTH + OMNIVORE_SIZE, SCREEN_WIDTH - EDGE_WIDTH - OMNIVORE_SIZE)
    y = random.randint(EDGE_WIDTH + OMNIVORE_SIZE, SCREEN_HEIGHT - EDGE_WIDTH - OMNIVORE_SIZE)
    return x, y

def place_decomposer():
    x = random.randint(EDGE_WIDTH + DECOMPOSER_SIZE, SCREEN_WIDTH - EDGE_WIDTH - DECOMPOSER_SIZE)
    y = random.randint(EDGE_WIDTH + DECOMPOSER_SIZE, SCREEN_HEIGHT - EDGE_WIDTH - DECOMPOSER_SIZE)
    return x, y

def calculate_population_cap(territory_requirement, area):
    return area // territory_requirement

def random_with_std(mean, stddev):
    return max(0.1, random.gauss(mean, stddev))

def place_near_parent(parent_x, parent_y, size):
    x = parent_x + random.randint(-10, 10)
    y = parent_y + random.randint(-10, 10)
    x = max(EDGE_WIDTH + size, min(SCREEN_WIDTH - EDGE_WIDTH - size, x))
    y = max(EDGE_WIDTH + size, min(SCREEN_HEIGHT - EDGE_WIDTH - size, y))
    return x, y

# Create entities with zróżnicowanie
carnivores = [Carnivore(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, CARNIVORE_SIZE, carnivore_image, CARNIVORE_HUNGER_COLOR,
                        random_with_std(CARNIVORE_CHASE_SPEED_MEAN, TRAIT_SPREAD),
                        random_with_std(CARNIVORE_RANDOM_SPEED_MEAN, TRAIT_SPREAD * CARNIVORE_RANDOM_SPEED_MEAN),
                        random_with_std(CARNIVORE_PREY_DETECTION_DISTANCE, TRAIT_SPREAD * CARNIVORE_PREY_DETECTION_DISTANCE)) for _ in range(NUM_CARNIVORES)]

herbivores = [Herbivore(*place_herbivore(), HERBIVORE_SIZE, herbivore_image, HERBIVORE_HUNGER_COLOR,
                        random_with_std(HERBIVORE_RANDOM_SPEED_MEAN, TRAIT_SPREAD * HERBIVORE_RANDOM_SPEED_MEAN),
                        random_with_std(HERBIVORE_FRUIT_DETECTION_DISTANCE, TRAIT_SPREAD * HERBIVORE_FRUIT_DETECTION_DISTANCE)) for _ in range(NUM_HERBIVORES)]

omnivores = [Omnivore(*place_omnivore(), OMNIVORE_SIZE, omnivore_image, OMNIVORE_HUNGER_COLOR,
                      random_with_std(OMNIVORE_CHASE_SPEED_MEAN, TRAIT_SPREAD),
                      random_with_std(OMNIVORE_RANDOM_SPEED_MEAN, TRAIT_SPREAD * OMNIVORE_RANDOM_SPEED_MEAN),
                      random_with_std(OMNIVORE_PREY_DETECTION_DISTANCE, TRAIT_SPREAD * OMNIVORE_PREY_DETECTION_DISTANCE)) for _ in range(NUM_OMNIVORES)]

decomposers = [Decomposer(*place_decomposer(), DECOMPOSER_SIZE, decomposer_image, DECOMPOSER_HUNGER_COLOR,
                          DECOMPOSER_SPEED, DECOMPOSER_SENSE) for _ in range(NUM_DECOMPOSERS)]

fruits = [Fruit(*place_fruit(), FRUIT_SIZE, fruit_image, FRUIT_REPRODUCTION_RATE, FRUIT_REPRODUCTION_CHANCE) for _ in range(NUM_FRUITS)]

# Population caps
carnivore_cap = calculate_population_cap(CARNIVORE_TERRITORY_REQUIREMENT, SCREEN_WIDTH * SCREEN_HEIGHT)

# Create the screen
screen = pygame.display.set_mode((TOTAL_WIDTH, SCREEN_HEIGHT + GUI_HEIGHT))
pygame.display.set_caption('Symulacja mięsożerców i roślinożerców')

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Initialize matplotlib figures
fig, ax = plt.subplots(figsize=(PLOT_WIDTH / 100, SCREEN_HEIGHT / 100))
ax.set_title('Początkowa prędkość vs Zasięg wykrywania jedzenia')
ax.set_xlabel('Początkowa prędkość')
ax.set_ylabel('Zasięg wykrywania jedzenia')

def update_plot(ax, carnivores, herbivores, omnivores, decomposers):
    ax.clear()
    ax.set_title('Początkowa prędkość vs Zasięg wykrywania jedzenia')
    ax.set_xlabel('Początkowa prędkość')
    ax.set_ylabel('Zasięg wykrywania jedzenia')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, max(DECOMPOSER_SENSE, OMNIVORE_PREY_DETECTION_DISTANCE, CARNIVORE_PREY_DETECTION_DISTANCE, HERBIVORE_FRUIT_DETECTION_DISTANCE) * 1.25)

    # Dictionary to count organisms with the same traits
    carnivore_counts = {}
    herbivore_counts = {}
    omnivore_counts = {}
    decomposer_counts = {}

    for carnivore in carnivores:
        key = (round(carnivore.chase_speed, 1), round(carnivore.sense, 1))
        if key not in carnivore_counts:
            carnivore_counts[key] = 0
        carnivore_counts[key] += 1

    for herbivore in herbivores:
        key = (round(herbivore.initial_speed, 1), round(herbivore.sense, 1))
        if key not in herbivore_counts:
            herbivore_counts[key] = 0
        herbivore_counts[key] += 1

    for omnivore in omnivores:
        key = (round(omnivore.chase_speed, 1), round(omnivore.sense, 1))
        if key not in omnivore_counts:
            omnivore_counts[key] = 0
        omnivore_counts[key] += 1

    for decomposer in decomposers:
        key = (round(decomposer.initial_speed, 1), round(decomposer.sense, 1))
        if key not in decomposer_counts:
            decomposer_counts[key] = 0
        decomposer_counts[key] += 1

    # Plot carnivores
    for (speed, sense), count in carnivore_counts.items():
        ax.scatter(speed, sense, c='green', s=20 + count * 20, label='Mięsożercy' if count == 1 else "")
        ax.text(speed + 0.25 + min(count * 0.01, 0.8), sense, str(count), fontsize=9, ha='left', va='bottom')

    # Plot herbivores
    for (speed, sense), count in herbivore_counts.items():
        ax.scatter(speed, sense, c='blue', s=20 + count * 20, label='Roślinożercy' if count == 1 else "")
        ax.text(speed + 0.25 + min(count * 0.01, 0.8), sense, str(count), fontsize=9, ha='left', va='bottom')

    # Plot omnivores
    for (speed, sense), count in omnivore_counts.items():
        ax.scatter(speed, sense, c='orange', s=20 + count * 20, label='Wszystkożercy' if count == 1 else "")
        ax.text(speed + 0.25 + min(count * 0.01, 0.8), sense, str(count), fontsize=9, ha='left', va='bottom')

    # Plot decomposers
    for (speed, sense), count in decomposer_counts.items():
        ax.scatter(speed, sense, c='brown', s=20 + count * 20, label='Detrytusożercy' if count == 1 else "")
        ax.text(speed + 0.25 + min(count * 0.01, 0.8), sense, str(count), fontsize=9, ha='left', va='bottom')

    # Fixed legend
    legend_labels = ['Mięsożercy', 'Roślinożercy', 'Wszystkożercy', 'Detrytusożercy']
    colors = ['green', 'blue', 'orange', 'brown']
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]
    ax.legend(handles, legend_labels, loc='upper right')

# Initialize pygame_gui
ui_manager = pygame_gui.UIManager((TOTAL_WIDTH, SCREEN_HEIGHT + GUI_HEIGHT))

# Dropdown menu for organism class selection
organism_dropdown = pygame_gui.elements.UIDropDownMenu(
    options_list=['Mięsożerca', 'Roślinożerca', 'Wszystkożerca'],
    starting_option='Roślinożerca',
    relative_rect=pygame.Rect((10, SCREEN_HEIGHT + 10), (200, 50)),
    manager=ui_manager
)

# Sliders for speed and food detection range
speed_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect((220, SCREEN_HEIGHT + 10), (200, 50)),
    start_value=HERBIVORE_RANDOM_SPEED_MEAN,
    value_range=(0.1, 20.0),
    manager=ui_manager
)

speed_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((220, SCREEN_HEIGHT + 60), (200, 40)),
    text=f'Prędkość: {HERBIVORE_RANDOM_SPEED_MEAN:.2f}',
    manager=ui_manager
)

sense_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect((430, SCREEN_HEIGHT + 10), (200, 50)),
    start_value=HERBIVORE_FRUIT_DETECTION_DISTANCE,
    value_range=(50, 600),
    manager=ui_manager
)

sense_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((430, SCREEN_HEIGHT + 60), (200, 40)),
    text=f'Zmysł: {HERBIVORE_FRUIT_DETECTION_DISTANCE:.2f}',
    manager=ui_manager
)

# Buttons for adding and removing organisms
add_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((640, SCREEN_HEIGHT + 10), (150, 50)),
    text='+ Dodaj',
    manager=ui_manager,
    object_id=pygame_gui.core.ObjectID(class_id="@button")
)
add_button.text_colour = pygame.Color(255, 255, 255)  # Set button text to white

remove_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((800, SCREEN_HEIGHT + 10), (150, 50)),
    text='- Usuń',
    manager=ui_manager,
    object_id=pygame_gui.core.ObjectID(class_id="@button")
)
remove_button.text_colour = pygame.Color(255, 255, 255)  # Set button text to white

# Add a replay button
replay_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((TOTAL_WIDTH - 160, SCREEN_HEIGHT + 110), (150, 50)),
    text='Od nowa',
    manager=ui_manager,
    object_id=pygame_gui.core.ObjectID(class_id="@button")
)
replay_button.text_colour = pygame.Color(255, 255, 255)  # Set button text to white

# Timer to track how long there are at least two of the three classes alive
start_time = time.time()
timer_running = True

# Real-time display for the timer
timer_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((TOTAL_WIDTH - 220, 10), (200, 50)),
    text='Czas: 0.00 s',
    manager=ui_manager
)


# Helper functions to update organism parameters
def update_herbivore_parameters(delta_speed, delta_sense):
    for herbivore in herbivores:
        herbivore.initial_speed = max(0.1, herbivore.initial_speed + delta_speed)
        herbivore.sense = max(0.1, herbivore.sense + delta_sense)
        herbivore.survival_time = calculate_survival_time(herbivore.initial_speed, herbivore.sense)
    speed_label.set_text(f'Prędkość: {speed_slider.get_current_value():.2f}')
    sense_label.set_text(f'Zmysł: {sense_slider.get_current_value():.2f}')

def update_carnivore_parameters(delta_speed, delta_sense):
    for carnivore in carnivores:
        carnivore.random_speed = max(0.1, carnivore.random_speed + delta_speed)
        carnivore.chase_speed = carnivore.random_speed + 2
        carnivore.sense = max(0.1, carnivore.sense + delta_sense)
        carnivore.survival_time = calculate_survival_time(carnivore.random_speed, carnivore.sense)
    speed_label.set_text(f'Prędkość: {speed_slider.get_current_value():.2f}')
    sense_label.set_text(f'Zmysł: {sense_slider.get_current_value():.2f}')

def update_omnivore_parameters(delta_speed, delta_sense):
    for omnivore in omnivores:
        omnivore.random_speed = max(0.1, omnivore.random_speed + delta_speed)
        omnivore.chase_speed = omnivore.random_speed + 2
        omnivore.sense = max(0.1, omnivore.sense + delta_sense)
        omnivore.survival_time = calculate_survival_time(omnivore.random_speed, omnivore.sense)
    speed_label.set_text(f'Prędkość: {speed_slider.get_current_value():.2f}')
    sense_label.set_text(f'Zmysł: {sense_slider.get_current_value():.2f}')

def reset_simulation():
    global carnivores, herbivores, omnivores, decomposers, fruits, dead_organisms, timer_running, start_time
    carnivores = [Carnivore(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, CARNIVORE_SIZE, carnivore_image, CARNIVORE_HUNGER_COLOR,
                            random_with_std(CARNIVORE_CHASE_SPEED_MEAN, TRAIT_SPREAD),
                            random_with_std(CARNIVORE_RANDOM_SPEED_MEAN, TRAIT_SPREAD * CARNIVORE_RANDOM_SPEED_MEAN),
                            random_with_std(CARNIVORE_PREY_DETECTION_DISTANCE, TRAIT_SPREAD * CARNIVORE_PREY_DETECTION_DISTANCE)) for _ in range(NUM_CARNIVORES)]
    
    herbivores = [Herbivore(*place_herbivore(), HERBIVORE_SIZE, herbivore_image, HERBIVORE_HUNGER_COLOR,
                            random_with_std(HERBIVORE_RANDOM_SPEED_MEAN, TRAIT_SPREAD * HERBIVORE_RANDOM_SPEED_MEAN),
                            random_with_std(HERBIVORE_FRUIT_DETECTION_DISTANCE, TRAIT_SPREAD * HERBIVORE_FRUIT_DETECTION_DISTANCE)) for _ in range(NUM_HERBIVORES)]
    
    omnivores = [Omnivore(*place_omnivore(), OMNIVORE_SIZE, omnivore_image, OMNIVORE_HUNGER_COLOR,
                          random_with_std(OMNIVORE_CHASE_SPEED_MEAN, TRAIT_SPREAD),
                          random_with_std(OMNIVORE_RANDOM_SPEED_MEAN, TRAIT_SPREAD * OMNIVORE_RANDOM_SPEED_MEAN),
                          random_with_std(OMNIVORE_PREY_DETECTION_DISTANCE, TRAIT_SPREAD * OMNIVORE_PREY_DETECTION_DISTANCE)) for _ in range(NUM_OMNIVORES)]
    
    decomposers = [Decomposer(*place_decomposer(), DECOMPOSER_SIZE, decomposer_image, DECOMPOSER_HUNGER_COLOR,
                              DECOMPOSER_SPEED, DECOMPOSER_SENSE) for _ in range(NUM_DECOMPOSERS)]
    
    fruits = [Fruit(*place_fruit(), FRUIT_SIZE, fruit_image, FRUIT_REPRODUCTION_RATE, FRUIT_REPRODUCTION_CHANCE) for _ in range(NUM_FRUITS)]
    
    dead_organisms = []
    timer_running = True
    start_time = time.time()

# Main loop
dead_organisms = []
selected_class = 'Roślinożerca'
last_speed_value = HERBIVORE_RANDOM_SPEED_MEAN
last_sense_value = HERBIVORE_FRUIT_DETECTION_DISTANCE

running = True
while running:
    time_delta = clock.tick(60 * SPEED_MULTIPLIER) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        ui_manager.process_events(event)

        if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == organism_dropdown:
                selected_class = event.text
                if selected_class == 'Roślinożerca':
                    speed_slider.set_current_value(HERBIVORE_RANDOM_SPEED_MEAN)
                    sense_slider.set_current_value(HERBIVORE_FRUIT_DETECTION_DISTANCE)
                    speed_label.set_text(f'Prędkość: {HERBIVORE_RANDOM_SPEED_MEAN:.2f}')
                    sense_label.set_text(f'Zmysł: {HERBIVORE_FRUIT_DETECTION_DISTANCE:.2f}')
                elif selected_class == 'Mięsożerca':
                    speed_slider.set_current_value(CARNIVORE_RANDOM_SPEED_MEAN)
                    sense_slider.set_current_value(CARNIVORE_PREY_DETECTION_DISTANCE)
                    speed_label.set_text(f'Prędkość: {CARNIVORE_RANDOM_SPEED_MEAN:.2f}')
                    sense_label.set_text(f'Zmysł: {CARNIVORE_PREY_DETECTION_DISTANCE:.2f}')
                elif selected_class == 'Wszystkożerca':
                    speed_slider.set_current_value(OMNIVORE_RANDOM_SPEED_MEAN)
                    sense_slider.set_current_value(OMNIVORE_PREY_DETECTION_DISTANCE)
                    speed_label.set_text(f'Prędkość: {OMNIVORE_RANDOM_SPEED_MEAN:.2f}')
                    sense_label.set_text(f'Zmysł: {OMNIVORE_PREY_DETECTION_DISTANCE:.2f}')
                last_speed_value = speed_slider.get_current_value()
                last_sense_value = sense_slider.get_current_value()

        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            delta_speed = speed_slider.get_current_value() - last_speed_value
            delta_sense = sense_slider.get_current_value() - last_sense_value
            if event.ui_element == speed_slider:
                if selected_class == 'Roślinożerca':
                    update_herbivore_parameters(delta_speed, 0)
                elif selected_class == 'Mięsożerca':
                    update_carnivore_parameters(delta_speed, 0)
                elif selected_class == 'Wszystkożerca':
                    update_omnivore_parameters(delta_speed, 0)
                last_speed_value = speed_slider.get_current_value()
            if event.ui_element == sense_slider:
                if selected_class == 'Roślinożerca':
                    update_herbivore_parameters(0, delta_sense)
                elif selected_class == 'Mięsożerca':
                    update_carnivore_parameters(0, delta_sense)
                elif selected_class == 'Wszystkożerca':
                    update_omnivore_parameters(0, delta_sense)
                last_sense_value = sense_slider.get_current_value()

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == add_button:
                if selected_class == 'Roślinożerca':
                    x, y = place_herbivore()
                    herbivores.append(Herbivore(x, y, HERBIVORE_SIZE, herbivore_image, HERBIVORE_HUNGER_COLOR,
                                                speed_slider.get_current_value(),
                                                sense_slider.get_current_value()))
                elif selected_class == 'Mięsożerca':
                    x, y = place_near_parent(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), CARNIVORE_SIZE)
                    carnivores.append(Carnivore(x, y, CARNIVORE_SIZE, carnivore_image, CARNIVORE_HUNGER_COLOR,
                                                speed_slider.get_current_value(),
                                                speed_slider.get_current_value(),
                                                sense_slider.get_current_value()))
                elif selected_class == 'Wszystkożerca':
                    x, y = place_omnivore()
                    omnivores.append(Omnivore(x, y, OMNIVORE_SIZE, omnivore_image, OMNIVORE_HUNGER_COLOR,
                                              speed_slider.get_current_value(),
                                              speed_slider.get_current_value(),
                                              sense_slider.get_current_value()))

            if event.ui_element == remove_button:
                if selected_class == 'Roślinożerca' and herbivores:
                    herbivores.pop()
                elif selected_class == 'Mięsożerca' and carnivores:
                    carnivores.pop()
                elif selected_class == 'Wszystkożerca' and omnivores:
                    omnivores.pop()

            if event.ui_element == replay_button:
                reset_simulation()

    ui_manager.update(time_delta)

    # Update positions and check interactions
    # Find the closest herbivore for each carnivore
    for carnivore in carnivores:
        if carnivore.is_full():
            carnivore.move_randomly()
        else:
            closest_herbivore = None
            min_distance = carnivore.sense
            for herbivore in herbivores:
                distance = math.sqrt((herbivore.x - carnivore.x) ** 2 + (herbivore.y - carnivore.y) ** 2)
                if distance < min_distance:
                    closest_herbivore = herbivore
                    min_distance = distance

            if closest_herbivore:
                carnivore.chase(closest_herbivore.x, closest_herbivore.y)

                # Check if carnivore has reached the herbivore
                if abs(carnivore.x - closest_herbivore.x) < CARNIVORE_SIZE and abs(carnivore.y - closest_herbivore.y) < CARNIVORE_SIZE:
                    herbivores.remove(closest_herbivore)  # Carnivore eats the herbivore
                    dead_organisms.append(DeadOrganism(closest_herbivore.x, closest_herbivore.y, HERBIVORE_SIZE, closest_herbivore.original_image))
                    carnivore.reset_hunger()  # Reset hunger timer
                    carnivore.set_full()  # Set full state

                    # Carnivore reproduction attempt
                    carnivore.try_reproduce()  # Try to reproduce only once after eating

            else:
                carnivore.move_randomly()

        # Increment carnivore hunger and check if it starves
        if carnivore.increment_hunger():
            carnivores.remove(carnivore)
            dead_organisms.append(DeadOrganism(carnivore.x, carnivore.y, CARNIVORE_SIZE, carnivore_image))

    # Move herbivores and check interactions with fruits
    for herbivore in herbivores:
        if herbivore.is_full():
            herbivore.move_randomly()
        else:
            closest_fruit = None
            min_distance = herbivore.sense
            for fruit in fruits:
                distance = math.sqrt((fruit.x - herbivore.x) ** 2 + (fruit.y - herbivore.y) ** 2)
                if distance < min_distance:
                    closest_fruit = fruit
                    min_distance = distance

            if closest_fruit:
                herbivore.move_towards(closest_fruit.x, closest_fruit.y, herbivore.initial_speed)
                
                # Check if herbivore has reached the fruit
                if abs(herbivore.x - closest_fruit.x) < HERBIVORE_SIZE and abs(herbivore.y - closest_fruit.y) < HERBIVORE_SIZE:
                    fruits.remove(closest_fruit)  # Herbivore eats the fruit
                    herbivore.reset_hunger()  # Reset hunger timer
                    herbivore.set_full()  # Set full state

                    # Herbivore reproduction chance
                    if random.random() < HERBIVORE_REPRODUCTION_CHANCE:
                        x, y = place_near_parent(herbivore.x, herbivore.y, HERBIVORE_SIZE)
                        herbivores.append(Herbivore(x, y, HERBIVORE_SIZE, herbivore_image, HERBIVORE_HUNGER_COLOR,
                                                    herbivore.initial_speed,
                                                    herbivore.sense))
            else:
                herbivore.move_randomly()

            # Check if herbivore needs to run away from carnivore
            if closest_herbivore and herbivore == closest_herbivore:
                herbivore.move_away_from(carnivore.x, carnivore.y, HERBIVORE_RUN_AWAY_SPEED_MEAN)

        # Increment herbivore hunger and check if it starves
        if herbivore.increment_hunger():
            herbivores.remove(herbivore)
            dead_organisms.append(DeadOrganism(herbivore.x, herbivore.y, HERBIVORE_SIZE, herbivore_image))

    # Move omnivores and check interactions with other entities
    for omnivore in omnivores:
        if omnivore.is_full():
            omnivore.move_randomly()
        else:
            closest_prey = None
            min_distance = omnivore.sense
            for entity_list in [herbivores, carnivores, fruits]:
                for entity in entity_list:
                    distance = math.sqrt((entity.x - omnivore.x) ** 2 + (entity.y - omnivore.y) ** 2)
                    if distance < min_distance:
                        closest_prey = entity
                        min_distance = distance

            if closest_prey:
                omnivore.chase(closest_prey.x, closest_prey.y)

                # Check if omnivore has reached the prey
                if abs(omnivore.x - closest_prey.x) < OMNIVORE_SIZE and abs(omnivore.y - closest_prey.y) < OMNIVORE_SIZE:
                    if isinstance(closest_prey, Fruit):
                        fruits.remove(closest_prey)  # Omnivore eats the fruit
                        omnivore.set_full_any()  # Set full state (any food)
                    else:
                        if isinstance(closest_prey, Carnivore):
                            carnivores.remove(closest_prey)  # Omnivore eats the carnivore
                        elif isinstance(closest_prey, Herbivore):
                            herbivores.remove(closest_prey)  # Omnivore eats the herbivore
                        omnivore.set_full_meat()  # Set full state (meat)
                        dead_organisms.append(DeadOrganism(closest_prey.x, closest_prey.y, closest_prey.size, closest_prey.original_image))
                    omnivore.reset_hunger()  # Reset hunger timer

                    # Omnivore reproduction chance
                    if random.random() < OMNIVORE_REPRODUCTION_CHANCE:
                        x, y = place_near_parent(omnivore.x, omnivore.y, OMNIVORE_SIZE)
                        omnivores.append(Omnivore(x, y, OMNIVORE_SIZE, omnivore_image, OMNIVORE_HUNGER_COLOR,
                                                  omnivore.chase_speed,
                                                  omnivore.random_speed,
                                                  omnivore.sense))
            else:
                omnivore.move_randomly()

        # Increment omnivore hunger and check if it starves
        if omnivore.increment_hunger():
            omnivores.remove(omnivore)
            dead_organisms.append(DeadOrganism(omnivore.x, omnivore.y, OMNIVORE_SIZE, omnivore_image))

    # Decomposers moving and eating dead organisms
    for decomposer in decomposers:
        if decomposer.is_eating():
            continue  # Decomposer is currently eating

        closest_dead = None
        min_distance = decomposer.sense
        for dead in dead_organisms:
            distance = math.sqrt((dead.x - decomposer.x) ** 2 + (dead.y - decomposer.y) ** 2)
            if distance < min_distance:
                closest_dead = dead
                min_distance = distance

        if closest_dead:
            decomposer.move_towards_dead(closest_dead.x, closest_dead.y)

            # Check if decomposer has reached the dead organism
            if abs(decomposer.x - closest_dead.x) < DECOMPOSER_SIZE and abs(decomposer.y - closest_dead.y) < DECOMPOSER_SIZE:
                dead_organisms.remove(closest_dead)  # Decomposer eats the dead organism
                decomposer.start_eating()  # Start eating process
                decomposer.reset_hunger()  # Reset hunger timer

                # Decomposer reproduction chance
                if random.random() < DECOMPOSER_REPRODUCTION_CHANCE:
                    x, y = place_near_parent(decomposer.x, decomposer.y, DECOMPOSER_SIZE)
                    decomposers.append(Decomposer(x, y, DECOMPOSER_SIZE, decomposer_image, DECOMPOSER_HUNGER_COLOR,
                                                  decomposer.initial_speed,
                                                  decomposer.sense))

    # Reproduce fruits and occasionally add new random fruits
    if len(fruits) < MAX_FRUITS:
        fruit_density = len(fruits) / (SCREEN_WIDTH * SCREEN_HEIGHT)
        for fruit in fruits:
            if fruit_density < FRUIT_DENSITY_THRESHOLD:
                fruit.reproduce(fruits)
        if random.random() < NEW_FRUIT_CHANCE:
            fruits.append(Fruit(*place_fruit(), FRUIT_SIZE, fruit_image, FRUIT_REPRODUCTION_RATE, FRUIT_REPRODUCTION_CHANCE))

    # Ensure minimum number of fruits
    while len(fruits) < MIN_FRUITS:
        fruits.append(Fruit(*place_fruit(), FRUIT_SIZE, fruit_image, FRUIT_REPRODUCTION_RATE, FRUIT_REPRODUCTION_CHANCE))

    # Check if any two of the three classes (herbivores, carnivores, omnivores) are extinct
    if timer_running:
        if sum([len(carnivores) > 0, len(herbivores) > 0, len(omnivores) > 0]) < 2:
            timer_running = False
            elapsed_time = time.time() - start_time
            timer_label.set_text(f'Czas: {elapsed_time:.2f} s')

    if timer_running:
        elapsed_time = time.time() - start_time
        timer_label.set_text(f'Czas: {elapsed_time:.2f} s')

    # Clear the screen
    screen.fill(BACKGROUND_COLOR)

    # Draw the edges
    pygame.draw.rect(screen, EDGE_COLOR, (0, 0, SCREEN_WIDTH, EDGE_WIDTH))  # Top edge
    pygame.draw.rect(screen, EDGE_COLOR, (0, 0, EDGE_WIDTH, SCREEN_HEIGHT))  # Left edge
    pygame.draw.rect(screen, EDGE_COLOR, (0, SCREEN_HEIGHT - EDGE_WIDTH, SCREEN_WIDTH, EDGE_WIDTH))  # Bottom edge
    pygame.draw.rect(screen, EDGE_COLOR, (SCREEN_WIDTH - EDGE_WIDTH, 0, EDGE_WIDTH, SCREEN_HEIGHT))  # Right edge

    # Draw the background image on the main game board
    screen.blit(background_surface, (0, 0))

    # Draw the carnivores, herbivores, omnivores, fruits, dead organisms, and decomposers
    for carnivore in carnivores:
        carnivore.draw(screen)
    for herbivore in herbivores:
        herbivore.draw(screen)
    for omnivore in omnivores:
        omnivore.draw(screen)
    for fruit in fruits:
        fruit.draw(screen)
    for dead in dead_organisms:
        dead.draw(screen)
    for decomposer in decomposers:
        decomposer.draw(screen)

    # Update matplotlib plot
    update_plot(ax, carnivores, herbivores, omnivores, decomposers)

    # Convert plot to surface
    canvas = FigureCanvas(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()  # Convert memoryview to bytes

    plot_surface = pygame.image.frombuffer(raw_data, canvas.get_width_height(), "RGBA")
    plot_surface = pygame.transform.scale(plot_surface, (PLOT_WIDTH, SCREEN_HEIGHT))

    screen.blit(plot_surface, (SCREEN_WIDTH, 0))

    # Render GUI elements
    ui_manager.draw_ui(screen)

    pygame.display.flip()

pygame.quit()
sys.exit()
