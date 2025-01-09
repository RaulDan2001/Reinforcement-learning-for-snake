import pygame
import random
from enum import Enum
from collections import namedtuple 
import numpy as np

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

#initializez modulele pygame
pygame.init()
pygame.font.init()

Point = namedtuple('Point', 'x, y')
font = pygame.font.SysFont('arial', 25, bold=True)
BLOCK_SIZE = 20
SPEED = 30

#CULORI RGB
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
SOFT_YELLOW = (255, 255, 204)
LIGHT_GREEN = (144, 238, 144)

class SnakeEnv(object):
    def __init__(self,width=640, height=480):
        self.width = width
        self.height = height
        #initializez fereastra de afisare
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        #initializez enviorementul
        self.reset()

    def reset(self):
        #initializez starea jocului
        self.direction = Direction.RIGHT

        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        #generez o pozitie random in display dar care sa fie multuplu al BLOCK_SIZE ca sa nu apara mancarea 
        #iesita din grid
        x = random.randint(0, (self.width-BLOCK_SIZE ) //BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.height-BLOCK_SIZE ) //BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_frame(self, action):
        self.frame_iteration += 1
        
        #Colectez input de la jucator
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        #misc sarpele
        
        self._move(action) #actualizez capul
        self.snake.insert(0, self.head)

        #verific daca jocul este gata
        reward = 0
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        #pun mancare noua sau doar misc sarpele
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        #calculez distanta Manhattan pana la mancare
        current_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        previous_distance = abs(self.snake[1].x - self.food.x) + abs(self.snake[1].y - self.food.y)

        #acord reward pentru miscare in directia mancarii
        if current_distance <  previous_distance:
            reward +=1
        else:
            reward -= 0.5

        #penalizare pentru mers in cercuri
        if self.head in self.snake[1:]:
            reward -= 1

        #recompensa negativa pentru fiecare miscare
        reward -= 0.01

        #actualizez interfata pygame si clock
        self._update_ui()
        self.clock.tick(SPEED)
        #returnez daca e gata jocul si scorul
        return reward ,False, self.score
        
    def is_collision(self, point=None):
        if point is None:
            point = self.head
        #verific daca loveste pereti
        if point.x > self.width - BLOCK_SIZE or point.x < 0 or point.y > self.height - BLOCK_SIZE or point.y < 0:
            return True

        #verific daca se loveste pe sine
        #capul mereu va face parte din lista si de aceea incep de la pozitia unu
        if point in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        #generez fundalul
        self.display.fill(SOFT_YELLOW)
        #desenez sarpele
        for point in self.snake:
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, LIGHT_GREEN, pygame.Rect(point.x+4, point.y+4, 12, 12))

        #desenez mancarea
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render(f"SCORE: {self.score}", True, BLUE1)
        self.display.blit(text, [0, 0])
        #actualizez suprafata pe ecran
        pygame.display.flip()

    def _move(self, action):
        #determin directia in functie de actiune
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[idx] #nici o schimbare
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_direction = clock_wise[next_idx] #intoarcere la dreapta
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_direction = clock_wise[next_idx] #intoarcere la stanga



        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
    
        self.head = Point(x, y)




