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

pygame. init()
pygame.font.init()

Point = namedtuple('Point', 'x, y')
font = pygame.font.SysFont('arial', 25, bold=True)
BLOCK_SIZE = 20
SPEED = 10

#CULORI RGB
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
SOFT_YELLOW = (255, 255, 204)
LIGHT_GREEN = (144, 238, 144)

class SnakeGame(object):
    def __init__(self, width= 640, height= 480):
        self. width = width
        self.height = height
        
        #initializez afisarea
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        #initializez starea jocului
        self.direction = Direction.RIGHT

        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()

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

    def _place_food(self):
        #generez o pozitie random in display dar care sa fie multuplu al BLOCK_SIZE ca sa nu apara mancarea 
        #iesita din grid
        x = random.randint(0, (self.width-BLOCK_SIZE ) //BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.height-BLOCK_SIZE ) //BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

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

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x,y)

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

    def play_frame(self):
        #colectez input de la user  
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT: 
                    self.direction = Direction.LEFT 
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT: 
                    self.direction = Direction.RIGHT 
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN: 
                    self.direction = Direction.UP 
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP: 
                    self.direction = Direction.DOWN
        
        #misc sarpele
        self._move(self.direction) #update la cap
        self.snake.insert(0, self.head)

        #check daca jocul este gata
        game_over = False
        if self.is_collision():
            game_over = True
            return game_over, self.score

        #pun mancare sau doar misc sarpele
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        #update pygame UI si clock
        self._update_ui()
        self.clock.tick(SPEED)

        #return daca e gata jocul si scorul
        game_over = False
        return game_over, self.score



