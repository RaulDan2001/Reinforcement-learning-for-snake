import torch
import random 
import numpy as np
from collections import deque
from snake_env import SnakeEnv, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer


MAX_MEMORY = 100000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent(object):
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomizare
        self.gamma = 0.9 #rata de discount
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(17, 256, 256, 3)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        #creez puncte in jurul capului sarpelui pentru a verifica daca este in pericol
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        #colectez directia in care merge sarpele
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        #distantele Manhattan pana la mancare (normalizate)
        distance_x = (game.food.x - game.head.x) / game.width
        distance_y = (game.food.y - game.head.y) / game.height

        state = [
            #bazat pe directia curenta
            # pericol in fata
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # pericol la drapta
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # pericol la stanga
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            #directia de miscare
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #locatia mancarii
            game.food.x < game.head.x, #este mancare la stanga
            game.food.x > game.head.x, #este mancare la dreapta
            game.food.y < game.head.y, #este mancare in partea de sus
            game.food.y > game.head.y, #este mancare in partea de jos

            #distantele Manhattan 
            distance_x,
            distance_y,
            ]

        #awareness pentru blocurile din jurul sarpelui
        state.extend([
            game.is_collision(Point(head.x - 4*BLOCK_SIZE, head.y)),
            game.is_collision(Point(head.x + 4*BLOCK_SIZE, head.y)),
            game.is_collision(Point(head.x, head.y - 4*BLOCK_SIZE)),
            game.is_collision(Point(head.x, head.y + 4*BLOCK_SIZE)),
            ])

        return np.array(state, dtype=int)

    def remeber(self, state, action, reward, next_state, game_over):
        #vrem sa retinem totul in deque
        self.memory.append((state, action, reward, next_state, game_over)) 

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #lista de perechi
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        #la inceput sunt miscari random 
        #tradeoff: exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def load_get_action(self, state):
        final_move = [0,0,0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move

    


