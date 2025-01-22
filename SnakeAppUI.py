from snakeGame import SnakeGame
from snake_env import SnakeEnv
from agent import Agent
from plotter import plot
import customtkinter as ctk
import pygame

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class SnakeAppUI(object):
    def __init__(self, root):
        self.root = root
        self.root.title("Snake Game")
        self.root.geometry("500x300")

        #Label pentru bun venit
        self.label = ctk.CTkLabel(root, text="Welcome to Snake Game AI", font=("Arial", 24, "bold"))
        self.label.pack(pady=20)

        #Buton pentru jucare umana
        self.play_button = ctk.CTkButton(root, text="PLAY", command=self.play_human, font=("Arial", 16))
        self.play_button.pack(pady=10)

        #Buton pentru invatare automata
        self.learn_button = ctk.CTkButton(root, text="WATCH AI", command=self.play_robot, font=("Arial", 16))
        self.learn_button.pack(pady=10)

        #Buton pentru iesire
        self.quit_button = ctk.CTkButton(root, text="QUIT", command=root.quit, font=("Arial", 16))
    

    def play_human(self):
        self.root.withdraw()
        #TODO: cod pentru joc uman
        game = SnakeGame()

        #loop pentru joc
        while True:
            game_over, score = game.play_frame()

            #resetez cand e gata jocul 
            if game_over:
                game.reset()
                print(f"Final score {score}")

        

    def play_robot(self):
        self.root.withdraw()
        self.robot_window = ctk.CTkToplevel(self.root)
        self.robot_window.title("Choose AI Mode")
        self.robot_window.geometry("500x300")

        # Label pentru alegere mod AI
        self.label = ctk.CTkLabel(self.robot_window, text="Choose AI Mode", font=("Arial", 24, "bold"))
        self.label.pack(pady=20)
        
        # Buton pentru model pre-antrenat
        self.pretrained_button = ctk.CTkButton(self.robot_window, text="Pretrained Model", command=self.load_model, font=("Arial", 16))
        self.pretrained_button.pack(pady=10)

        # Buton pentru model antrenat in loc
        self.train_button = ctk.CTkButton(self.robot_window, text="Train Model", command=self.train, font=("Arial", 16))
        self.train_button.pack(pady=10)

    def load_model(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        agent = Agent()
        game = SnakeEnv()

        #incarc modelul cel mai bun
        agent.model.load('model/model.pth')

        while True:
            #calculez starea curenta
            state_old = agent.get_state(game)
        
            #Calculez miscarea potrivita
            final_move = agent.load_get_action(state_old)

            #fac miscarea si calculez starea noua
            reward, game_over, score = game.play_frame(final_move)

            if game_over:
                game.reset()
                agent.n_games += 1

                print(f'Game: {agent.n_games}, Score: {score}, Record: {record}')
        
                plot_scores.append(score)
                total_score += score
                if agent.n_games > 0:
                    mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

    def train(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 45
        agent = Agent()
        game = SnakeEnv() 
        while True:
            #calculez starea curenta
            state_old = agent.get_state(game)

            #calculez miscarea
            final_move = agent.get_action(state_old)

            #efectuez miscarea si calculez starea noua
            reward, game_over, score = game.play_frame(final_move)
            state_new = agent.get_state(game)

            #antrenez memoria scurta a agentului
            agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

            #tin minte acesti parametri
            agent.remeber(state_old, final_move, reward, state_new, game_over)

            if game_over: 
                #antrenez memoria lunga, afisez rezultatele
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                print(f'Game: {agent.n_games}, Score: {score}, Record: {record}')

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

