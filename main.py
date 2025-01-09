import pygame
from agent import Agent
from snake_env import SnakeEnv
from plotter import plot
import customtkinter as ctk

root = ctk.CTk()
root.title("Welcome")
root.geometry('640*480')

def load_model():
    root.withdraw()
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

def train():
    root.withdraw()
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 61
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

def main():
    #definesc butoanele
    buttontrain = ctk.CTkButton(root, text='TRAIN', command=train)
    buttontrain.pack()

    buttonplay = ctk.CTkButton(root, text='PLAY', command=load_model)
    buttonplay.pack()
    
    root.mainloop()

if __name__ == '__main__':
    main() 
