import pygame
from agent import Agent
from snake_env import SnakeEnv
from plotter import plot
import customtkinter as ctk
from SnakeAppUI import SnakeAppUI


def main():
    root = ctk.CTk()
    app = SnakeAppUI(root)
    root.mainloop()

if __name__ == '__main__':
    main() 
