import matplotlib.pyplot as plt
import numpy as np

import chess.pgn
import chess.svg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageTk
import io
import cairosvg

from elo_ai.helper_functions.elo_range import get_elo_prediction, get_rating_ranges
import torch

rating_ranges = get_rating_ranges()[:, 0].tolist()


def plot_predictions(sequential_predictions, game, is_chessdotcom=False, save=False):    
    padding = torch.full_like(sequential_predictions[0], 1 / len(rating_ranges))
    sequential_predictions = [padding] + sequential_predictions
    moves = list(game.mainline_moves())

    move_counter = 0
    
    root, chessboard, ax, canvas = init_window()
    plot_chessboard(chessboard, moves, move_counter)
    plot_preditions(ax, sequential_predictions, move_counter, is_chessdotcom)

    def update(next_move):
        nonlocal move_counter
        if next_move and move_counter < len(moves):
            move_counter += 1
        if (not next_move) and  move_counter > 0:
            move_counter -= 1
        
        idx = move_counter
            
        plot_chessboard(chessboard, moves, idx)
        plot_preditions(ax, sequential_predictions, idx, is_chessdotcom)
        canvas.draw()
        
    root.bind('<Left>', lambda event: update(False))
    root.bind('<Right>', lambda event: update(True))
    
    root.mainloop()

def plot_chessboard(chessboard, moves, idx):
    chessboard.config(image="")

    board = chess.Board()
    for i in range(idx):
        board.push(moves[i])
    
    svg = chess.svg.board(board=board, size=800)
    img = svg_to_image(svg)
    chessboard.img = ImageTk.PhotoImage(img)
    chessboard.config(image=chessboard.img)

def plot_preditions(axs, predictions, idx, is_chessdotcom):
    plot_bar_graph(axs[0], predictions, (idx + 1) // 2, 0, is_chessdotcom)
    plot_bar_graph(axs[1], predictions, idx // 2, 1, is_chessdotcom)
    

def plot_bar_graph(ax, predicitons, idx, is_black, is_chessdotcom):    
    ax.cla()
    ax.bar(np.arange(len(rating_ranges)), predicitons[idx][is_black].tolist(), color='blue' if is_black else 'orange')
    ax.set_title(f'{"Black" if is_black else "White"}\nMove: {idx}, Prediction: {get_elo_prediction(predicitons[idx][is_black], is_chessdotcom)}')
    
    ax.set_xlabel('Rating Range')
    ax.set_ylabel('Probability')
    
    ax.set_xticks(range(0, len(rating_ranges), 5))
    ax.set_xticklabels(rating_ranges[::5])
    ax.set_yticks([])
    ax.set_ylim(0, 0.15)

# Function to convert SVG to an image usable in tkinter
def svg_to_image(svg):
    """Convert an SVG string to a PIL Image."""
    png_data = cairosvg.svg2png(bytestring=svg)
    return Image.open(io.BytesIO(png_data))
    

def init_window():
    # Create the tkinter window
    root = tk.Tk()
    root.title("Chess Game Visualizer")

    # Create a frame for the chessboard
    chessboard_frame = tk.Frame(root)
    chessboard_frame.pack(side=tk.LEFT)

    # Create a label to display the chessboard
    chessboard_label = tk.Label(chessboard_frame)
    chessboard_label.pack()

    # Create a frame for the matplotlib graph
    graph_frame = tk.Frame(root)
    graph_frame.pack(side=tk.RIGHT)

    # Create a matplotlib figure and embed it in the tkinter window
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    fig.subplots_adjust(hspace=0.5)
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.get_tk_widget().pack()
    
    return root, chessboard_label, ax, canvas


if __name__ == "__main__":
    game = chess.pgn.read_game(open("game.pgn"))

    # Generate random predictions for demonstration purposes
    num_moves = len(list(game.mainline_moves()))
    num_classes = len(rating_ranges)
    preds = [torch.rand((2, num_classes)) for _ in range(num_moves)]

    # Plot the predictions
    plot_predictions(preds, game, save=True)
