import torch
import chess.engine

from tqdm import tqdm

def fen_to_bitboard(fen_str):
    mappings = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
    }   
    
    bitboard = torch.zeros(12, 64)
    fen, move, _castle, _en_passant, _halfmove, _fullmove = fen_str.split(" ")
    row, col = 0, 0
    for char in fen:
        if char == "/":
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            bitboard[mappings[char], row * 8 + col] = 1
            col += 1
    # Flatten the bitboard and add whose move it is
    return torch.cat((torch.tensor([1 if move == "w" else -1]), bitboard.flatten()))


def pad_tensor(tensor, length, pad_value):
    return torch.cat((tensor, torch.ones(length - len(tensor)) * pad_value))

def analyze_game(game, engine, mate_score, nof_moves=10):
    board = game.board()
    analysis = []
    
    moves = list(game.mainline_moves())
    # for i, move in enumerate(game.mainline_moves()):
    for i in tqdm(range(len(moves)), desc="Analyzing game"):
        move = moves[i]
        # Engine evaluation before the move
        top_moves = engine.analyse(board, chess.engine.Limit(time=0.1), multipv=nof_moves)
        # A tensor with the score of the top nof_moves moves (normalized to be between -1 and 1)
        top_moves_tensor = torch.Tensor([eval["score"].relative.score(mate_score=mate_score) for eval in top_moves]) 
        # Pad the tensor if there are less than nof_moves legal moves
        top_moves_tensor = pad_tensor(top_moves_tensor, nof_moves, -mate_score) / mate_score
        # Win, draw, loss chance tensor before the move
        before_wdl = top_moves[0]["score"].relative.wdl()
        before_wdl_tensor = torch.Tensor([
            before_wdl.winning_chance(), 
            before_wdl.drawing_chance(), 
            before_wdl.losing_chance()
        ])
        board.push(move)
        # Engine evaluation after the move
        after_move = engine.analyse(board, chess.engine.Limit(time=0.1))
        # Now it's the opponent's turn so negate the score
        after_move_tensor = torch.Tensor([after_move["score"].relative.score(mate_score=mate_score) * -1]) / mate_score
        # Reverse the list so that it's from the perspective of the player who just moved
        after_wdl = after_move["score"].relative.wdl()
        after_wdl_tensor = torch.Tensor([
            after_wdl.losing_chance(), 
            after_wdl.drawing_chance(), 
            after_wdl.winning_chance()
        ])

        analysis.append(torch.cat((
            top_moves_tensor, before_wdl_tensor, 
            after_move_tensor, after_wdl_tensor
        )))
        # print(f"before move: {move}, top_moves {top_moves_tensor}, wdl {before_wdl_tensor}")
        # print(f"after:               top_moves {after_move_tensor}, wdl {after_wdl_tensor}")
    
    # Pad the game if it ends on white's turn to batch white's and black's analysis together
    if len(analysis) % 2:
        analysis.append(torch.ones_like(analysis[0]) * (-mate_score))
    
    white_analysis = torch.stack([position for position in analysis[::2]])
    black_analysis = torch.stack([position for position in analysis[1::2]])
    
    return torch.stack((white_analysis, black_analysis))


def convert_position(game, func):
    board = game.board()
    positions = []
    
    for move in game.mainline_moves():
        board_position_tensor = func(board.fen())
        positions.append(board_position_tensor)
        board.push(move)
        
    # Pad the game if it ends on white's turn to batch white's and black's analysis together later
    if len(positions) % 2:
        positions.append(torch.zeros_like(positions[0]))
    
    white_positions = torch.stack([position for position in positions[::2]])
    black_positions = torch.stack([position for position in positions[1::2]])
    
    return torch.stack((white_positions, black_positions))

def fen_to_bitboard(fen_str):
    bitboard = torch.zeros(12, 64)
    fen, move, _castle, _en_passant, _halfmove, _fullmove = fen_str.split(" ")
    
    mappings = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
    }
    
    row, col = 0, 0
    for char in fen:
        if char == "/":
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            bitboard[mappings[char], row * 8 + col] = 1
            col += 1
    # Flatten the bitboard and add whose move it is
    return torch.cat((torch.tensor([1 if move == "w" else -1]), bitboard.flatten()))

def fen_to_bitboard_mirror(fen_str):
    bitboard = torch.zeros(12, 64)
    fen, move, _castle, _en_passant, _halfmove, _fullmove = fen_str.split(" ")
    
    mappings = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
    }   if move == "w" else {
        "p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5,
        "P": 6, "N": 7, "B": 8, "R": 9, "Q": 10, "K": 11,
    } 
    fen = fen if move == "w" else fen[::-1]
    row, col = 0, 0
    for char in fen:
        if char == "/":
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            bitboard[mappings[char], row * 8 + col] = 1
            col += 1
    # Flatten the bitboard and add whose move it is
    return torch.cat((torch.tensor([1 if move == "w" else -1]), bitboard.flatten()))

def fen_to_board(fen_str):
    board = torch.zeros(8, 8)
    fen, move, _castle, _en_passant, _halfmove, _fullmove = fen_str.split(" ")
    normalizer = 20
    mappings = {
        "P": 1, "N": 3, "B": 3.5, "R": 5, "Q": 9, "K": 20,
        "p": -1, "n": -3, "b": -3.5, "r": -5, "q": -9, "k": -20,
    } 
    row, col = 0, 0
    for char in fen:
        if char == "/":
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            board[row, col] = mappings[char] / normalizer
            col += 1
    return torch.cat((torch.tensor([1 if move == "w" else -1]), board.flatten())) 

def fen_to_board_mirror(fen_str):
    board = torch.zeros(8, 8)
    fen, move, _castle, _en_passant, _halfmove, _fullmove = fen_str.split(" ")
    normalizer = 20
    mappings = {
        "P": 1, "N": 3, "B": 3.5, "R": 5, "Q": 9, "K": 20,
        "p": -1, "n": -3, "b": -3.5, "r": -5, "q": -9, "k": -20,
    } if move == "w" else {
        "P": -1, "N": -3, "B": -3.5, "R": -5, "Q": -9, "K": -20,
        "p": 1, "n": 3, "b": 3.5, "r": 5, "q": 9, "k": 20,
    }
    fen = fen if move == "w" else fen[::-1]
    row, col = 0, 0
    for char in fen:
        if char == "/":
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            board[row, col] = mappings[char] / normalizer
            col += 1
    return torch.cat((torch.tensor([1 if move == "w" else -1]), board.flatten())) 