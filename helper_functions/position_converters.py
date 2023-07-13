import torch

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


position_converters = {
    "bitboards": fen_to_bitboard,
    "bitboards_mirrors": fen_to_bitboard_mirror,
    "boards": fen_to_board,
    "boards_mirrors": fen_to_board_mirror,
}
