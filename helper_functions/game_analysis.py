import torch
import tqdm
import chess.engine


def pad_tensor(tensor, length, pad_value):
    return torch.cat((tensor, torch.ones(length - len(tensor)) * pad_value))


def analyze_game(game, engine, mate_score, nof_moves=10, time_limit=0.1, progress_bar=False):
    board = game.board()
    analysis = []

    moves = list(game.mainline_moves())
    iterations = tqdm(range(len(moves)),
                      desc="Analyzing game") if progress_bar else len(moves)

    for i in iterations:
        move = moves[i]
        # Engine evaluation before the move
        top_moves_tensor, before_wdl_tensor = analysis_before_move(
            engine, mate_score, nof_moves, time_limit, board)
        board.push(move)
        # Engine evaluation after the move
        after_move_tensor, after_wdl_tensor = analysis_after_move(
            engine, mate_score, time_limit, board)

        analysis.append(torch.cat((
            top_moves_tensor, before_wdl_tensor,
            after_move_tensor, after_wdl_tensor
        )))

    # Pad the game if it ends on white's turn to batch white's and black's analysis together
    if len(analysis) % 2:
        analysis.append(torch.ones_like(analysis[0]) * (-mate_score))

    white_analysis = torch.stack([position for position in analysis[::2]])
    black_analysis = torch.stack([position for position in analysis[1::2]])

    return torch.stack((white_analysis, black_analysis))


def analysis_after_move(engine, mate_score, time_limit, board):
    after_move = engine.analyse(board, chess.engine.Limit(time=time_limit))
    # Now it's the opponent's turn so negate the score
    after_move_tensor = torch.Tensor(
        [after_move["score"].relative.score(mate_score=mate_score) * -1]) / mate_score
    # Reverse the list so that it's from the perspective of the player who just moved
    after_wdl = after_move["score"].relative.wdl()
    after_wdl_tensor = torch.Tensor([
        after_wdl.losing_chance(),
        after_wdl.drawing_chance(),
        after_wdl.winning_chance()
    ])

    return after_move_tensor, after_wdl_tensor


def analysis_before_move(engine, mate_score, nof_moves, time_limit, board):
    top_moves = engine.analyse(board, chess.engine.Limit(
        time=time_limit), multipv=nof_moves)
    # A tensor with the score of the top nof_moves moves (normalized to be between -1 and 1)
    top_moves_tensor = torch.Tensor(
        [eval["score"].relative.score(mate_score=mate_score) for eval in top_moves])
    # Pad the tensor if there are less than nof_moves legal moves
    top_moves_tensor = pad_tensor(
        top_moves_tensor, nof_moves, -mate_score) / mate_score
    # Win, draw, loss chance tensor before the move
    before_wdl = top_moves[0]["score"].relative.wdl()
    before_wdl_tensor = torch.Tensor([
        before_wdl.winning_chance(),
        before_wdl.drawing_chance(),
        before_wdl.losing_chance()
    ])

    return top_moves_tensor, before_wdl_tensor
