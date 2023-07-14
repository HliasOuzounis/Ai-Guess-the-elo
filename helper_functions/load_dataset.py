import torch


def load_games(position_type, dataset_path="/datasets/"):
    positions, elo = torch.load(
        dataset_path + f"positions/all_{position_type}_0-2000.pt")

    game_analysis1 = torch.load(
        dataset_path + "analysis/all_analysis_0-500.pt")
    game_analysis2 = torch.load(
        dataset_path + "analysis/all_analysis_500-1000.pt")
    game_analysis3 = torch.load(
        dataset_path + "analysis/all_analysis_1000-1500.pt")
    game_analysis4 = torch.load(
        dataset_path + "analysis/all_analysis_1500-2000.pt")

    analysis = []
    elo_validation = []
    for i in range(10):
        analysis += game_analysis1[0][i * 500: (i + 1) * 500] + game_analysis2[0][i * 500: (
            i + 1) * 500] + game_analysis3[0][i * 500: (i + 1) * 500] + game_analysis4[0][i * 500: (i + 1) * 500]
        elo_validation += game_analysis1[1][i * 500: (i + 1) * 500] + game_analysis2[1][i * 500: (
            i + 1) * 500] + game_analysis3[1][i * 500: (i + 1) * 500] + game_analysis4[1][i * 500: (i + 1) * 500]

    # Check that the data is loaded correctly, positions match with analysis
    for i, (pos, anal) in enumerate(zip(positions, analysis)):
        assert pos.size()[1] == anal.size()[1] # Same number of moves
    for i, (game_elo1, game_elo2) in enumerate(zip(elo, elo_validation)):
        assert game_elo1[0] == game_elo2[0] # White elo
        assert game_elo1[1] == game_elo2[1] # Black elo

    dataset = [
        (torch.cat((game_position, game_analysis), dim=-1), game_elo)
        for game_position, game_analysis, game_elo in zip(positions, analysis, elo)
    ]

    return dataset
