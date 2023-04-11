import torch
from torch import nn, optim


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.output = nn.Softmax(dim=1)

    def forward(self, X, h0=None, c0=None):
        if c0 is None:
            c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)

        out, (hn, cn) = self.lstm(X, (h0, c0))
        out = self.fc(out[:, -1, :])
        propabilities = self.output(out)

        return propabilities, (hn, cn)


def initialize_model(input_size, hidden_size, num_layers, num_classes):
    lstm_model = LSTM(input_size, hidden_size, num_layers, num_classes)

    learning_rate = 0.001
    optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

    loss_func = nn.CrossEntropyLoss()

    return lstm_model, optimizer, loss_func


def train_model(model, optimizer, loss_func, train_data, num_epochs):

    for epoch in range(num_epochs):
        for batch, (x_train, y_train) in enumerate(train_data):
            optimizer.zero_grad()

            cell_states = torch.zeros(
                model.num_layers, x_train.size(0), model.hidden_size
            )
            hidden_states = torch.zeros(
                model.num_layers, x_train.size(0), model.hidden_size
            )

            # BATCH GAMES TOGETHER BASED ON NUMBER OF MOVES
            for move in x_train:
                propabilities, (hidden_states, cell_states) = model(
                    move, hidden_states, cell_states
                )

                loss = loss_func(propabilities, y_train)
                loss.backward()
                optimizer.step()

            if batch % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}")


def main():
    # For the input we pass a bitboard representation of the position,
    # That means a 8 x 8 matrix with 12 values for each square, one for each piece and color
    # Plus 2 for castling rights and - for the stockfish evaluations before and after the move
    # The hope is that the LSTM can analyze the position and calculate a "complexity" score
    # and using the stockfish evaluation, compare how worse the human move is to the computer move.
    # With the complexity of the position in mind and previous guesses, guess an ELO range for the player

    # TODO: how many stockfish evaluations
    input_size = 64 * 12 + 2  # + - for stockfish evaluation
    hidden_size = 128
    num_layers = 2

    # Target output is a vector of 10 ranges for the players ELO
    # <700, 700-900, 900-1100, 1100-1300, 1300-1500, 1500-1700, 1700-1900, 1900-2100, 2100-2300, >2300
    num_classes = 10
