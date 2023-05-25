
import torch
from torch import nn, optim


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, num_classes=1):
        super(LSTM, self).__init__()
        self.device = device
        
        self.classes =  num_classes

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # 2 different prediction modes, single output prediction or elo range classification
        if self.classes == 1:
            # Sigmoid function outputs a value between 0 and 1, Elo will be normalized
            self.output = nn.Sigmoid()
        else:
            # Softmax function outputs a probability distribution over the 10 classes
            self.output = nn.Softmax(dim=1)
            
        self.to(device)

    def forward(self, X, h0=None, c0=None, train=True):
        # If X is a single sample, add a batch dimension
        if X.dim() == 2:
            X = X.unsqueeze(0)
            
        if c0 is None:
            c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(self.device)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(self.device)

        out, (hn, cn) = self.lstm(X, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        
        # Single elo prediction
        if self.classes == 1:
            return self.output(out), (hn, cn)
        
        # Elo range classification
        # CrossEntropyLoss applies the softmax function itself
        # Only apply softmax to get predictions, not training
        if not train:
            out = self.output(out)

        return out, (hn, cn)


def initialize_model(input_size, hidden_size, num_layers, device, learning_rate, num_classes=1):
    lstm_model = LSTM(input_size, hidden_size, num_layers, device, num_classes)

    optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

    return lstm_model, optimizer


def train_model(model, optimizer, loss_func, train_data, num_epochs):

    # train_data is of the form [X_train, y_train]
    x_train, y_train = train_data
    
    # x_train is a list of games, where each game is a tensor of the form.
    # Both perspectives, white and black are batched together
    # x_train.shape = (batch_size, num_moves, input_size)
    
    # y_train is a list of the expected outputs for each game
    # y_train.shape = (batch_size, 1)

    total_games = len(x_train)
    
    # Split off 15% of the data for validation
    validation_games = int(total_games * 0.15)

    x_val = x_train[-validation_games:]
    y_val = y_train[-validation_games:]

    x_train = x_train[:-validation_games]
    y_train = y_train[:-validation_games]

    batch_size = len(x_train[0])

    # save the validation loss for each epoch for visualization
    loss_per_epoch = []
    
    for epoch in range(num_epochs):
        for _batch, (x_game, y_game) in enumerate(zip(x_train, y_train)):
            # Reset the hidden and cell states
            cell_states = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(model.device)
            hidden_states = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(model.device)
            
            # get output prediction
            prediction, (_hidden_states, _cell_states) = model(
                x_game, hidden_states, cell_states
            )
            # calculate loss and optimize
            loss = loss_func(prediction, y_game)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        # Calculate validation loss
        cell_states = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(model.device)
        hidden_states = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(model.device)
        loss_sum = sum(
            loss_func(model(x_val[i], cell_states, hidden_states)[0], y_val[i]).item()
            for i in range(validation_games)
        )

        print(f"Epoch: {epoch + 1}, Validation Loss: {loss_sum/validation_games}")

        loss_per_epoch.append(loss_sum / (validation_games + 1))

    return loss_per_epoch


def main():
    # For the input we pass a bitboard representation of the position,
    # That means a 8 x 8 matrix with 12 values for each square, one for each piece and color
    # Plus 1 to show whos turn it is to move
    # Plus 2 for the stockfish evaluations before and after the move
    # The hope is that the LSTM can analyze the position and calculate a "complexity" score
    # and using the stockfish evaluation, compare how worse the human move is to the computer move.
    # With the complexity of the position in mind and previous guesses, guess an ELO range for the player

    input_size = 64 * 12 + 1 + 2
    hidden_size = 128
    num_layers = 2

    # Target output is a vector of 10 ranges for the players ELO
    # <900, 900-1100, 1100-1300, 1300-1500, 1500-1700, 1700-1900, 1900-2100, 2100-2300, 2300-2500, >2500
    num_classes = 10
    # Or a single value for the players ELO
    num_classes = 1
    
    lstm, optimizer = initialize_model(input_size, hidden_size, num_layers, num_classes)
    
    return lstm, optimizer, 