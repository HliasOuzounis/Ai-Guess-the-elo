import torch
from torch import nn, optim


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, X, h0=None, c0=None, train=True):
        # If X is a single sample, add a batch dimension
        if X.dim() == 2:
            X = X.unsqueeze(0)
            
        if c0 is None:
            c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)

        out, (hn, cn) = self.lstm(X, (h0, c0))
        # Packed batches test, not working
        # padded_output, output_lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=5)
        # out = self.fc(padded_output[:, -1, :])
        
        out = self.fc(out[:, -1, :])
        
        # CrossEntropyLoss applies the softmax function itself
        # Only apply softmax to get predictions, not training
        # if not train:
        #     out = self.softmax(out)

        return out, (hn, cn)


def initialize_model(input_size, hidden_size, num_layers, num_classes):
    lstm_model = LSTM(input_size, hidden_size, num_layers, num_classes)

    learning_rate = 0.001
    optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

    loss_func = nn.CrossEntropyLoss()

    return lstm_model, optimizer, loss_func


def train_model(model, optimizer, loss_func, train_data, num_epochs):
   
    # train_data = [X_train, y_train]
    # x_train.shape = (batch_size, num_moves, input_size)
    # y_train.shape = (batch_size, num_classes)
    for epoch in range(num_epochs):
        for batch, (x_train, y_train) in enumerate(zip(*train_data)):
            
            batch_size = len(x_train)
            optimizer.zero_grad()

            cell_states = torch.zeros(
                model.num_layers, batch_size, model.hidden_size
            )
            hidden_states = torch.zeros(
                model.num_layers, batch_size, model.hidden_size
            )
            
            # BATCH GAMES TOGETHER BASED ON NUMBER OF MOVES
            # Train the model on each move => 
            # + More samples and better predictions for small games    
            # - Computationally expensive
            # Will maybe change to transformers later
            if x_train.dim() == 2:
                x_train = x_train.unsqueeze(0)
            
            if y_train.dim() == 1:
                y_train = y_train.unsqueeze(0)
            
            for num_moves in range(1, x_train.shape[1] + 1):
                propabilities, (_hidden_states, _cell_states) = model(
                    x_train[:, :num_moves, :], hidden_states, cell_states
                )
                
                loss = loss_func(propabilities, y_train)
                loss.backward()

                optimizer.step()

            if True: #batch % 100 == 0:
                print(f"Epoch: {epoch + 1}, Batch: {batch + 1}, Loss: {loss.item()}")


def main():
    # For the input we pass a bitboard representation of the position,
    # That means a 8 x 8 matrix with 12 values for each square, one for each piece and color
    # Plus 2 for castling rights and - for the stockfish evaluations before and after the move
    # The hope is that the LSTM can analyze the position and calculate a "complexity" score
    # and using the stockfish evaluation, compare how worse the human move is to the computer move.
    # With the complexity of the position in mind and previous guesses, guess an ELO range for the player

    # TODO: how many stockfish evaluations
    input_size = 2
    # input_size = 64 * 12 + 14 # for stockfish evaluation, top 10 moves and win draw lose odds
    hidden_size = 16
    # hidden_size = 64
    num_layers = 2

    # Target output is a vector of 10 ranges for the players ELO
    # <900, 900-1100, 1100-1300, 1300-1500, 1500-1700, 1700-1900, 1900-2100, 2100-2300, 2300-2500, >2500
    num_classes = 2
    # num_classes = 10
    
    lstm, optimizer, loss_func = initialize_model(input_size, hidden_size, num_layers, num_classes)
    
    return lstm, optimizer, loss_func
