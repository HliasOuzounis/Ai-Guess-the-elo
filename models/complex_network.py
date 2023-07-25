from elo_guesser.helper_functions.get_device import get_device
import torch
from torch import nn

NUM_LAYERS = 1
EVALUATION_OUTPUT_SIZE = 32
DENSE_LAYER_SIZE = 128
LSTM_HIDDEN_SIZE = 64

device = get_device()


class EloGuesser(nn.Module):
    def __init__(self, input_size, input_channels=1, num_classes=1):
        super(EloGuesser, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels

        self.position_parser = self.create_position_parser()
        self.evaluation_parser = self.create_evaluation_parser()
        self.dense_layer1 = nn.Linear(EVALUATION_OUTPUT_SIZE + 256, DENSE_LAYER_SIZE)
        self.lstm_layer = self.create_lstm_layer()
        self.dense_layer2 = nn.Linear(LSTM_HIDDEN_SIZE, num_classes)
        self.softmax = nn.Softmax(dim=-1)

        self.to(device)

    def forward(self, X, h0=None, c0=None):
        position, evaluation = X
        white, black = position
        parsed_position = torch.stack(
            (self.position_parser(white), self.position_parser(black)))
        # print(parsed_position.shape)
        parsed_evaluation = self.evaluation_parser(evaluation)
        # print(parsed_evaluation.shape)
        
        combined_output = torch.cat((parsed_position, parsed_evaluation), dim=-1)
        lstm_input = self.dense_layer1(combined_output)

        # lstm_input = torch.cat((parsed_position, parsed_evaluation), dim=-1)
        if c0 is None:
            c0 = torch.zeros(NUM_LAYERS, lstm_input.size(0),
                             LSTM_HIDDEN_SIZE).to(device)
        if h0 is None:
            h0 = torch.zeros(NUM_LAYERS, lstm_input.size(0),
                             LSTM_HIDDEN_SIZE).to(device)
        lstm_output, (hn, cn) = self.lstm_layer(lstm_input, (h0, c0))

        output = self.dense_layer2(lstm_output[:, -1, :])
        if not self.training: 
            output = self.softmax(output)

        return output, (hn, cn)

    def create_position_parser(self):
        # 8x8 board, 2 channels, seq, batch=2
        # shape is (2, seq, 2, 8, 8)
        return nn.Sequential(
            nn.Conv2d(self.input_channels, 32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Dropout(0.4),
            nn.Flatten(),
        )

    def create_evaluation_parser(self):
        return nn.Sequential(
            nn.Linear(self.input_size, EVALUATION_OUTPUT_SIZE),
            nn.ReLU(),
        )

    def create_lstm_layer(self):
        return nn.LSTM(DENSE_LAYER_SIZE, LSTM_HIDDEN_SIZE, NUM_LAYERS, batch_first=True)


if __name__ == "__main__":
    input_size = 18
    channels =2
    num_classes = 10
    model = EloGuesser(input_size, input_channels=channels, num_classes=num_classes)

    seq = 5
    position_input = torch.randn(2, seq, channels, 8, 8).to(device)
    eval_input = torch.randn(2, seq, input_size).to(device)

    out1, (_h, _c) = model.eval()((position_input, eval_input))
    print(out1.shape)
    
    # out2, (_h, _c) = model.train()((position_input, eval_input))
    # print(out2)
    

