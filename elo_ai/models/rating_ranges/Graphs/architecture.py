import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    
    to_head( '..' ),
    to_cor(),
    to_begin(),

    to_input('/home/hlias/code/machine_learning/elo_guesser/video/chessboard_transparecy/20.png', to='(-4.5,0,0)',),
    to_input('/home/hlias/code/machine_learning/elo_guesser/video/chessboard_transparecy/20.png', to='(-4,0,0)',),
    to_input('/home/hlias/code/machine_learning/elo_guesser/video/chessboard_transparecy/20.png', to='(-3.5,0,0)',),
    to_input('/home/hlias/code/machine_learning/elo_guesser/video/chessboard_transparecy/20.png'),

    to_Conv('conv1', s_filer=8, n_filer=32, offset="(0,0,0)", to="(0,0,0)", width=2, height=40, depth=40, caption="Conv1"),
    to_Pool('pool1', offset="(0,0,0)", to="(conv1-east)", width=1, height=32, depth=32, opacity=0.5),

    to_Conv('conv2', s_filer=8, n_filer=64, offset="(2,0,0)", to="(pool1-east)", width=2, height=32, depth=32, caption="Conv2"),
    to_Pool('pool2', offset="(0,0,0)", to="(conv2-east)", width=1, height=25, depth=25, opacity=0.5),

    to_connection('pool1', 'conv2'),

    to_Sum('cat', offset="(1,0,0)", to="(pool2-east)", opacity=0.7),

    to_SoftMax('dense', offset="(1,0,0)", to="(cat-east)", s_filer=2, height=30, depth=1, opacity=0.9),
    to_connection('cat', 'dense'),

    to_SoftMax('eval', offset="(-3, -2, 0)", to='(0,0,0)', s_filer=2, height=20, depth=1, opacity=0.9, caption="Eval"),

    to_connection('pool2', 'cat'),
    to_connection('eval', 'cat'),

    to_UnPool('lstm', offset="(2,0,0)", to="(dense-east)", width=20, height=20, depth=20, opacity=0.9, caption="LSTM"),
    to_connection('dense', 'lstm'),

    to_SoftMax('output', offset="(2,0,0)", to="(lstm-east)", height=30, depth=1, opacity=0.9, caption="Output"),
    to_connection('lstm', 'output'),
    
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()