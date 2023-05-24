ratings1=("800" "1000" "1200" "1400" "1600" "1800" "2000" "2200" "2400" "2600")
ratings2=("1000" "1200" "1400" "1600" "1800" "2000" "2200" "2400" "2600" "3000")

lenght=${#ratings1[@]}
echo $lenght
for ((i=0; i<$lenght; i++)) do
    low=${ratings1[$i]}
    high=${ratings2[$i]}

    sed -i 's/WhiteElo <= "[0-9]*"/WhiteElo <= "'"$high"'"/g' datasets/pgn_args
    sed -i 's/BlackElo >= "[0-9]*"/BlackElo >= "'"$low"'"/g' datasets/pgn_args
    
    echo $low "-" $high
    pgn-extract -A datasets/pgn_args -o datasets/outputs/$low-$high.pgn datasets/lichess_db_standard_rated_2018-06.pgn

done
