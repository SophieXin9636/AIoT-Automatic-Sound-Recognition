#!/bin/sh

python3 spectrogram.py -T stft -i ./combine_data/incorrect/ -o ./combine_data/incorrect/
python3 spectrogram.py -T stft -i ./combine_data/correct/ -o ./combine_data/correct/

# training
#python3 binary_cnn.py --correct 0727_data/training/img/stft/ --incorrect 0727_data/incorrect/img/stft/

# validation
#python3 validation.py --path0 ./0727_data/incorrect/ --path1 ./0727_data/validation/
