if [ "$#" -ne 1 ]; then
    echo "Usage : ./librispeech_preprocess.sh <LibriSpeech folder>"
fi

python3 librispeech_preprocess.py $1 train-clean-100/
python3 librispeech_preprocess.py --char_map $1/idx2chap.csv $1 dev-clean/
python3 librispeech_preprocess.py --char_map $1/idx2chap.csv $1 test-clean/