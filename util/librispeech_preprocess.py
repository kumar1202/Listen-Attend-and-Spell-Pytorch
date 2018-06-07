
from pydub import AudioSegment
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from python_speech_features import logfbank
import scipy.io.wavfile as wav

import argparse


parser = argparse.ArgumentParser(description='Librispeech preprocess.')

parser.add_argument('root', metavar='root', type=str,
                     help='Absolute file path to LibriSpeech. (e.g. /usr/downloads/LibriSpeech/)')

parser.add_argument('sets', metavar='sets', type=str, nargs='+',
                     help='Datasets to process in LibriSpeech. (e.g. train-clean-100/)')

parser.add_argument('--n_jobs', dest='n_jobs', action='store', default=-2 ,
                   help='number of cpu availible for preprocessing.\n -1: use all cpu, -2: use all cpu but one')
parser.add_argument('--n_filters', dest='n_filters', action='store', default=40 ,
                   help='number of filters for fbank. (Default : 40)')
parser.add_argument('--win_size', dest='win_size', action='store', default=0.025 ,
                   help='window size during feature extraction (Default : 0.025 [25ms])')
parser.add_argument('--char_map', dest='char_map', action='store', default=None ,
                   help='Character2Index mapping file, generated during training data preprocessing. Specify this argument when processing dev/test data.')



paras = parser.parse_args()

root = paras.root
libri_path = paras.sets
n_jobs = paras.n_jobs
n_filters = paras.n_filters
win_size = paras.win_size
char_map_path = paras.char_map


# # flac 2 wav


def flac2wav(f_path):
    flac_audio = AudioSegment.from_file(f_path, "flac")
    flac_audio.export(f_path[:-5]+'.wav', format="wav")


    
print('Processing flac2wav...',flush=True)
print(flush=True)


file_list = []

for p in libri_path:
    p = root + p
    for sub_p in sorted(os.listdir(p)):
        for sub2_p in sorted(os.listdir(p+sub_p+'/')):
            for file in sorted(os.listdir(p+sub_p+'/'+sub2_p)):
                if '.flac' in file:
                    file_path = p+sub_p+'/'+sub2_p+'/'+file
                    file_list.append(file_path)
                    

results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(flac2wav)(i) for i in tqdm(file_list))

                    
print('done')


# # wav 2 log-mel fbank


def wav2logfbank(f_path):
    (rate,sig) = wav.read(f_path)
    fbank_feat = logfbank(sig,rate,winlen=win_size,nfilt=n_filters)
    np.save(f_path[:-3]+'fb'+str(n_filters),fbank_feat)

print('Processing wav2logfbank...',flush=True)
print(flush=True)

results = Parallel(n_jobs=n_jobs,backend="threading")(delayed(wav2logfbank)(i[:-4]+'wav') for i in tqdm(file_list))
                    
print('done')   


# # log-mel fbank 2 feature

print('Preparing dataset...',flush=True)

file_list = []
text_list = []

for p in libri_path:
    p = root + p
    for sub_p in sorted(os.listdir(p)):
        for sub2_p in sorted(os.listdir(p+sub_p+'/')):
            # Read trans txt
            with open(p+sub_p+'/'+sub2_p+'/'+sub_p+'-'+sub2_p+'.trans.txt','r') as txt_file:
                for line in txt_file:
                    text_list.append(' '.join(line[:-1].split(' ')[1:]))
            # Read acoustic feature
            for file in sorted(os.listdir(p+sub_p+'/'+sub2_p)):
                if '.fb'+str(n_filters) in file:
                    file_path = p+sub_p+'/'+sub2_p+'/'+file
                    file_list.append(file_path)


X = []
for f in file_list:
    X.append(np.load(f))



audio_len = [len(x) for x in X]



# Sort data by signal length (long to short)
file_list = [file_list[idx] for idx in reversed(np.argsort(audio_len))]
text_list = [text_list[idx] for idx in reversed(np.argsort(audio_len))]


if char_map_path:
    # Load char mapping
    char_map = {}
    with open(char_map_path,'r') as f:
        for line in f:
            if 'idx,char' in line:
                continue
            idx = int(line.split(',')[0])
            char = line[:-1].split(',')[1]
            char_map[char] = idx
    
else:
    assert 'train' in libri_path[0]
    # Create char mapping
    char_map = {}
    char_map['<sos>'] = 0
    char_map['<eos>'] = 1
    char_idx = 2

    # map char to index
    for text in text_list:
        for char in text:
            if char not in char_map:
                char_map[char] = char_idx
                char_idx +=1
                
    # Reverse mapping
    rev_char_map = {v:k for k,v in char_map.items()}

    # Save mapping
    with open(root+'idx2chap.csv','w') as f:
        f.write('idx,char\n')
        for i in range(len(rev_char_map)):
            f.write(str(i)+','+rev_char_map[i]+'\n')

# text to index sequence
tmp_list = []
for text in text_list:
    tmp = []
    for char in text:
        tmp.append(char_map[char])
    tmp_list.append(tmp)
text_list = tmp_list
del tmp_list


# write dataset


if 'train' in libri_path[0]:
    file_name = 'train.csv'
elif 'test' in libri_path[0]:
    file_name = 'test.csv'
elif 'dev' in libri_path[0]:
    file_name = 'dev.csv'

print('Writing dataset to '+root+file_name+'...',flush=True)

with open(root+file_name,'w') as f:
    f.write('idx,input,label\n')
    for i in range(len(file_list)):
        f.write(str(i)+',')
        f.write(file_list[i]+',')
        for char in text_list[i]:
            f.write(' '+str(char))
        f.write('\n')
print('done')
