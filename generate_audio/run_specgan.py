import numpy as np
import codecs
import json
import csv
from keras.models import model_from_json, load_model
from keras.preprocessing.image import load_img, img_to_array
import soundfile as sf
import librosa
from scipy import signal
from hparams import hparams
from utils import audio_tools as audio
from matplotlib import pyplot as plt
import tensorflow as tf
import argparse
import os
import wave

def save_audio(y, path):
    """ generate a wav file from a given spectrogram and save it """
    s = np.squeeze(y)
    s = denormalize(s)
    w = audio.inv_melspectrogram(s)
    sf.write(path, w, 16000, subtype='PCM_16')

def denormalize(norm_s):
    """ normalized spectrogram to original spectrogram using the calculated mean/standard deviation """
    assert norm_s.shape[0] == mel_means.shape[0]
    Y = (norm_s * (3.0 * mel_stds)) + mel_means
    return Y

def transform_to_float(data):
    new_data = []
    for row in data:
        new_vector = []
        for column in row:
            new_vector.append(float(column))
        new_data.append(new_vector)

    return new_data

#----------------------------------------
#----------------------------------------
# Arguments
parser = argparse.ArgumentParser(description="Unofficial implementation of SpecGAN - Generate audio through spectrogram image with adversarial training")
parser.add_argument("--input_dir", "-i", required=True, help="Directory to input generated csv files to")
parser.add_argument("--situation", "-s", required=True, help="Situation")
args = parser.parse_args()

if os.path.exists("./audios/" + args.input_dir) is False:
    os.mkdir("./audios/" + args.input_dir)

# for denomalizing mel_spectrogram
mel_means = np.load("all_data.npz")["mean"]
mel_stds = np.load("all_data.npz")["std"]

MODEL_ARC_PATH = "./generator_epoch_1000.h5"

with open('model_in_json.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights(MODEL_ARC_PATH)


latent_vectors_path = "./../csv_files/latent_vectors/" + args.situation + "/" + args.input_dir + "/children.csv"

latent_vectors_csv = open(latent_vectors_path, 'r')
latent_vectors = list(csv.reader(latent_vectors_csv))
latent_vectors = transform_to_float(latent_vectors)
latent_vectors = np.array(latent_vectors)
test = model.predict(latent_vectors)

for i in range(len(latent_vectors)):
    w = test[i]
    s = np.squeeze(w)
    s = denormalize(s)
    wavfile = audio.inv_melspectrogram(s)
    sf.write("./audios/" + args.input_dir + "/audio_%s%s_%i.wav" % (args.situation, args.input_dir, i), wavfile, 16000, subtype='PCM_16')

    plt.plot(wavfile)
    plt.savefig("./audios/" + args.input_dir + "/image_%s%s_%i.png" % (args.situation, args.input_dir, i))
    plt.close()
