import csv
import argparse
import os
import functions

parser = argparse.ArgumentParser(description="Unofficial implementation of SpecGAN - Generate audio through spectrogram image with adversarial training")
parser.add_argument("--situation", "-s", required=True, help="Situation")
parser.add_argument("--output_dir", "-o", required=True, help="Directry to output children index file")
args = parser.parse_args()

filepath = './../csv_files/latent_vectors/' + args.situation + "/" + args.output_dir + "/children_index.csv"
indexes = input().split()

functions.write_csv_for_vector(filepath, indexes)
