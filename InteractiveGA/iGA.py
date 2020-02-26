import argparse
import functions
import numpy as np 
import csv
import os

#----------------------------------------
#----------------------------------------
# Arguments
parser = argparse.ArgumentParser(description="Unofficial implementation of SpecGAN - Generate audio through spectrogram image with adversarial training")
parser.add_argument("--input_dir", "-i", required=True, help="csv file for input to generate audio")
parser.add_argument("--output_dir", "-o", required=True, help="Directory to output generated csv files to")
parser.add_argument("--situation", "-s", required=True, help="Situation")
parser.add_argument("--num_parent", "-p", default=4, type=int, help="Number of parents")
parser.add_argument("--num_children", "-c", default=16, type=int, help="Number of children")
args = parser.parse_args()

csv_dir = './../csv_files/latent_vectors/' + args.situation + "/"

if os.path.exists(csv_dir + args.output_dir) is True:
    exit()
if os.path.exists(csv_dir + args.output_dir) is False:
    os.mkdir(csv_dir + args.output_dir)

input_path = csv_dir + args.input_dir + '/latent_vectors.csv'
latent_vectors = functions.read_csv(input_path)
latent_vectors = functions.transform_to_float(latent_vectors)
latent_vectors = np.array(latent_vectors)

num_vectors = len(latent_vectors)
parents_index = functions.make_random_list(num_vectors, args.num_parent)

parent_vectors = np.zeros((args.num_parent, len(latent_vectors[0])))
for index in range(len(parents_index)):
    parent_vectors[index] = np.copy(latent_vectors[parents_index[index]])

children = functions.make_children(parent_vectors, args.num_children)

# 親に使ったベクトルのインデックスのcsvファイルを作成
functions.write_csv_for_vector(csv_dir + args.output_dir + "/parents_index.csv", parents_index)
# 親に使ったベクトルのcsvファイルを作成
functions.write_csv(csv_dir + args.output_dir + "/parents.csv", parent_vectors)
# 新しい子のcsvファイルの作成
functions.write_csv(csv_dir + args.output_dir + "/children.csv", children)
