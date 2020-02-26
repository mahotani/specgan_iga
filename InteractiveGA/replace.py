import argparse
import csv
import os
import functions

#----------------------------------------
#----------------------------------------
# Arguments
parser = argparse.ArgumentParser(description="Unofficial implementation of SpecGAN - Generate audio through spectrogram image with adversarial training")
parser.add_argument("--output_dir", "-o", required=True, help="Directory to output generated csv files to")
parser.add_argument("--old_dir", "-old", required=True, help="Directory that has old data")
parser.add_argument("--situation", "-s", required=True, help="Situation")
args = parser.parse_args()

csv_dir = './../csv_files/latent_vectors/' + args.situation + "/"

if os.path.exists(csv_dir + args.output_dir) is False:
    print("対象のファイルが見つかりません。")
    exit()

original_data_path = csv_dir + args.old_dir + "/latent_vectors.csv"
parents_index_path = csv_dir + args.output_dir + "/parents_index.csv"
children_index_path = csv_dir + args.output_dir + "/children_index.csv"
children_path = csv_dir + args.output_dir + "/children.csv"

original_data = functions.read_csv(original_data_path)
parents_index = functions.read_csv(parents_index_path)
parents_index = functions.transform_to_int(parents_index)
parents_index = parents_index[0]
children_index = functions.read_csv(children_index_path)
children_index = functions.transform_to_int(children_index)
children_index = children_index[0]
children = functions.read_csv(children_path)
children = functions.transform_to_float(children)

new_data = functions.replace(original_data, children, parents_index, children_index)

# 新しい潜在ベクトル群をcsvファイルに書き込む
functions.write_csv(csv_dir + args.output_dir + "/latent_vectors.csv", new_data)
