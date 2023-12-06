import argparse
import json

from sklearn.manifold import TSNE
#from numpy import reshape
import seaborn as sns
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt

def create_viz(args):

    with open(args.dataset_metadata_path) as f:
        df_metadata = json.load(f)

    df = pd.read_csv(args.dataset_path, index_col='index', dtype = str)

    #seperate out class from label
    y = df['class_label']
    X = df.drop('class_label', axis=1)

    tsne = TSNE(n_components=2, verbose=1, random_state=20, perplexity=30, n_iter=1000)
    z = tsne.fit_transform(X)
    df = pd.DataFrame()
    df["y"] = y.replace(df_metadata)
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    #palette=sns.color_palette("hls", 15),
                    palette="tab20",
                    data=df).set(title="Fruit_Smell T-SNE")

    #plt.show()
    plt.savefig(args.output_image_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Smell Dataset Visualization')

    # general args
    parser.add_argument('--project_name', type=str, default='smell_dataset_viz', help='name of project')
    parser.add_argument('--dataset_path', type=str, default='smell_dataset.csv', help='location of dataset')
    parser.add_argument('--dataset_metadata_path', type=str, default='smell_dataset_metadata.json', help='location of dataset')

    parser.add_argument('--output_image_path', type=str, default='results/smell_dataset_viz.jpg', help='location of dataset')


    args = parser.parse_args()

    create_viz(args)