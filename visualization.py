import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from data import *


def pca_visualization(df):
    def get_features():
        return df.drop(columns=['label'], axis=1).values

    def get_labels():
        return df['label']

    def get_n_components(data):
        pca = PCA()
        X = pca.fit_transform(data)
        # Calculate cumulative explained variance ratio
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        # Find the number of components that explain 90% of variance
        optimum = np.argmax(cumulative_variance_ratio >= 0.9) + 1

        return optimum

    def convert_to_dataframe(data):
        columns = ['feature' + str(i) for i in range(data.shape[1])]
        return pd.DataFrame(data=data, columns=columns)

    def plot_pca(components=2):
        plt.figure(figsize=(10, 10))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=14)
        plt.xlabel(f'Principal Component - 1', fontsize=20)
        plt.ylabel(f'Principal Component - 2', fontsize=20)
        plt.title(f"Principal Component Analysis with {components} components", fontsize=20)

        targets = y.unique()
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'][:len(targets)]  # Adjust colors based on number of unique labels

        for target, color in zip(targets, colors):
            indicesToKeep = y == target
            plt.scatter(dataframe.loc[indicesToKeep, f'feature0'],
                        dataframe.loc[indicesToKeep, f'feature1'],
                        c=color, s=50, label=target)

        plt.legend(prop={'size': 15})
        plt.show()

    X = get_features()
    y = get_labels()
    n_components = 2
    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X)
    dataframe = convert_to_dataframe(X_pca)
    # print(dataframe.head())
    plot_pca()


if __name__ == '__main__':
    data = load_numeric_data()
    pca_visualization(data)
