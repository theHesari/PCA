import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from data import *


def pca_visualization(df):
    def get_features():
        x_values = df.drop(columns=['label'], axis=1).values
        return x_values

    def get_labels():
        return df['label']

    def get_components():
        labels = {
            str(i): f"PCA {i + 1} ({var:.2f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }
        return labels.values()

    def convert_to_dataframe(data):
        return pd.DataFrame(data=data, columns=get_components())

    def plot_pca():
        plt.figure(figsize=(10, 10))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=14)
        plt.xlabel(f'Principal Component - 1', fontsize=20)
        plt.ylabel(f'Principal Component - 2', fontsize=20)
        plt.title(f"Principal Component Analysis with {2} components", fontsize=20)

        targets = y.unique()
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'][:len(targets)]  # Adjust colors based on number of unique labels

        for target, color in zip(targets, colors):
            indices = y == target
            plt.scatter(dataframe.loc[indices, f'feature0'],
                        dataframe.loc[indices, f'feature1'],
                        c=color, s=50, label=target)

        plt.legend(prop={'size': 15})
        plt.show()

    def plotly_pca():
        if n_components == 1:
            fig = px.scatter(dataframe, color=y)
            fig.show()
        else:
            fig = px.scatter_matrix(
                dataframe,
                dimensions=dataframe.columns,
                color=y
            )
            fig.update_traces(diagonal_visible=False)
            fig.show()

    X = get_features()
    y = get_labels()
    n_components = 3
    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X)
    dataframe = convert_to_dataframe(X_pca)
    print(dataframe.head())
    plotly_pca()


if __name__ == '__main__':
    data = load_numeric_data()
    pca_visualization(data)
