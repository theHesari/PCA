from sklearn.datasets import load_breast_cancer
# from keras.datasets import cifar10
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_numeric_data():
    def clean_labels():
        label_map = {0: 'Benign', 1: 'Malignant'}
        return dataframe['label'].replace(label_map, inplace=True)

    # Load Data
    breast_dataset = load_breast_cancer()
    data = breast_dataset.data
    data = StandardScaler().fit_transform(data)
    labels = breast_dataset.target

    # Reshape and Concatenate data
    labels = np.reshape(labels, (569, 1))
    dataset = np.concatenate([data, labels], axis=1)
    features = breast_dataset.feature_names
    features = np.append(features, 'label')

    # Convert to DataFrame
    dataframe = pd.DataFrame(dataset, columns=features)
    clean_labels()

    return dataframe


def load_images():
    pass


if __name__ == '__main__':
    df = load_numeric_data()
    print(df.head())
