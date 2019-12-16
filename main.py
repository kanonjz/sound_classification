import argparse
import nn
import svm
import knn
import random_forest
import logistic_regression
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

AUDIO_PATH = '../ESC-50-master/audio/'


def get_numpy_array(features_df):
    X = np.array(features_df.feature.tolist())
    y = np.array(features_df.class_label.tolist())

    # encode classification labels
    le = LabelEncoder()
    # one hot encoded labels
    yy = to_categorical(le.fit_transform(y))
    return X, yy, le


def get_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    k = np.array([np.where(r == 1)[0][0] for r in y_test])
    return X_train, X_test, y_train, y_test


def converter(instr):
    return np.fromstring(instr[1:-1], sep=' ')


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        help="support following method: svm, lr, knn, rf, cnn, mlp")
    args = parser.parse_args()

    # extract features
    print("Extracting features..")
    features_df = pd.read_csv('features.csv', converters={'feature': converter})

    # 1. SVM
    if args.method == "svm":
        svm.svm(features_df)

    # 2. Logistic Regression
    elif args.method == "lr":
        logistic_regression.logistic_regression(features_df)

    # 3. k-NN
    elif args.method == "knn":
        knn.knn(features_df)

    # 4. Random Forest
    elif args.method == "rf":
        random_forest.random_forest(features_df)

    # 5. Convolutional Neural Network
    elif args.method == "cnn":

        # convert into numpy array
        X, y, le = get_numpy_array(features_df)

        # split into training and testing data
        X_train, X_test, y_train, y_test = get_train_test(X, y)
        num_labels = y.shape[1]
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

        # create model architecture
        model = nn.create_cnn(num_labels)

        # train model
        print("Training..")
        nn.train(model, X_train, X_test, y_train, y_test, "./saved_models/trained_cnn.h5")

        # compute test loss and accuracy
        test_loss, test_accuracy = nn.compute(X_test, y_test, "./saved_models/trained_cnn.h5")
        print("Test loss", test_loss)
        print("Test accuracy", test_accuracy)

        # predicting using trained model with any test file in dataset
        nn.predict(AUDIO_PATH + '2-100786-A-1.wav', le, "./saved_models/trained_cnn.h5")

    # 6. multilayer perceptron
    elif args.method == "mlp":

        # convert into numpy array
        X, y, le = get_numpy_array(features_df)

        # split into training and testing data
        X_train, X_test, y_train, y_test = get_train_test(X, y)
        num_labels = y.shape[1]

        # create model architecture
        model = nn.create_mlp(num_labels)

        # train model
        print("Training..")
        nn.train(model, X_train, X_test, y_train, y_test, "./saved_models/trained_mlp.h5")

        # compute test loss and accuracy
        test_loss, test_accuracy = nn.compute(X_test, y_test, "./saved_models/trained_mlp.h5")
        print("Test loss", test_loss)
        print("Test accuracy", test_accuracy)

        # predicting using trained model with any test file in dataset
        nn.predict(AUDIO_PATH + '2-100786-A-1.wav', le, "./saved_models/trained_mlp.h5")
