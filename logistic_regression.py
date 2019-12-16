import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA


def pca(X, n_components):
    p = PCA(n_components=n_components)
    new_X = p.fit_transform(X)
    print('Components after PCA: ', len(p.explained_variance_ratio_))
    return new_X


def logistic_regression(features_df):
    X = np.array(features_df.feature.tolist())
    X = pca(X, 40)
    y = np.array(features_df.class_label.tolist())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    clf = LogisticRegressionCV(cv=5, solver='sag', multi_class='multinomial', max_iter=500, penalty='l2')
    clf = clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print("Accuracy", acc*100)