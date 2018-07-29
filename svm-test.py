import unittest
from sklearn import datasets
import numpy as np
import glob
from svm import SVM


def add_ones(X):
    return np.column_stack((np.ones(len(X)), X))


def generate_data(data_type, n_samples=100):
    np.random.seed(42)
    if data_type == "blobs":
        X, y = datasets.make_blobs(
            n_samples=n_samples,
            centers=[[2, 2], [1, 1]],
            cluster_std=0.4
        )
    elif data_type == "circle":
        X = (np.random.rand(n_samples, 2) - 0.5) * 20
        y = (np.sqrt(np.sum(X ** 2, axis=1)) > 8).astype(int)

    X = add_ones(X)
    return X, y


def get_text_data(origin="text-data"):
    dirs = glob.glob(origin + "/*")
    X, y = [], []
    for i, d in enumerate(dirs):
        files = glob.glob(d + "/*")
        for file_name in files:
            with open(file_name, "rt", encoding="utf8") as file:
                X.append(" ".join(file.readlines()))
        y.extend([i] * len(files))
    return np.array(X), np.array(y)


class SVMTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_blob_coefficients(self):
        X, y = generate_data("blobs")
        svm = SVM(C=1, rate=0.001, epochs=5000, kernel="linear")
        svm.fit(X, y)
        ref = np.array([3.8, -1.6, -1.0])

        np.testing.assert_array_almost_equal(ref, svm.get_weights(), decimal=1)

    def test_blob_prediction(self):
        X, y = generate_data("blobs")
        svm = SVM(C=1, rate=0.001, epochs=5000, kernel="linear")
        svm.fit(X, y)
        errors = np.sum(np.abs(svm.predict(X) - y))
        self.assertLess(errors, 8)

    def test_circle_prediction(self):
        X, y = generate_data("circle", n_samples=200)
        svm = SVM(C=1, rate=0.001, epochs=5000, kernel="rbf")
        svm.fit(X, y)
        errors = np.sum(np.abs(svm.predict(X) - y))
        self.assertLess(errors, 20)

    def test_text_training(self):
        X, y = get_text_data("text-data")
        svm = SVM(C=1, rate=0.001, epochs=100, kernel="text")
        svm.fit(X, y)
        errors = np.sum(np.abs(svm.predict(X) - y))
        self.assertEqual(errors, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)