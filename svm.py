import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib.backends.backend_pdf as pdfs
import zlib
from numpy.linalg import norm
from sklearn import datasets


def add_ones(X):
    """
    Adds column of ones to the left of the matrix X.
    :param X: matrix
    :return: extended matrix
    """
    return np.column_stack((np.ones(len(X)), X))


def generate_data(data_type, n_samples=100):
    """
    Generates data for testing.
    :param data_type: type of data
    :param n_samples: number of samples
    :return: data matrix X with corresponding classes y
    """
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


def get_text_data(origin='text-data'):
    """
    Reads text data and saves it to X and classes to y.
    :param origin: relative path to text files
    :return: data matrix X with corresponding classes y
    """
    dirs = glob.glob(origin + "/*")
    X, y = [], []
    for i, d in enumerate(dirs):
        files = glob.glob(d + "/*")
        for file_name in files:
            with open(file_name, "rt", encoding="utf8") as file:
                X.append(" ".join(file.readlines()))
        y.extend([i] * len(files))
    return np.array(X), np.array(y)


def shuffle_copies(a, b):
    """
    Shuffles data (same permutation for a and b) and returns their copies.
    :param a: array or matrix
    :param b: array or matrix
    :return: shuffled copies of a and b
    """
    p = np.random.permutation(len(a))
    return a[p], b[p]


def draw(name, n_samples, kernel, linspace):
    """
    Draws examples and shows boundary between them. Saves plots to pdf files.
    :param name: type of data
    :param n_samples: number of samples
    :param kernel: name of the kernel
    :param linspace: (start, end) of the linear space
    :return: None
    """
    X, y = generate_data(name, n_samples=n_samples)
    svm = SVM(C=1, rate=0.001, epochs=5000, kernel=kernel)
    svm.fit(X, y)

    colors = list(map(lambda x: 'red' if x == 0 else 'blue', y))
    sizes = 15 * (svm.coef_ + 1)

    points = 50
    l = np.linspace(linspace[0], linspace[1], points)
    m = np.meshgrid(l, l)
    X_test = np.zeros(shape=(points ** 2, 2))
    X_test[:, 0] = m[0].reshape(points ** 2)
    X_test[:, 1] = m[1].reshape(points ** 2)

    predicted = svm.predict(add_ones(X_test))
    mask = np.where(predicted == 1)[0]
    X_test = X_test[mask]

    fig = plt.figure()
    plt.scatter(X[:, 1], X[:, 2], s=sizes, c=colors, marker='.')
    plt.scatter(X_test[:, 0], X_test[:, 1], s=100, c='gold', alpha=0.3, edgecolors='none', marker='.')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('SVM - ' + name)
    plt.show()

    pdf = pdfs.PdfPages('blob.pdf') if name == 'blobs' else pdfs.PdfPages('circle.pdf')
    pdf.savefig(fig)
    pdf.close()


class SVM:
    """
    SVM implementation for binary classification.
    Three different kernels implemented: linear, rbf and text.
    """
    def __init__(self, C, kernel, epochs, rate):
        self.C = C
        self.kernel = kernel
        self.epochs = epochs
        self.rate = rate

        self.coef_ = None
        self.X, self.y, self.m = None, None, None

    @staticmethod
    def linear_kernel(A, B):
        """
        Implementation of linear kernel.
        :param A: matrix
        :param B: matrix
        :return: linear kernel matrix
        """
        return A.dot(B.T)

    @staticmethod
    def rbf_kernel(A, B, sigma=1):
        """
        Implementation of rbf kernel.
        :param A: matrix
        :param B: matrix
        :param sigma: parameter of rbf kernel
        :return: rbf kernel matrix
        """
        X = np.zeros(shape=(len(A), len(B)))
        for i, a in enumerate(A):
            X[i] = np.apply_along_axis(norm, 1, B - a) ** 2
        return np.exp(- X / sigma)

    @staticmethod
    def text_kernel(A, B):
        """
        Implementation of text kernel.
        :param A: matrix
        :param B: matrix
        :return: text kernel matrix
        """
        X = np.zeros(shape=(len(A), len(B)))
        compress_B = []
        for i, b in enumerate(B):
            compress_B.append(zlib.compress(b.tobytes()))
        for i, a in enumerate(A):
            compress_a = zlib.compress(a.tobytes())
            for j, b in enumerate(B):
                compress_b = compress_B[j]
                compress_ab = zlib.compress(np.array('\n'.join([a, b])).tobytes())
                compress_ba = zlib.compress(np.array('\n'.join([b, a])).tobytes())
                distance = (1 / 2) * (((len(compress_ab) - len(compress_a)) / len(compress_a)) +
                                      ((len(compress_ba) - len(compress_b)) / len(compress_b)))
                similarity = 10 - distance
                X[i, j] = similarity
        return X

    def calculate_kernel(self, A, B):
        """
        Calculates and returns kernel according to self.kernel.
        :param A: matrix
        :param B: matrix
        :return: kernel matrix
        """
        if self.kernel == 'linear':
            return self.linear_kernel(A, B)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(A, B)
        elif self.kernel == 'text':
            return self.text_kernel(A, B)
        else:
            raise ValueError('Kernel ' + self.kernel + ' is not supported.')

    def add_axis(self):
        """
        Adds axis to two class attributes.
        :return: None
        """
        self.y = self.y[:, np.newaxis]
        self.coef_ = self.coef_[:, np.newaxis]

    def fit(self, X, y):
        """
        Fits the SVM model. Finds and saves Lagrangian coefficients.
        :param X: data matrix
        :param y: binary classes for each data point
        :return: array of Lagrangian coefficients
        """
        self.X, self.y = shuffle_copies(X, y)
        self.y = np.array(list(map(lambda x: -1 if x == 0 else 1, self.y)))  # change 0 in self.y to -1
        self.m = len(y)

        kernel = self.calculate_kernel(self.X, self.X)
        alphas = np.zeros(self.m)
        for i in range(self.epochs):
            deltas = self.rate * (1 - self.y * (alphas * self.y).dot(kernel))
            alphas = np.minimum(self.C, np.maximum(0, alphas + deltas))

        self.coef_ = alphas
        self.add_axis()
        return self.coef_

    def predict(self, X):
        """
        Predicts binary classes for input data matrix X.
        :param X: matrix
        :return: array of predicted classes
        """
        res = self.calculate_kernel(X, self.X).dot(self.coef_ * self.y)
        return np.array(list(map(lambda x: 1 if x >= 0 else 0, res)))

    def get_weights(self):
        """
        Returns weights (w) of the SVM model.
        :return: weights (w)
        """
        return np.sum((self.coef_ * self.y) * self.X, axis=0)


if __name__ == '__main__':
    # test SVM implementation on real text data
    X, y = get_text_data()
    X_train = np.concatenate((X[:20], X[25:45]))
    X_test = np.concatenate((X[20:25], X[45:]))
    y_train = np.concatenate((y[:20], y[25:45]))
    y_test = np.concatenate((y[20:25], y[45:]))

    svm = SVM(C=1, rate=0.001, epochs=100, kernel="text")
    svm.fit(X_train, y_train)
    predicted = svm.predict(X_test)
    ok = len(np.where(predicted == y_test)[0])
    print('Classification accuracy for ' + str(len(predicted)) + ' test examples: ' + str(ok) + '/' +
          str(len(predicted)) + ', CA: ' + str((ok / len(predicted)) * 100) + ' %')

    # draw blobs and circles
    draw('blobs', 100, 'linear', (0, 3))
    draw('circle', 200, 'rbf', (-10, 10))
