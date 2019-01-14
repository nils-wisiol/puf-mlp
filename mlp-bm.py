from sklearn.neural_network import MLPClassifier
from numpy.random import RandomState
from numpy import dot, prod, zeros, sign, average, seterr, exp, transpose, diagonal, array, \
    repeat, newaxis, ones, clip, einsum, int8, linspace
from numpy.linalg import norm
import time
import matplotlib.pyplot as plt


class LTFArray:

    def __init__(self, n, k, transform, combine, random_state=0):
        self.k = k
        self.n = n
        self.weights = RandomState(seed=random_state).normal(0, 1, (k, n)) * 2**63  # Array creation (k, n)
        self.transform = transform
        self.combine = combine
        self.eval = lambda challenges: self.combine(self.eval_einsum(self.transform(challenges, self.k)))

    def eval_einsum(self, inputs):
        return einsum('ji,...ji->...j', self.weights, inputs, optimize=True)

    def eval_slow(self, inputs):
        return diagonal(
            transpose(
                dot(
                    self.weights,
                    transpose(inputs, axes=(0, 2, 1))
                ),
                axes=(1, 0, 2)
            ),
            axis1=1,
            axis2=2
        )  # Array creation (N, k)

    def val(self, x):
        return prod(self.eval(array([x]))[0])


def learn(challenges, responses, k, **args):
    (N, n) = challenges.shape

    start = time.time()
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(2**k, 2**k, 2**k), **args,
                        random_state=2)
    clf.fit(challenges, responses)
    print('learning time: %.1fs' % (time.time() - start))

    class LearnResult:
        def __init__(self, clf, n, k):
            self.clf = clf
            self.n = n
            self.k = k

        def eval(self, challenges):
            return self.clf.predict(challenges)

    return LearnResult(clf, n, k)


def accuracy(a, b, N=1000):
    assert a.n == b.n and a.k == b.k
    n = a.n
    k = a.k
    cs = RandomState(seed=4).choice((-1.0, 1.0), (N, n))
    return .5 * average([sign(a.eval(cs) * b.eval(cs))]) + .5


def transform_id(challenges, k):
    (N, n) = challenges.shape
    transformed = repeat(challenges, k).reshape(N, n, k)
    return transpose(transformed, axes=(0, 2, 1))


def combine_xor(responses):
    return sign(prod(responses, axis=1))


def simulate_and_learn(n, k, N, random_state=3, **args):
    simulation = LTFArray(n, k, transform_id, combine_xor, random_state=123)
    challenges = RandomState(seed=random_state).choice((-1.0, 1.0), (N, n))
    responses = simulation.eval(challenges)

    model = learn(challenges, responses, k, **args)
    #print(model.weights)
    #return accuracy(model, simulation)
    return model, simulation


(n, k, N) = (64, 4, int(.4 * 10e6))
(model, simulation) = simulate_and_learn(n, k, N, learning_rate_init=0.001)
acc = accuracy(model, simulation)
print("accuracy: %.3f" % acc)
