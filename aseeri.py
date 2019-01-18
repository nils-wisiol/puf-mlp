import numpy as np
from itertools import product
from time import sleep, time
from sklearn.neural_network import MLPClassifier


def generate_challenges(n, N):
    size = n * N
    print("generating %i %i-bit challenges, that's about %.1fMB (one byte per challenge bit)" %
          (N, n, size / 1024**2))

    return 2 * np.random.randint(0, 2, (N, n), dtype=np.int8) - 1


def transform_id(challenges, k):
    (N, n) = challenges.shape
    print("transforming with identity function from %s to %s, using no additional memory" % ((N, n), (N, k, n)))
    return np.transpose(np.broadcast_to(challenges, (k, N, n)), axes=(1, 0, 2))


def accuracy(a, b, N=1000):
    cs = generate_challenges(n, N)
    return .5 * np.average([np.sign(a(cs) * b(cs))]) + .5


def mlp_learn_xor_arbiter_puf(n, N, k):
    print("running MLP on simulated XOR Arbiter PUF for (N, k, n) = %s" % ((N, k, n),))
    print("max memory usage should be about %3.2fGB, that's" % (((n * N) + (N * k * 8) + (N * 8)) / 1024 ** 3))
    print("- raw challenges:       % 3.2f GB, O(nN) " % ((n * N) / 1024 ** 3))
    print("- individual responses: % 3.2f GB, O(8Nk)" % ((N * k * 8) / 1024 ** 3))
    print("- combined responses:   % 3.2f GB, O(8N)" % ((N * 8) / 1024 ** 3))

    challenges = generate_challenges(n, N)
    transformed_challenges = transform_id(challenges, k)

    weights = np.random.RandomState(seed=3).normal(300, 40, (k, n))

    # begin eval
    eval_step_1 = np.einsum('ji,...ji->...j', weights, transformed_challenges, optimize=True)
    eval_step_2 = np.prod(eval_step_1, axis=1)
    del eval_step_1
    responses = np.sign(eval_step_2)
    del eval_step_2

    print("starting learner")
    start = time()
    clf = MLPClassifier(
        solver='adam',
        alpha=1e-5,
        hidden_layer_sizes=(2 ** k, 2 ** k, 2 ** k),
        random_state=2,
        learning_rate_init=10e-3,
        batch_size=1000,
        shuffle=False,
    )
    clf.partial_fit(challenges, responses, classes=[-1, 1])
    print("learned one iteration in %.2fs, accuracy %.3f" % (
        time() - start,
        accuracy(
            lambda cs: np.sign(
                np.prod(np.einsum('ji,...ji->...j', weights, transform_fixed_permutation(cs, k), optimize=True),
                        axis=1)),
            lambda cs: clf.predict(cs)
        )
    ))


n = 64
mlp_learn_xor_arbiter_puf(n, int(  .4 * 10e6), 4)
mlp_learn_xor_arbiter_puf(n, int(  .8 * 10e6), 5)
mlp_learn_xor_arbiter_puf(n, int( 2   * 10e6), 6)
mlp_learn_xor_arbiter_puf(n, int( 5   * 10e6), 7)
mlp_learn_xor_arbiter_puf(n, int(30   * 10e6), 8)
