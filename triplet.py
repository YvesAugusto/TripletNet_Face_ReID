import numpy as np
import numpy.linalg as nl
import tensorflow as tf
class TripletModel(tf.keras.Model):

    def __init__(self, model):
        super(TripletModel, self).__init__()
        self.model = model
        self.n_hard_batches = 3

    def compile(self, loss, optimizer):
        super(TripletModel, self).compile()
        self.loss_fn = loss
        self.optimizer = optimizer

    def get_hard_triplets(self, triplets_batch):

        sorted_batches = sorted(
            triplets_batch,
            key=lambda x: nl.norm(x[0] - x[1]) - nl.norm(x[0] - x[2]),
            reverse=True
        )
        return np.array(sorted_batches[:self.n_hard_batches])

    def train_gen(self, data, batch_sz):
        X = []
        Y = []

        for _ in range(batch_sz):
            X_batch = []
            for __ in range(50):
                i = tf.experimental.numpy.random.randint(0, data.shape[0]) - 1
                user = data[i]
                j = tf.experimental.numpy.random.randint(0, user.shape[0]) - 1
                # anchor image
                anchor = user[j]
                k = tf.experimental.numpy.random.randint(0, user.shape[0]) - 1
                while (k == j):
                    k = tf.experimental.numpy.random.randint(0, user.shape[0]) - 1
                # positive image
                positive = user[k]

                # now get some other user
                k = tf.experimental.numpy.random.randint(0, data.shape[0]) - 1
                while (k == i):
                    k = tf.experimental.numpy.random.randint(0, data.shape[0]) - 1

                user_ = data[k]
                j = tf.experimental.numpy.random.randint(0, user_.shape[0]) - 1
                # negative image
                negative = user_[j]

                X_batch.append([anchor, positive, negative])

            hard_triplets = self.get_hard_triplets(X_batch)
            len_y = hard_triplets.shape[0]
            X += list(hard_triplets)
            Y += [np.zeros(3) for i in range(len_y)]

        return tf.cast(X, tf.float32), Y

    # @tf.function
    def train_step(self, data):
        X, Y = self.train_gen(data, 15)

        with tf.GradientTape(persistent=True) as tape:
            input_A, input_P, input_N = tf.split(X, [1, 1, 1], axis=1)
            shape = tf.shape(input_A)
            shape = [shape[0], shape[2], shape[3], shape[4]]
            input_A = tf.reshape(input_A, shape)
            input_P = tf.reshape(input_P, shape)
            input_N = tf.reshape(input_N, shape)
            # print(f'Got inputs: {tf.shape(input_A)}, {tf.shape(input_P)}, {tf.shape(input_N)}')
            anchor = self.model(input_A)
            # print(f'Anchor output: {tf.shape(anchor)}')
            positive = self.model(input_P)
            # print(f'Positive output: {tf.shape(positive)}')
            negative = self.model(input_N)
            # print(f'Negative output: {tf.shape(negative)}')
            # print(f'-------- Got anchor, positive and negative outputs --------')

            y_pred = tf.cast([anchor, positive, negative], tf.float32)
            loss = self.loss_fn(y_pred)

        grad_A = tape.gradient(loss, self.model.trainable_variables)
        # print(f'Computed gradients')

        # print(f'Before: {model_A.layers[-1].weights}')
        self.optimizer.apply_gradients(zip(grad_A, self.model.trainable_variables))
        # print(f'After: {model_A.layers[-1].weights}')

        del tape, anchor, positive, negative, X, Y
        print(f'Loss value: {loss}')

        return {'loss': loss}

    def call(self, X):
        anchor, positive, negative = X

        anchor = self.model(anchor)
        positive = self.model(positive)
        negative = self.model(negative)

        return self.loss_fn(
            tf.cast([anchor, positive, negative], tf.float32)
        )