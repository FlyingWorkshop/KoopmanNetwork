import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import CSVLogger


from .dynamics import TIME, TIMESTEPS_PER_TRAJECTORY


class KoopmanLayer(Layer):
    def __init__(self, dim, kernel_regularizer=None):
        super(KoopmanLayer, self).__init__()
        self.dim = dim
        self.regularizer = kernel_regularizer
        self.kernel = None

    def build(self, _):
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.dim, self.dim],
            regularizer=self.regularizer
        )

    def call(self, inputs):
        matrix_exponentials = tf.convert_to_tensor([tf.linalg.expm(self.kernel * t) for t in TIME])
        trajs = tf.einsum('ijk,lk->lij', matrix_exponentials, inputs[:, 0, :])
        return trajs


class KoopmanNetwork:
    def __init__(self,
                 input_dim, intrinsic_dim=None,
                 encoder_hidden_widths=(80, 80), decoder_hidden_widths=(80, 80),
                 activation="relu", optimizer="adam", regularizer="l2",
                 alpha1=0.01, alpha2=0.01):
        """
        The main idea for a Koopman network is to solve nonlinear systems using neural networks. The idea is that
        many nonlinear systems have some "intrinsic" linearity that we can learn. The model first encodes our
        input system as a linear system in some intrinsic coordinate space. For example, raw neural data can be
        extremely high-dimensional and non-linear, but the 4-dimensional nonlinear Hodgkin-Huxley model underlies spiking
        neuron activity. The network then predicts how the system evolves in the intrinsic space before decoding that
        prediction and returning it to the user.

        :param intrinsic_dim: the dimension of the linear space that we encode our system into
        """

        # 'private' attributes
        self._input_dim = input_dim
        if intrinsic_dim is None:
            self._intrinsic_dim = input_dim
        else:
            self._intrinsic_dim = intrinsic_dim
        self._regularizer = regularizer
        self._activation = activation
        self._autoencoder_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=1, restore_best_weights=True)
        self._model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=10, restore_best_weights=True)

        # models
        self._encoder = self._build_coder("encoder", (None, self._input_dim), encoder_hidden_widths, self._intrinsic_dim)
        self._decoder = self._build_coder("decoder", (None, self._intrinsic_dim), decoder_hidden_widths, self._input_dim)
        self.autoencoder = self._build_autoencoder()
        self._koopman = self._build_koopman(self._intrinsic_dim)
        self.model = self._build_model(alpha1, alpha2)  # alpha3 is implicitly specified in the regularizer

        # compile
        self.autoencoder.compile(optimizer=optimizer, loss="mse")
        self.model.compile(optimizer=optimizer, metrics=[tf.keras.metrics.MeanSquaredError()], loss=None)  # custom loss and other metrics added during _build_model()

    def _build_coder(self, name: str, input_dim, hidden_widths, output_dim):
        inp = Input(shape=input_dim, name=f"{name}_input")
        x = inp
        for i, w in enumerate(hidden_widths):
            x = Dense(w, activation=self._activation, kernel_regularizer=self._regularizer, name=f"{name}_hidden{i}")(x)
        out = Dense(output_dim, activation="linear", kernel_regularizer=self._regularizer, name=f"{name}_output")(x)
        return Model(inp, out, name=name)

    def _build_autoencoder(self):
        """
        encoder -> decoder
        """
        autoencoder_input = Input(shape=(None, self._input_dim), name="autoencoder_input")
        autoencoder_output = self._decoder(self._encoder(autoencoder_input))
        return Model(autoencoder_input, autoencoder_output, name="autoencoder")

    def _build_koopman(self, dim):
        koopman_input = Input(shape=(1, dim), name="koopman_input")
        koopman_output = KoopmanLayer(dim, kernel_regularizer=self._regularizer)(koopman_input)
        return Model(koopman_input, koopman_output, name="koopman")

    def _build_model(self, alpha1, alpha2):
        """
        encoder -> koopman -> decoder
        """
        x_true = Input(shape=(None, self._input_dim), name="x_true")
        x0 = Input(shape=(1, self._input_dim), name="model_input")
        encoded = self._encoder(x0)
        advanced = self._koopman(encoded)
        decoded = self._decoder(advanced)
        model = Model([x_true, x0], decoded, name="model")

        # custom loss function
        mse = tf.keras.losses.MeanSquaredError()
        x0_recon = self._decoder(encoded)
        L_recon = mse(x0, x0_recon)
        L_pred = tf.reduce_sum(mean_squared_error(x_true, decoded)) / TIMESTEPS_PER_TRAJECTORY
        L_lin = tf.reduce_sum(mean_squared_error(self._encoder(x_true), advanced)) / TIMESTEPS_PER_TRAJECTORY
        L_inf = tf.norm(x0 - x0_recon, ord=np.inf) + tf.norm(x_true[:, 1, :] - decoded[:, 1, :], ord=np.inf)
        L = alpha1 * (L_recon + L_pred) + L_lin + alpha2 * L_inf
        model.add_loss(L)

        # add metrics
        model.add_metric(L_recon, name='reconstruction_loss')
        model.add_metric(L_pred, name='state_prediction_loss')
        model.add_metric(L_lin, name='linear_dynamics_loss')
        model.add_metric(L_inf, name='infinity_norm')

        return model

    def _train_autoencoder(self, trajectories, epochs, batch_size, verbose="auto"):
        x = tf.convert_to_tensor(trajectories)
        self.autoencoder.fit(x, x, callbacks=[self._autoencoder_early_stopping],
                             epochs=epochs, batch_size=batch_size, verbose=verbose)

    def _train_model(self, trajectories, epochs, batch_size, filename="", verbose="auto"):
        x_true = tf.convert_to_tensor(trajectories)
        x0 = x_true[:, :1, :]
        callbacks = [self._model_early_stopping]
        if filename:
            callbacks.append(CSVLogger(filename))

        self.model.fit([x_true, x0], x_true, callbacks=callbacks,
                       epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.2)

    def _autoencoder_predict(self, trajs: np.ndarray, verbose="auto"):
        x = tf.convert_to_tensor(trajs)
        return self.autoencoder.predict(x, verbose=verbose)

    def _model_predict(self, x0: np.ndarray, verbose="auto"):
        assert x0.ndim == 2
        num_examples, dim = x0.shape
        x0 = tf.convert_to_tensor(np.expand_dims(x0, axis=1))
        dummy = np.ones((num_examples, TIMESTEPS_PER_TRAJECTORY, dim))
        return self.model.predict([dummy, x0], verbose=verbose)

    def train(self, trajectories, autoencoder_epochs, autoencoder_batch_size, model_epochs, model_batch_size, filename="", verbose="auto"):
        # dimensions = samples, n steps, dim
        self._train_autoencoder(trajectories, autoencoder_epochs, autoencoder_batch_size, verbose=verbose)
        self._train_model(trajectories, model_epochs, model_batch_size, filename=filename, verbose=verbose)

    def predict(self, x0: np.ndarray, num_timesteps=TIMESTEPS_PER_TRAJECTORY, verbose="auto"):
        """
        Wrapper for model_predict that lets you predict arbitrary number of timesteps
        """
        q, r = divmod(num_timesteps, TIMESTEPS_PER_TRAJECTORY)
        result = []
        for _ in range(q + (r != 0)):
            result.append(self._model_predict(x0, verbose=verbose))
            x0 = result[-1][:, -1, :]
        if r != 0:
            result[-1] = result[-1][:r]
        result = np.concatenate(result)
        return result

    def save(self, filepath):
        self.autoencoder.save(f"auto_{filepath}")
        self.model.save(f"model_{filepath}")