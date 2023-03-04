import collections
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.utils import io_utils
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import CSVLogger, EarlyStopping, Callback
from tensorflow.python.keras.layers import Dense, Layer
from tensorflow.python.keras.losses import mean_squared_error

from .constants import TIME, TIMESTEPS_PER_TRAJECTORY, CACHE, MAX_LINES
from .utils import _rand_alphanumeric, make_pca, _apply_pca_to_grid


class Recorder(Callback):
    """
    Adapted from CSVLogger
    """
    def __init__(self, x, filename, pca=None, max_lines=MAX_LINES, separator=",", append=False):
        x = x[:max_lines]
        self.pca = pca
        self.gold_filename = f"{filename}-gold"
        if self.pca is not None:
            x_proj = _apply_pca_to_grid(x, pca)
            np.save(self.gold_filename, x_proj)  # saves x projected onto PCA subspace as a *.npy file
        else:
            np.save(self.gold_filename, x)

        x0 = x[:, 0, :]
        assert x0.ndim == 2
        x0 = tf.convert_to_tensor(np.expand_dims(x0, axis=1))
        self.x = [x, x0]  # NOTE: x isn't really used besides as a dummy input

        # CSVLogger inheritance
        self.sep = separator
        self.filename = io_utils.path_to_string(f"{filename}-preds.csv")
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        super().__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if tf.io.gfile.exists(self.filename):
                with tf.io.gfile.GFile(self.filename, "r") as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"
        self.csv_file = tf.io.gfile.GFile(self.filename, mode)

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.x, verbose=0)
        if self.pca is not None:
            pred = _apply_pca_to_grid(pred, self.pca)

        logs = {"trajs": pred.flatten()}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif (
                    isinstance(k, collections.abc.Iterable)
                    and not is_zero_dim_ndarray
            ):
                return ', '.join(map(str, k))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict(
                (k, logs[k]) if k in logs else (k, "NA") for k in self.keys
            )

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["epoch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


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
                 input_dim,
                 intrinsic_dim=None,
                 encoder_hidden_widths=(80, 80),
                 decoder_hidden_widths=(80, 80),
                 activation="relu",
                 optimizer="adam",
                 regularizer="l2",
                 alpha1=0.01,
                 alpha2=0.01,
                 ):
        """
        The main idea for a Koopman network is to solve nonlinear systems using neural networks. The idea is that
        many nonlinear systems have some "intrinsic" linearity that we can learn. The model first encodes our
        input system as a linear system in some intrinsic coordinate space. For example, raw neural data can be
        extremely high-dimensional and non-linear, but the 4-dimensional nonlinear Hodgkin-Huxley model underlies spiking
        neuron activity. The network then predicts how the system evolves in the intrinsic space before decoding that
        prediction and returning it to the user.

        :param intrinsic_dim: the dimension of the linear space that we encode our system into
        """
        # establish cache
        Path(CACHE).mkdir(exist_ok=True)
        self._filename = f"{CACHE}/{_rand_alphanumeric()}"

        # 'private' attributes
        self._input_dim = input_dim
        self._intrinsic_dim = intrinsic_dim or input_dim
        self._regularizer = regularizer
        self._activation = activation

        # callbacks
        self._autoencoder_callbacks = [
            EarlyStopping(monitor='loss',
                          patience=10,
                          min_delta=1e4,
                          restore_best_weights=True)
        ]
        self._model_callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=10,
                          min_delta=1e5,
                          restore_best_weights=True),
            CSVLogger(f"{self._filename}.csv")
        ]

        # models
        self._encoder = self._build_coder("encoder", (None, self._input_dim), encoder_hidden_widths,
                                          self._intrinsic_dim)
        self._decoder = self._build_coder("decoder", (None, self._intrinsic_dim), decoder_hidden_widths,
                                          self._input_dim)
        self.autoencoder = self._build_autoencoder()
        self._koopman = self._build_koopman(self._intrinsic_dim)
        self.model = self._build_model(alpha1, alpha2)  # alpha3 is implicitly specified in the regularizer

        # compile
        self.autoencoder.compile(optimizer=optimizer, loss="mse")
        self.model.compile(optimizer=optimizer, loss=None)  # custom loss and other metrics added during _build_model()

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

    def _make_recorders(self, record, record_dim=-1):
        trajs = np.array([elem[0] for elem in record])
        pca = None if record_dim == -1 else make_pca(trajs, record_dim)
        recorders = [Recorder(x, f"{self._filename}-{label}", pca) for x, label in record]
        return recorders

    def train_autoencoder(self, trajectories, epochs, batch_size, verbose="auto"):
        x = tf.convert_to_tensor(trajectories)
        self.autoencoder.fit(x, x, callbacks=self._autoencoder_callbacks,
                             epochs=epochs, batch_size=batch_size, verbose=verbose)

    def train_model(self,
                    trajectories,
                    epochs,
                    batch_size,
                    validation_split=0.2,
                    validation_freq=1,
                    verbose="auto",
                    record=None,
                    record_dim=-1,  # ignored if record is None or record_dim=-1
                    ):
        if record is None:
            record = []
        x_true = tf.convert_to_tensor(trajectories)
        x0 = x_true[:, :1, :]

        recorders = self._make_recorders(record, record_dim)
        self._model_callbacks += recorders

        self.model.fit([x_true, x0],
                       x_true,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=validation_split,
                       validation_freq=validation_freq,
                       callbacks=self._model_callbacks,
                       verbose=verbose)

    def autoencoder_predict(self, trajs: np.ndarray, verbose="auto"):
        x = tf.convert_to_tensor(trajs)
        return self.autoencoder.predict(x, verbose=verbose)

    def model_predict(self, x0: np.ndarray, verbose="auto"):
        assert x0.ndim == 2
        num_examples, dim = x0.shape
        x0 = tf.convert_to_tensor(np.expand_dims(x0, axis=1))
        dummy = np.ones((num_examples, TIMESTEPS_PER_TRAJECTORY, dim))
        return self.model.predict([dummy, x0], verbose=verbose)

    def model_evaluate(self, x0: np.ndarray, x, verbose="auto"):
        """
        # TODO: write a general evaluate method for evaluating on an arbitrary number of timesteps
        """
        assert x0.ndim == 2
        num_examples, dim = x0.shape
        x0 = tf.convert_to_tensor(np.expand_dims(x0, axis=1))
        dummy = np.ones((num_examples, TIMESTEPS_PER_TRAJECTORY, dim))
        return self.model.evaluate([dummy, x0], x, return_dict=True, verbose=verbose)

    def train(self,
              trajectories,
              autoencoder_epochs,
              autoencoder_batch_size,
              model_epochs,
              model_batch_size,
              validation_split=0.2,
              validation_freq=1,
              record=None,
              record_dim=-1,
              verbose="auto"):
        # dimensions = samples, n steps, dim
        if record is None:
            record = []
        self.train_autoencoder(trajectories,
                               autoencoder_epochs,
                               autoencoder_batch_size,
                               verbose=verbose)
        self.train_model(trajectories,
                         model_epochs,
                         model_batch_size,
                         validation_split=validation_split,
                         validation_freq=validation_freq,
                         verbose=verbose,
                         record=record,
                         record_dim=record_dim
                         )

    def predict(self, x0: np.ndarray, num_timesteps=TIMESTEPS_PER_TRAJECTORY, verbose="auto"):
        """
        Wrapper for model_predict that lets you predict arbitrary number of timesteps
        """
        q, r = divmod(num_timesteps, TIMESTEPS_PER_TRAJECTORY)
        result = []
        for _ in range(q + (r != 0)):
            result.append(self.model_predict(x0, verbose=verbose))
            x0 = result[-1][:, -1, :]
        if r != 0:
            result[-1] = result[-1][:r]
        result = np.concatenate(result)
        return result

    def save(self, filepath):
        self.autoencoder.save(f"auto_{filepath}")
        self.model.save(f"model_{filepath}")
