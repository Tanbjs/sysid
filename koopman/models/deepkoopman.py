import tensorflow as tf
from tensorflow.keras import Model, Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
from tensorflow.keras.utils import register_keras_serializable # type: ignore

@register_keras_serializable(package='DeepKoopman')
class KoopmanAutoencoder(Model):
    def __init__(self, n_state: int, n_control: int, hidden_e: dict, hidden_d: dict, **kwargs):
        """
        Initialize the KoopmanAutoencoder model.

        Args:
            n_state (int): Dimension of the original state vector x.
            n_control (int): Dimension of the control input vector u.
            hidden_e (dict): Encoder hidden layer config {name: (units, activation)}.
            hidden_d (dict): Decoder hidden layer config {name: (units, activation)}.
            **kwargs: Additional layer kwargs.
        """
        super().__init__(**kwargs)
        self.n_state = n_state
        self.n_control = n_control
        self.hidden_e = hidden_e
        self.hidden_d = hidden_d
        self._build_networks()

    def _build_networks(self):

        # Encoder network
        self.encoder = Sequential(name="encoder")
        self.encoder.add(Input(shape=(self.n_state,), dtype=tf.float32))
        for name, (units, act) in self.hidden_e.items():
            self.encoder.add(Dense(units, activation=act, name=name))

        # Determine latent dimension (encoder output + original state)
        latent_dim = list(self.hidden_e.values())[-1][0] + self.n_state

        # Koopman operator A
        self.A = Sequential(name="A")
        self.A.add(Input(shape=(latent_dim,), dtype=tf.float32))
        self.A.add(Dense(latent_dim, activation="linear", use_bias=False, name="A"))

        # Koopman input matrix B
        self.B = Sequential(name="B")
        self.B.add(Input(shape=(self.n_control,), dtype=tf.float32))
        self.B.add(Dense(latent_dim, activation="linear", use_bias=False, name="B"))

        # Decoder network
        self.decoder = Sequential(name="decoder")
        self.decoder.add(Input(shape=(latent_dim,), dtype=tf.float32))
        for name, (units, act) in self.hidden_d.items():
            self.decoder.add(Dense(units, activation=act, name=name))
        self.decoder.add(Dense(self.n_state, activation='linear', name='reconstruct'))

    def call(self, inputs, training=False):
        x_t, u_t = inputs
        phi_t = tf.concat([x_t, self.encoder(x_t)], axis=-1)
        z_next = self.A(phi_t) + self.B(u_t)
        x_next = self.decoder(z_next)
        return x_next
