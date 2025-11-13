import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np

# Check Keras version for compatibility
try:
    # Keras 3.x
    from keras.saving import register_keras_serializable
    KERAS_3 = True
except (ImportError, AttributeError):
    # Keras 2.x (TensorFlow 2.x)
    try:
        from tensorflow.keras.utils import register_keras_serializable
        KERAS_3 = False
    except ImportError:
        # Older TensorFlow versions - use custom_objects instead
        register_keras_serializable = lambda: lambda cls: cls
        KERAS_3 = False


@register_keras_serializable(package="Custom", name="MinLSTMCell")
class MinLSTMCell(layers.Layer):
    """
    Minimal LSTM Cell implementation
    A simplified version of LSTM with fewer gates for efficiency
    """
    def __init__(self, units, **kwargs):
        super(MinLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = [units, units]  # [hidden_state, cell_state]
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Forget gate
        self.Wf = self.add_weight(
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            name='Wf'
        )
        self.Uf = self.add_weight(
            shape=(self.units, self.units),
            initializer='orthogonal',
            name='Uf'
        )
        self.bf = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bf'
        )
        
        # Input gate
        self.Wi = self.add_weight(
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            name='Wi'
        )
        self.Ui = self.add_weight(
            shape=(self.units, self.units),
            initializer='orthogonal',
            name='Ui'
        )
        self.bi = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bi'
        )
        
        # Cell candidate
        self.Wc = self.add_weight(
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            name='Wc'
        )
        self.Uc = self.add_weight(
            shape=(self.units, self.units),
            initializer='orthogonal',
            name='Uc'
        )
        self.bc = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bc'
        )
        
        super(MinLSTMCell, self).build(input_shape)
    
    def call(self, inputs, states):
        h_prev, c_prev = states
        
        # Forget gate
        f = tf.sigmoid(tf.matmul(inputs, self.Wf) + tf.matmul(h_prev, self.Uf) + self.bf)
        
        # Input gate
        i = tf.sigmoid(tf.matmul(inputs, self.Wi) + tf.matmul(h_prev, self.Ui) + self.bi)
        
        # Cell candidate
        c_candidate = tf.tanh(tf.matmul(inputs, self.Wc) + tf.matmul(h_prev, self.Uc) + self.bc)
        
        # New cell state
        c = f * c_prev + i * c_candidate
        
        # New hidden state (simplified - no output gate)
        h = tf.tanh(c)
        
        return h, [h, c]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # Handle None dtype by defaulting to float32
        if dtype is None:
            dtype = tf.float32
        return [tf.zeros((batch_size, self.units), dtype=dtype),
                tf.zeros((batch_size, self.units), dtype=dtype)]
    
    def get_config(self):
        """Return config for serialization"""
        config = super(MinLSTMCell, self).get_config()
        config.update({
            'units': self.units,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create layer from config"""
        return cls(**config)


@register_keras_serializable(package="Custom", name="BiMinLSTM")
class BiMinLSTM(Model):
    """
    Bidirectional Minimal LSTM Model for Hate Speech Detection
    """
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=64, num_classes=3, 
                 max_len=100, dropout_rate=0.3, **kwargs):
        """
        Initialize Bi-MinLSTM model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units
            num_classes: Number of output classes
            max_len: Maximum sequence length
            dropout_rate: Dropout rate for regularization
        """
        super(BiMinLSTM, self).__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        
        # Embedding layer
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len,
            mask_zero=True
        )
        
        # Spatial Dropout for embeddings
        self.spatial_dropout = layers.SpatialDropout1D(dropout_rate)
        
        # Bidirectional MinLSTM layer
        self.bi_minlstm = layers.Bidirectional(
            layers.RNN(MinLSTMCell(lstm_units), return_sequences=False),
            merge_mode='concat'
        )
        
        # Dropout layer
        self.dropout = layers.Dropout(dropout_rate)
        
        # Dense layer
        self.dense = layers.Dense(32, activation='relu')
        
        # Dropout layer
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # Output layer
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        """Forward pass"""
        x = self.embedding(inputs)
        x = self.spatial_dropout(x, training=training)
        x = self.bi_minlstm(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.dropout2(x, training=training)
        output = self.output_layer(x)
        return output
    
    def get_config(self):
        """Return config for serialization"""
        config = super(BiMinLSTM, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'num_classes': self.num_classes,
            'max_len': self.max_len,
            'dropout_rate': self.dropout_rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from config"""
        return cls(**config)
    
    def model(self):
        """Build and return the model"""
        x = keras.Input(shape=(self.max_len,))
        return Model(inputs=[x], outputs=self.call(x))


def create_bi_min_lstm_model(vocab_size, embedding_dim=128, lstm_units=64, 
                              num_classes=3, max_len=100, dropout_rate=0.3):
    """
    Create and compile Bi-MinLSTM model
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        lstm_units: Number of LSTM units
        num_classes: Number of output classes
        max_len: Maximum sequence length
        dropout_rate: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    model = BiMinLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        num_classes=num_classes,
        max_len=max_len,
        dropout_rate=dropout_rate
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Build the model
    model.build(input_shape=(None, max_len))
    
    print("\nModel Architecture:")
    print("="*60)
    model.summary()
    print("="*60)
    
    return model


# Custom objects for model loading (backward compatibility)
def get_custom_objects():
    """
    Return custom objects dictionary for model loading
    Useful for loading saved models
    """
    return {
        'MinLSTMCell': MinLSTMCell,
        'BiMinLSTM': BiMinLSTM,
    }