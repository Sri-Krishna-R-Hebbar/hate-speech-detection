import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np

# Check Keras version for compatibility
try:
    from keras.saving import register_keras_serializable
    KERAS_3 = True
except (ImportError, AttributeError):
    try:
        from tensorflow.keras.utils import register_keras_serializable
        KERAS_3 = False
    except ImportError:
        register_keras_serializable = lambda package="Custom", name=None: lambda cls: cls
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
        if dtype is None:
            dtype = tf.float32
        return [tf.zeros((batch_size, self.units), dtype=dtype),
                tf.zeros((batch_size, self.units), dtype=dtype)]
    
    def get_config(self):
        config = super(MinLSTMCell, self).get_config()
        config.update({'units': self.units})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="Custom", name="BiMinLSTM")
class BiMinLSTM(Model):
    """
    Bidirectional Minimal LSTM Model for Hate Speech Detection
    """
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=64, num_classes=3, 
                 max_len=100, dropout_rate=0.3, **kwargs):
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
            mask_zero=True,
            name='embedding'
        )
        
        # Spatial Dropout for embeddings
        self.spatial_dropout = layers.SpatialDropout1D(dropout_rate, name='spatial_dropout')
        
        # Bidirectional MinLSTM layer
        self.bi_minlstm = layers.Bidirectional(
            layers.RNN(MinLSTMCell(lstm_units), return_sequences=True),
            merge_mode='concat',
            name='bidirectional_minlstm'
        )
        
        # Dropout layer
        self.dropout1 = layers.Dropout(dropout_rate, name='dropout_1')

        # self.bi_minlstm2 = layers.Bidirectional(
        #     layers.RNN(MinLSTMCell(lstm_units), return_sequences=False),
        #     merge_mode='concat',
        #     name='bidirectional_minlstm_2'
        # )

        self.pooling = layers.GlobalMaxPool1D(name='global_max_pooling')

        # Dense layer
        self.dense1 = layers.Dense(32, activation='relu', name='dense_1')

        # Dropout layer
        self.dropout2 = layers.Dropout(dropout_rate, name='dropout_2')
        
        # Output layer
        self.output_layer = layers.Dense(num_classes, activation='softmax', name='output')
    
    def call(self, inputs, training=False):
        """Forward pass"""
        x = self.embedding(inputs)
        x = self.spatial_dropout(x, training=training)
        x = self.bi_minlstm(x, training=training)
        x = self.dropout1(x, training=training)
        # x = self.bi_minlstm2(x, training=training)
        x = self.pooling(x)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        output = self.output_layer(x)
        return output
    
    def get_config(self):
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
        return cls(**config)


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
    
    # Build the model by passing dummy data through it with batch size > 1
    print("\nBuilding model layers...")
    dummy_input = tf.zeros((2, max_len), dtype=tf.int32)  # Use batch_size=2 to properly build all layers
    _ = model(dummy_input, training=False)
    print("Model layers built successfully!")
    
    print("\nModel Architecture:")
    print("="*60)
    model.summary()
    print("="*60)
    
    return model


def get_custom_objects():
    """
    Return custom objects dictionary for model loading
    """
    return {
        'MinLSTMCell': MinLSTMCell,
        'BiMinLSTM': BiMinLSTM,
    }


def predict_text(model, text, preprocessor, class_names=['None', 'Racist', 'Sexist']):
    """
    Predict hate speech class for a given text
    
    Args:
        model: Trained model
        text: Input text string
        preprocessor: DataPreprocessor instance with fitted tokenizer
        class_names: List of class names
        
    Returns:
        Dictionary with prediction results
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Clean the text
    cleaned = preprocessor.clean_text(text)
    
    # Tokenize and pad
    sequence = preprocessor.tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence, 
        maxlen=preprocessor.max_len, 
        padding='post', 
        truncating='post'
    )
    
    # Predict
    probabilities = model.predict(padded, verbose=0)[0]
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    
    return {
        'text': text,
        'cleaned_text': cleaned,
        'predicted_class': class_names[predicted_class],
        'predicted_label': int(predicted_class),
        'confidence': float(confidence),
        'probabilities': {
            class_names[i]: float(probabilities[i]) 
            for i in range(len(class_names))
        }
    }


def test_model_predictions(model, preprocessor, class_names=['None', 'Racist', 'Sexist']):
    """
    Test the model with sample texts from each class
    
    Args:
        model: Trained model
        preprocessor: DataPreprocessor instance
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("TESTING MODEL WITH SAMPLE TEXTS")
    print("="*60 + "\n")
    
    # Test samples - 2 from each class
    test_samples = [
        # None (neutral) class
        "I love spending time with my family on weekends",
        "The weather is beautiful today perfect for a walk in the park",
        
        # Racist class
        "People from that country are all the same and inferior to us",
        "That ethnic group does not belong here and should go back",
        
        # Sexist class  
        "Women are not smart enough for leadership positions at all",
        "She only got that job because she is a woman not merit",
    ]
    
    expected_classes = [0, 0, 1, 1, 2, 2]  # Expected labels for each sample
    
    correct_predictions = 0
    
    print("Running predictions on test samples...\n")
    
    for i, (text, expected) in enumerate(zip(test_samples, expected_classes)):
        result = predict_text(model, text, preprocessor, class_names)
        
        is_correct = result['predicted_label'] == expected
        if is_correct:
            correct_predictions += 1
            status = "CORRECT ✓"
        else:
            status = "WRONG ✗"
        
        print(f"{'='*60}")
        print(f"Test {i+1}/6: {status}")
        print(f"{'='*60}")
        print(f"Original Text:")
        print(f'  "{text}"')
        print(f"\nCleaned Text:")
        print(f'  "{result["cleaned_text"]}"')
        print(f"\nPrediction:")
        print(f"  Expected Class:  {class_names[expected]}")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Confidence:      {result['confidence']:.2%}")
        print(f"\nClass Probabilities:")
        for cls_name, prob in result['probabilities'].items():
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"  {cls_name:8s} [{bar}] {prob:.4f}")
        print()
    
    print("="*60)
    accuracy = correct_predictions / len(test_samples)
    print(f"Overall Test Accuracy: {correct_predictions}/{len(test_samples)} ({accuracy:.1%})")
    print("="*60 + "\n")