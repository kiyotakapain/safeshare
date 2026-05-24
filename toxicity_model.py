import os
import pickle
import re
import numpy as np
import importlib

try:
    tf = importlib.import_module('tensorflow')
    keras_models = importlib.import_module('tensorflow.keras.models')
    load_model = getattr(keras_models, 'load_model')
    seq_module = importlib.import_module('tensorflow.keras.preprocessing.sequence')
    pad_sequences = getattr(seq_module, 'pad_sequences')

    try:
        tf.get_logger().setLevel('ERROR')
    except Exception:
        pass
except Exception:
    tf = None
    load_model = None
    pad_sequences = None

class ToxicityDetector:
    def __init__(self, model_path='./saved_models/cnn_model.keras', 
                 tokenizer_path='./saved_models/tokenizer.pkl',
                 threshold=0.552):
        self.threshold = threshold
        self.max_sequence_length = 150 
        
        custom_objects = {
            'focal_loss_fixed': self._focal_loss,
            'focal_loss': self._focal_loss
        }
        
        try:
            self.model = load_model(model_path, custom_objects=custom_objects, compile=False)
            print(f"✓ Loaded CNN model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure the model file exists in 'saved_models/' directory")
            self.model = None
        
        # Load tokenizer
        try:
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"✓ Loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            self.tokenizer = None
    
    def _focal_loss(self, y_true, y_pred, gamma=2.0, alpha=0.75):
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        loss = -alpha_t * tf.pow(1 - pt, gamma) * tf.math.log(pt + 1e-8)
        return tf.reduce_mean(loss)
    
    def _clean_text(self, text):
        if not text:
            return ""
        
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions/hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Keep only letters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove digits
        text = re.sub(r'\d+', '', text)
        
        # Lowercase and normalize spaces
        text = text.lower()
        text = ' '.join(text.split())
        
        return text
    
    def predict(self, text):
        if self.model is None or self.tokenizer is None:
            return {
                'toxic': False,
                'score': 0.0,
                'confidence': 0.0,
                'reason': 'Model not loaded properly'
            }
        
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        if not cleaned_text.strip():
            # If cleaning removes everything, try simple lowercase
            cleaned_text = text.lower().strip()
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')
        
        # Predict
        score = float(self.model.predict(padded, verbose=0)[0][0])
        
        # Determine if toxic based on threshold
        is_toxic = score > self.threshold
        confidence = score if is_toxic else 1 - score
        
        return {
            'toxic': is_toxic,
            'score': score,
            'confidence': confidence,
            'reason': f'CNN model prediction (threshold: {self.threshold})'
        }
    
    def predict_batch(self, texts):
        if self.model is None or self.tokenizer is None:
            return [{'toxic': False, 'score': 0.0, 'confidence': 0.0} for _ in texts]
        
        cleaned_texts = [self._clean_text(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        scores = self.model.predict(padded, verbose=0).flatten()
        
        results = []
        for score in scores:
            is_toxic = score > self.threshold
            results.append({
                'toxic': is_toxic,
                'score': float(score),
                'confidence': float(score if is_toxic else 1 - score),
                'reason': 'CNN model prediction'
            })
        
        return results

_detector = None

def get_detector():
    global _detector
    if _detector is None:
        # Check if model files exist
        model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'cnn_model.keras')
        tokenizer_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'tokenizer.pkl')
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            _detector = ToxicityDetector(model_path, tokenizer_path)
        else:
            print(f"Warning: Model files not found at {model_path}")
            print("Please ensure your trained model is in the 'saved_models' folder")
            _detector = ToxicityDetector()  # Will return None model
    return _detector

