import os
import pickle
import re
import numpy as np
import importlib
import joblib

HEURISTIC_TOXIC_PATTERNS = {
    r'\bstupid\b': 0.75,
    r'\bidiot\b': 0.8,
    r'\bmoron\b': 0.8,
    r'\bdumb\b': 0.7,
    r'\btrash\b': 0.6,
    r'\bworthless\b': 0.9,
    r'\bgarbage\b': 0.7,
    r'\bfuck\b': 0.95,
    r'\bshit\b': 0.85,
    r'\basshole\b': 0.95,
    r'kill yourself': 1.0,
    r'shut up': 0.55,
    r'go to hell': 0.85,
}

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
        
        # scikit-learn fallback (vectorizer + classifier)
        self.sk_model = None
        self.vectorizer = None
        
        # Load tokenizer
        try:
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"✓ Loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            self.tokenizer = None

        # If Keras model/tokenizer failed, try sklearn pipeline fallback
        if self.model is None or self.tokenizer is None:
            try:
                vec_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'tfidf_vectorizer.pkl')
                clf_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'logistic_regression.pkl')
                if os.path.exists(vec_path) and os.path.exists(clf_path):
                    self.vectorizer = joblib.load(vec_path)
                    self.sk_model = joblib.load(clf_path)
                    print(f"✓ Loaded sklearn fallback: {clf_path} + {vec_path}")
                else:
                    # try other classifiers if logistic not present
                    alt_clf = os.path.join(os.path.dirname(__file__), 'saved_models', 'random_forest.pkl')
                    if os.path.exists(vec_path) and os.path.exists(alt_clf):
                        self.vectorizer = joblib.load(vec_path)
                        self.sk_model = joblib.load(alt_clf)
                        print(f"✓ Loaded sklearn fallback: {alt_clf} + {vec_path}")
            except Exception as e:
                print(f"Error loading sklearn fallback models: {e}")
    
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

    def _heuristic_predict(self, text):
        lowered = (text or '').lower()
        matched = []
        score = 0.0

        for pattern, weight in HEURISTIC_TOXIC_PATTERNS.items():
            if re.search(pattern, lowered):
                matched.append(pattern)
                score = max(score, weight)

        is_toxic = score >= self.threshold
        if matched and not is_toxic:
            score = max(score, 0.6)
            is_toxic = score >= self.threshold

        return {
            'toxic': is_toxic,
            'score': float(score),
            'confidence': float(score if is_toxic else 1 - score),
            'reason': 'heuristic toxicity fallback' if matched else 'heuristic safe fallback'
        }
    
    def predict(self, text):
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        if not cleaned_text.strip():
            # If cleaning removes everything, try simple lowercase
            cleaned_text = text.lower().strip()
        
        # If sklearn fallback is available, use it
        if self.sk_model is not None and self.vectorizer is not None:
            try:
                feats = self.vectorizer.transform([cleaned_text])
                if hasattr(self.sk_model, 'predict_proba'):
                    proba = self.sk_model.predict_proba(feats)[0]
                    # assume positive class is index 1
                    score = float(proba[1]) if len(proba) > 1 else float(proba[0])
                else:
                    pred = int(self.sk_model.predict(feats)[0])
                    score = 1.0 if pred == 1 else 0.0
                is_toxic = score > self.threshold
                confidence = score if is_toxic else 1 - score
                return {
                    'toxic': is_toxic,
                    'score': score,
                    'confidence': confidence,
                    'reason': 'sklearn classifier fallback'
                }
            except Exception as e:
                print(f"Error during sklearn prediction fallback: {e}")

        # Final fallback: rule-based moderation so obvious abuse still gets blocked
        heuristic = self._heuristic_predict(cleaned_text)
        if heuristic['toxic']:
            return heuristic

        if self.model is None or self.tokenizer is None:
            return heuristic

        # Tokenize and pad for Keras model
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
        # If sklearn fallback is available, prefer it when Keras artifacts are unavailable.
        if self.sk_model is not None and self.vectorizer is not None:
            cleaned_texts = [self._clean_text(t) for t in texts]
            try:
                feats = self.vectorizer.transform(cleaned_texts)
                if hasattr(self.sk_model, 'predict_proba'):
                    probas = self.sk_model.predict_proba(feats)
                    scores = [float(p[1]) if len(p) > 1 else float(p[0]) for p in probas]
                else:
                    preds = self.sk_model.predict(feats)
                    scores = [1.0 if int(p) == 1 else 0.0 for p in preds]
                results = []
                for score in scores:
                    is_toxic = score > self.threshold
                    results.append({
                        'toxic': is_toxic,
                        'score': float(score),
                        'confidence': float(score if is_toxic else 1 - score),
                        'reason': 'sklearn classifier fallback'
                    })
                return results
            except Exception as e:
                print(f"Error during sklearn batch prediction fallback: {e}")

        if self.model is None or self.tokenizer is None:
            return [self._heuristic_predict(text) for text in texts]
        
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

