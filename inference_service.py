import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


@dataclass
class Detection:
    class_name: str
    start_time: float  # seconds
    end_time: float
    confidence: float
    peak_confidence: float


@dataclass
class SegmentPrediction:
    segment_idx: int
    start_time: float
    end_time: float
    probabilities: Dict[str, float]  
    detected_classes: List[str]


class TemporalDetectionProcessor:
    
    def __init__(self, segment_duration: float = 1.0, hop_duration: float = 0.5,
                 classes: List[str] = None):
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.classes = classes or ["drone", "gunshot", "jetplane", "vehicle"]
        self.threshold = 0.45 
    
    def create_segments(self, audio: np.ndarray, sr: int) -> List[Tuple[np.ndarray, float, float]]:
        segment_samples = int(self.segment_duration * sr)
        hop_samples = int(self.hop_duration * sr)
        
        segments = []
        for start_idx in range(0, len(audio) - segment_samples + 1, hop_samples):
            end_idx = min(start_idx + segment_samples, len(audio))
            segment = audio[start_idx:end_idx]
            
            if len(segment) < segment_samples:
                segment = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')
            
            start_time = start_idx / sr
            end_time = end_idx / sr
            segments.append((segment, start_time, end_time))
        
        return segments
    
    def merge_detections(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []
        
        sorted_dets = sorted(detections, key=lambda x: (x.class_name, x.start_time))
        
        merged = []
        current = None
        gap_threshold = 0.3 
        
        for det in sorted_dets:
            if current is None:
                current = det
            elif det.class_name == current.class_name:
                gap = det.start_time - current.end_time
                if gap <= gap_threshold:
                    current.end_time = det.end_time
                    current.peak_confidence = max(current.peak_confidence, det.peak_confidence)
                    w1 = current.confidence
                    w2 = det.confidence
                    current.confidence = (w1 + w2) / 2
                else:
                    merged.append(current)
                    current = det
            else:
                merged.append(current)
                current = det
        
        if current is not None:
            merged.append(current)
        
        return merged


class DefenseSoundModel:
    
    def __init__(self, model_path: str, norm_stats_path: Optional[str] = None,
                 thresholds_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.mean = None
        self.std = None
        self.class_thresholds = {}
        self.classes = ["drone", "gunshot", "jetplane", "vehicle"]
        self.sr = 22050
        self.duration = 6
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        
        self._load_model()
        self._load_normalization_stats(norm_stats_path)
        self._load_thresholds(thresholds_path)
    
    @staticmethod
    def _weighted_binary_crossentropy(pos_weights):
        def loss(y_true, y_pred):
            epsilon = 1e-7
            y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
            loss = - (pos_weights * y_true * K.log(y_pred) +
                     (1 - y_true) * K.log(1 - y_pred))
            return K.mean(loss)
        return loss
    
    def _load_model(self):
        try:
            try:
                self.model = tf.keras.models.load_model(self.model_path)
            except (ValueError, TypeError) as e:
                if "loss" in str(e) or "custom" in str(e).lower():
                    import tensorflow.keras.backend as K
                    print("⚠ Loading model without compiled loss function...")
                    
                    self.model = tf.keras.models.load_model(
                        self.model_path, compile=False
                    )
                    
                    pos_weights = np.array([1.5, 1.0, 1.5, 1.0])
                    custom_loss = self._weighted_binary_crossentropy(pos_weights)
                    
                    self.model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=custom_loss,
                        metrics=[
                            tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall'),
                            tf.keras.metrics.AUC(name='auc', multi_label=True)
                        ]
                    )
                    print("✓ Model compiled with custom loss function")
                else:
                    raise
            
            print(f"✓ Model loaded from {self.model_path}")
            print(f"  Input shape: {self.model.input_shape}")
            print(f"  Output shape: {self.model.output_shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _load_normalization_stats(self, path: Optional[str]):
        if path and Path(path).exists():
            try:
                import pandas as pd
                df = pd.read_csv(path)
                self.mean = df["mean"].values[0]
                self.std = df["std"].values[0]
                print(f"✓ Normalization stats loaded: mean={self.mean:.4f}, std={self.std:.4f}")
            except Exception as e:
                print(f"⚠ Could not load normalization stats: {e}")
                self.mean = 0.0
                self.std = 1.0
        else:
            self.mean = 0.0
            self.std = 1.0
    
    def _load_thresholds(self, path: Optional[str]):
        if path and Path(path).exists():
            try:
                import pandas as pd
                df = pd.read_csv(path, index_col=0)
                for cls in self.classes:
                    self.class_thresholds[cls] = df.loc[cls, "threshold"]
                print(f"✓ Per-class thresholds loaded")
            except Exception as e:
                print(f"⚠ Could not load thresholds: {e}")
                self.class_thresholds = {cls: 0.45 for cls in self.classes}
        else:
            self.class_thresholds = {cls: 0.45 for cls in self.classes}
    
    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        target_samples = self.sr * self.duration
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        elif len(audio) > target_samples:
            audio = audio[:target_samples]
        
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        return mel_db.astype(np.float32)
    
    def _normalize_mel(self, mel: np.ndarray) -> np.ndarray:
        return (mel - self.mean) / (self.std + 1e-8)
    
    def predict_segment(self, audio: np.ndarray) -> Dict[str, float]:
        try:
            mel = self._extract_mel_spectrogram(audio)
            mel = self._normalize_mel(mel)
            mel = mel[..., np.newaxis]  
            mel = np.expand_dims(mel, axis=0) 
            
            probs = self.model.predict(mel, verbose=0)[0]
            
            return {cls: float(prob) for cls, prob in zip(self.classes, probs)}
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {cls: 0.0 for cls in self.classes}
    
    def predict_full_audio(self, audio: np.ndarray, use_temporal: bool = True
                          ) -> Tuple[List[Detection], List[SegmentPrediction]]:
        processor = TemporalDetectionProcessor(
            segment_duration=1.0, hop_duration=0.5, classes=self.classes
        )
        
        segments = processor.create_segments(audio, self.sr)
        all_detections = []
        segment_predictions = []
        
        for segment_audio, start_time, end_time in segments:
            probs = self.predict_segment(segment_audio)
            
            for cls, prob in probs.items():
                threshold = self.class_thresholds.get(cls, 0.45)
                if prob >= threshold:
                    det = Detection(
                        class_name=cls,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=prob,
                        peak_confidence=prob
                    )
                    all_detections.append(det)
            
            detected = [cls for cls, prob in probs.items() 
                       if prob >= self.class_thresholds.get(cls, 0.45)]
            seg_pred = SegmentPrediction(
                segment_idx=len(segment_predictions),
                start_time=start_time,
                end_time=end_time,
                probabilities=probs,
                detected_classes=detected
            )
            segment_predictions.append(seg_pred)
        
        merged = processor.merge_detections(all_detections)
        
        return merged, segment_predictions
    
    def predict_with_confidence(self, audio: np.ndarray) -> Dict[str, Dict]:
        detections, segments = self.predict_full_audio(audio, use_temporal=True)
        
        result = {}
        for cls in self.classes:
            class_dets = [d for d in detections if d.class_name == cls]
            result[cls] = {
                "detected": len(class_dets) > 0,
                "max_confidence": max([d.peak_confidence for d in class_dets], default=0.0),
                "detections": [
                    {
                        "start_time": d.start_time,
                        "end_time": d.end_time,
                        "confidence": d.confidence,
                        "duration": d.end_time - d.start_time
                    }
                    for d in class_dets
                ]
            }
        
        return result


def load_audio(file_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    try:
        audio, sr_loaded = librosa.load(file_path, sr=sr, mono=True)
        return audio.astype(np.float32), sr_loaded
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {file_path}: {e}")


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak * 0.95
    return audio
