from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "model"

MODEL_PATH = MODELS_DIR / "multilabel_model.keras"
NORMALIZATION_STATS_PATH = MODELS_DIR / "generated_multilabel" / "normalization_stats.csv"
OPTIMAL_THRESHOLDS_PATH = MODELS_DIR / "generated_multilabel" / "optimal_thresholds.csv"

ORIGINAL_DATASET_PATH = MODELS_DIR / "Dataset"
GENERATED_DATASET_PATH = MODELS_DIR / "generated_multilabel"
NOISE_DIR_PATH = MODELS_DIR / "noise"

SAMPLE_RATE = 22050
AUDIO_DURATION = 6  
NUM_SAMPLES = SAMPLE_RATE * AUDIO_DURATION
NUM_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

CLASSES = ["drone", "gunshot", "jetplane", "vehicle"]
NUM_CLASSES = len(CLASSES)

DEFAULT_THRESHOLD = 0.45
SEGMENT_DURATION = 1.0  
SEGMENT_HOP = 0.5  
TEMPORAL_GAP_THRESHOLD = 0.3  

BATCH_SIZE = 32
INFERENCE_MODE = "temporal" 

STREAMLIT_CONFIG = {
    "page_title": "Sound Threat Detection System",
    "page_icon": "🛡️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

COLORS = {
    "primary": "#02a85d",      
    "dark": "#0d1710",         
    "danger": "#ff0000",       
    "warning": "#ffff00",      
    "info": "#00ffff",         
    "drone": "#00ff00",
    "gunshot": "#ff0000",
    "jetplane": "#00ffff",
    "vehicle": "#ffff00"
}

CLASS_POSITIONS = {
    "drone": 3,
    "gunshot": 2,
    "jetplane": 1,
    "vehicle": 0
}

def validate_paths():
    errors = []
    
    if not MODEL_PATH.exists():
        errors.append(f"Model not found at: {MODEL_PATH}")
    
    if not ORIGINAL_DATASET_PATH.exists():
        errors.append(f"Dataset not found at: {ORIGINAL_DATASET_PATH}")
    
    return errors


def validate_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(MODEL_PATH))
        return True, "Model loaded successfully"
    except Exception as e:
        return False, f"Model loading failed: {e}"


def get_audio_info():
    return {
        "sample_rate": SAMPLE_RATE,
        "duration": AUDIO_DURATION,
        "num_samples": NUM_SAMPLES,
        "n_mels": NUM_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH
    }


if __name__ == "__main__":
    print("🔍 Configuration Validator")
    print("=" * 50)
    
    errors = validate_paths()
    if errors:
        print("\n❌ Path Errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("✓ All paths valid")
    
    success, msg = validate_model()
    if success:
        print(f"✓ {msg}")
    else:
        print(f"❌ {msg}")
    
    print("\n📊 Audio Configuration:")
    info = get_audio_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
