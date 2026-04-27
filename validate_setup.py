import sys
from pathlib import Path
import traceback

def check_python_version():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, you have {version.major}.{version.minor}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_imports():
    required_packages = {
        'tensorflow': 'TensorFlow',
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'streamlit': 'Streamlit',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            missing.append(name)
            print(f"❌ {name} - NOT INSTALLED")
    
    return len(missing) == 0, missing


def check_model():
    try:
        from config import MODEL_PATH
        
        if not MODEL_PATH.exists():
            print(f"❌ Model file not found: {MODEL_PATH}")
            return False
        
        print(f"✓ Model file found: {MODEL_PATH.name}")
        
        try:
            import tensorflow as tf
            import tensorflow.keras.backend as K
            import numpy as np
            
            try:
                model = tf.keras.models.load_model(str(MODEL_PATH))
            except (ValueError, TypeError) as e:
                if "loss" in str(e) or "custom" in str(e).lower():
                    print("⚠ Custom loss function detected, loading without compile...")
                    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
                    
                    def weighted_loss(pos_weights):
                        def loss(y_true, y_pred):
                            epsilon = 1e-7
                            y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
                            loss = - (pos_weights * y_true * K.log(y_pred) +
                                     (1 - y_true) * K.log(1 - y_pred))
                            return K.mean(loss)
                        return loss
                    
                    pos_weights = np.array([1.5, 1.0, 1.5, 1.0])
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=weighted_loss(pos_weights),
                        metrics=[tf.keras.metrics.BinaryAccuracy()]
                    )
                else:
                    raise
            
            print(f"✓ Model loaded successfully")
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)[:100]}...")
            return False
    except Exception as e:
        print(f"❌ Error with model: {e}")
        traceback.print_exc()
        return False


def check_audio_libs():
    try:
        import librosa
        import soundfile
        import numpy as np
        
        y = np.random.randn(22050) 
        mel = librosa.feature.melspectrogram(y=y, sr=22050)
        print(f"✓ Audio processing works")
        return True
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
        return False


def check_config():
    try:
        from config import validate_paths, validate_model, CLASSES, SAMPLE_RATE
        
        errors = validate_paths()
        if errors:
            print("⚠️ Configuration warnings:")
            for e in errors:
                print(f"  - {e}")
        else:
            print("✓ Configuration valid")
        
        print(f"✓ Classes: {CLASSES}")
        print(f"✓ Sample rate: {SAMPLE_RATE} Hz")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("DEFENSE SOUND IDENTIFICATION - STARTUP CHECK")
    print("=" * 60 + "\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Package Dependencies", lambda: check_imports()[0]),
        ("Model File", check_model),
        ("Audio Libraries", check_audio_libs),
        ("Configuration", check_config),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n[*] {name}")
        print("-" * 40)
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"[X] Check failed with exception: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "[+] PASS" if result else "[X] FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n[+] All systems ready! You can now run:")
        print("  streamlit run app.py")
        return 0
    else:
        print("\n[X] Some checks failed. Please fix issues above.")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
