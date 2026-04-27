import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import tempfile
import traceback
import numpy as np
import librosa
from datetime import datetime

from inference_service import DefenseSoundModel, load_audio, normalize_audio
from visualization import (
    RadarVisualizer, StatisticsCalculator, LogFormatter, MilitaryThemeCSS
)

GRID_CSS = """
<style>
body {
    background-color: #0b0f14;
    background-image:
        linear-gradient(rgba(0, 255, 200, 0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 200, 0.06) 1px, transparent 1px);
    background-size: 40px 40px;
}

.stApp {
    background-color: #0b0f14;
    background-image:
        linear-gradient(rgba(0, 255, 200, 0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 200, 0.06) 1px, transparent 1px);
    background-size: 20px 20px;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    box-shadow: inset 0 0 120px rgba(0, 255, 200, 0);
    pointer-events: none;
}
</style>
"""

st.markdown(GRID_CSS, unsafe_allow_html=True)

st.set_page_config(
    page_title="Sound Threat Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(MilitaryThemeCSS.get_custom_css(), unsafe_allow_html=True)

@st.cache_resource
def load_model_cached():
    try:
        model_path = r'S:\sem8\Sound-Threat-Detection-System\model\multilabel_model.keras'
        norm_stats_path = r'S:\sem8\Sound-Threat-Detection-System\model\generated_multilabel\normalization_stats.csv'
        thresholds_path = r'S:\sem8\Sound-Threat-Detection-System\model\generated_multilabel\optimal_thresholds.csv'
       
        if not Path(model_path).exists():
            st.error(f"❌ Model not found at: {model_path}")
            return None
        
        model = DefenseSoundModel(
            model_path=model_path,
            norm_stats_path=norm_stats_path if Path(norm_stats_path).exists() else None,
            thresholds_path=thresholds_path if Path(thresholds_path).exists() else None
        )
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return None


def initialize_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = load_model_cached()
    
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    
    if 'audio_sr' not in st.session_state:
        st.session_state.audio_sr = 22050
    
    if 'detections' not in st.session_state:
        st.session_state.detections = None
    
    if 'segments' not in st.session_state:
        st.session_state.segments = None
    
    if 'stats' not in st.session_state:
        st.session_state.stats = None
    
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False
    
    if 'last_uploaded_name' not in st.session_state:
        st.session_state.last_uploaded_name = None


def save_results():
    if not st.session_state.processing_done or not st.session_state.detections:
        return False, "No results to save"
    
    try:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_folder = results_dir / f"result_{timestamp}"
        result_folder.mkdir(exist_ok=True)
        
        detections = st.session_state.detections
        stats = st.session_state.stats
        audio_duration = len(st.session_state.audio_data) / st.session_state.audio_sr
        
        log_text = LogFormatter.format_detection_log(detections)
        
        timeline_img = RadarVisualizer.create_timeline_visualization(
            detections, audio_duration, figsize=(14, 6)
        )
        image_file = result_folder / "timeline.png"
        timeline_img.save(image_file)
        
        summary_file = result_folder / "detection_log.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("DEFENSE SOUND IDENTIFICATION - ANALYSIS RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Audio File: {st.session_state.last_uploaded_name}\n")
            f.write(f"Audio Duration: {audio_duration:.2f}s\n")
            f.write(f"Total Detections: {stats['total_detections']}\n")
            f.write(f"Classes Found: {', '.join(stats['classes_detected']) or 'None'}\n")
            f.write(f"Detection Density: {stats['detection_density']:.1%}\n\n")
            f.write(log_text)
        
        return True, f"Results saved to: {result_folder}"
    
    except Exception as e:
        return False, f"Error saving results: {e}"


def display_header():
    
    st.markdown(
            """
            <div style='text-align: center; margin-bottom: 20px;'>
                <h1 style='color: #02a85d; text-shadow: 0 0 20px rgba(0,255,0,0.4); 
                          font-family: Courier New; letter-spacing: 3px;'>
                <svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-shield h-8 w-8 text-primary"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"></path></svg>
                     SOUND THREAT DETECTION SYSTEM
                </h1>
                
        </div>
        """,
        unsafe_allow_html=True
    )

def display_radar():
    
    radar_html = """
    <div style="display: flex; justify-content: center; align-items: center; margin: 20px 0;">
        <div style="position: relative; width: 300px; height: 300px;">
            <div style="position: absolute; inset: 0; border-radius: 50%; background: radial-gradient(circle at 30% 30%, #0a0f1a, #05080f); box-shadow: 0 0 20px rgba(0,255,200,0.2); overflow: hidden;">
                
                <!-- Rings -->
                <div style="position: absolute; inset: 0; border-radius: 50%; border: 1px solid rgba(0,255,200,0.2);"></div>
                <div style="position: absolute; top: 25%; left: 25%; width: 50%; height: 50%; border-radius: 50%; border: 1px solid rgba(0,255,200,0.15);"></div>
                <div style="position: absolute; top: 12.5%; left: 12.5%; width: 75%; height: 75%; border-radius: 50%; border: 1px solid rgba(0,255,200,0.15);"></div>
                
                <!-- Crosshairs -->
                <div style="position: absolute; top: 50%; left: 0; width: 100%; height: 1px; background: rgba(0,255,200,0.15); transform: translateY(-50%);"></div>
                <div style="position: absolute; top: 0; left: 50%; width: 1px; height: 100%; background: rgba(0,255,200,0.15); transform: translateX(-50%);"></div>
                
                <!-- Rotating sweep -->
                <div style="position: absolute; top: 50%; left: 50%; width: 60%; height: 1%; background: linear-gradient(90deg, rgba(0,255,200,0.3), rgba(0,255,200,0)); border-radius: 0 100% 100% 0; transform-origin: 0% 50%; animation: spin 2s linear infinite;"></div>
                
                <!-- Center dot -->
                <div style="position: absolute; top: 50%; left: 50%; width: 8px; height: 8px; background: #00ffc8; border-radius: 50%; transform: translate(-50%, -50%); animation: pulse 1s infinite;"></div>
            </div>
        </div>
    </div>
    
    <style>
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 0.5; box-shadow: 0 0 5px #00ffc8; }
            50% { opacity: 1; box-shadow: 0 0 15px #00ffc8; }
        }
    </style>
    """
    
    components.html(radar_html, height=350)
 

def display_audio_input_section():
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown(
            """
            <div class='status-box status-active'>
                <h3>🎙️ AUDIO CAPTURE</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            audio_source = st.radio(
                "Select input source:",
                ["📁 Upload File", "🎙️ Record Audio"],
                label_visibility="collapsed"
            )
        
        with col2:
            st.write("")  
        
        audio_data = None
        audio_sr = 22050
        
        if audio_source == "📁 Upload File":
            uploaded_file = st.file_uploader(
                "Choose audio file (.wav, .mp3, .ogg)",
                type=["wav", "mp3", "ogg"],
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                try:
                    if st.session_state.last_uploaded_name != uploaded_file.name:
                        st.session_state.processing_done = False
                        st.session_state.detections = None
                        st.session_state.segments = None
                        st.session_state.stats = None
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_path = temp_file.name
                    
                    st.session_state.last_uploaded_name = uploaded_file.name
                    audio_data, audio_sr = load_audio(temp_path, sr=22050)
                    rms = np.mean(librosa.feature.rms(y=audio_data))
                    zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))

                    st.success(f"✓ Loaded: {uploaded_file.name}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duration", f"{len(audio_data)/audio_sr:.2f}s")
                    with col2:
                        st.metric("Loudness", f"{rms:.4f}")
                    with col3:
                        st.metric("Noise Activity", f"{zcr:.4f}")
                    
                    st.audio(uploaded_file, format='audio/wav')
                    
                    st.session_state.audio_data = audio_data
                    st.session_state.audio_sr = audio_sr
                    
                except Exception as e:
                    st.error(f"❌ Error loading audio: {e}")
                    st.write(traceback.format_exc())
        
        else:  
            audio_data_recorded = st.audio_input(
                "Click to record audio (max 60 seconds):",
                label_visibility="collapsed"
            )
            
            if audio_data_recorded is not None:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        temp_file.write(audio_data_recorded.getbuffer())
                        temp_path = temp_file.name
                    
                    audio_data, audio_sr = load_audio(temp_path, sr=22050)
                    rms = np.mean(librosa.feature.rms(y=audio_data))
                    zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
                    st.success("✓ Recording captured")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duration", f"{len(audio_data)/audio_sr:.2f}s")
                    with col2:
                        st.metric("Loudness", f"{rms:.4f}")
                    with col3:
                        st.metric("Noise Activity", f"{zcr:.4f}")
                    
                    st.session_state.audio_data = audio_data
                    st.session_state.audio_sr = audio_sr

                    if st.session_state.last_uploaded_name != "recorded_audio":
                        st.session_state.processing_done = False
                        st.session_state.detections = None
                        st.session_state.segments = None
                        st.session_state.stats = None
                        st.session_state.last_uploaded_name = "recorded_audio"
                    
                except Exception as e:
                    st.error(f"❌ Error processing recording: {e}")
                    audio_data = None
        
        return audio_data is not None
    
    with col_right:
        return None
        
display_radar()

def process_audio():
    if st.session_state.audio_data is None:
        st.warning("⚠️ No audio loaded yet")
        return False
    
    if st.session_state.model is None:
        st.error("❌ Model not loaded")
        return False
    
    try:
        with st.spinner("🔄 SCANNING AUDIO..."):
            normalized_audio = normalize_audio(st.session_state.audio_data)
            
            detections, segments = st.session_state.model.predict_full_audio(
                normalized_audio, use_temporal=True
            )
            
            detections = [d for d in detections if d.peak_confidence > 0.6]

            stats = StatisticsCalculator.calculate_stats(
                detections, len(normalized_audio) / st.session_state.audio_sr
            )
            
            st.session_state.detections = detections
            st.session_state.segments = segments
            st.session_state.stats = stats
            st.session_state.processing_done = True
        
        st.success("✓ Analysis complete")
        return True
    
    except Exception as e:
        st.error(f"❌ Processing failed: {e}")
        st.write(traceback.format_exc())
        return False


def display_results():
    if not st.session_state.processing_done:
        return
    
    detections = st.session_state.detections
    stats = st.session_state.stats
    audio_duration = len(st.session_state.audio_data) / st.session_state.audio_sr
    
    st.markdown(
        """
        <div class='status-box status-active' style='margin: 20px 0;'>
            <h3>📊 DETECTION SUMMARY</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Detections", stats['total_detections'],
                 delta=None, delta_color="normal")
    with col2:
        st.metric("Classes Found", len(stats['classes_detected']),
                 delta=None, delta_color="normal")
    with col3:
        st.metric("Audio Duration", f"{audio_duration:.2f}s",
                 delta=None, delta_color="normal")
    
    tab1, tab2 = st.tabs(["📈 Timeline", "📋 Detailed Log"])
    
    with tab1:
        st.markdown("**Detection Timeline**")
        st.write("Shows when each class was detected throughout the audio")
        
        try:
            timeline_img = RadarVisualizer.create_timeline_visualization(
                detections, audio_duration, figsize=(14, 6)
            )
            st.image(timeline_img)
        except Exception as e:
            st.error(f"Could not render timeline: {e}")
    
    with tab2:
        st.markdown("**Detection Details**")
        
        if detections:
            det_data = []
            for i, det in enumerate(sorted(detections, key=lambda x: x.start_time), 1):
                det_data.append({
                    "#": i,
                    "Class": det.class_name.upper(),
                    "Start": f"{det.start_time:.2f}s",
                    "End": f"{det.end_time:.2f}s",
                    "Duration": f"{det.end_time - det.start_time:.2f}s",
                    "Confidence": f"{det.peak_confidence:.2%}"
                })
            
            import pandas as pd
            df = pd.DataFrame(det_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No detections found in audio")
    
    st.markdown(
        """
        <div class='status-box status-active' style='margin: 20px 0;'>
            <h3>📊 CLASSIFICATION STATISTICS</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Per-Class Overview**")

        emoji_map = {
            'drone': '𖥂',
            'gunshot': '𖦏',
            'jetplane': '🛩️',
            'vehicle': '🚚'
        }

        c1, c2, c3, c4 = st.columns(4)
        cols = [c1, c2, c3, c4]

        for i, cls in enumerate(['drone', 'gunshot', 'jetplane', 'vehicle']):
            info = stats['per_class'][cls]
            count = info['count'] if info['detected'] else 0

            with cols[i]:
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(10,14,39,0.8);
                        border: 1px solid rgba(0,255,200,0.2);
                        border-radius: 12px;
                        padding: 20px;
                        text-align: center;
                        box-shadow: 0 0 15px rgba(0,255,200,0.1);
                    ">
                        <div style="font-size: 28px;">{emoji_map[cls]}</div>
                        <div style="
                            font-size: 32px;
                            font-weight: bold;
                            color: #00cc88;
                            margin: 10px 0;
                            font-family: Courier New;
                        ">
                            {count}
                        </div>
                        <div style="
                            color: #00ffff;
                            font-size: 13px;
                            letter-spacing: 2px;
                            font-family: Courier New;
                        ">
                            {cls.upper()}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    with col2:
        st.markdown("**Detection Log**")
        log_text = LogFormatter.format_detection_log(detections)
        st.code(log_text, language="text")


def main():
    initialize_session_state()
    
    display_header()
    
    st.markdown("---")
    
    audio_loaded = display_audio_input_section()
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔍 RUN ANALYSIS", use_container_width=True):
            if audio_loaded:
                process_audio()
            else:
                st.warning("⚠️ Load audio first")      
    
    with col2:
        if st.button("🔄 RESET", use_container_width=True):
            st.session_state.audio_data = None
            st.session_state.detections = None
            st.session_state.segments = None
            st.session_state.stats = None
            st.session_state.processing_done = False
            st.rerun()  
    
    with col3:
        if st.button("💾 SAVE RESULTS", use_container_width=True):
            if st.session_state.processing_done and st.session_state.detections:
                success, message = save_results()
                if success:
                    st.success(f"📦 {message}")
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("⚠️ Run analysis first")
    
    st.markdown("---")
    
    if st.session_state.processing_done:
        display_results()
    else:
        st.info(" Load audio and click 'RUN ANALYSIS' to begin detection")
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-family: Courier New; font-size: 11px; color: #02a85d;'>
            <p>SOUND THREAT DETECTION SYSTEM</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()