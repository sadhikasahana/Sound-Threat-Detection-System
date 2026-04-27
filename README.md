# Sound Threat Detection System

**Multi-label acoustic threat detection for military environments**  
Detects drones, gunshots, jetplanes, and ground vehicles from audio files or live microphone input.

---

## Overview

This project implements a real‑time multi‑label sound classification system for four military threat categories:

| Class       | Description                       |
| ----------- | --------------------------------- |
| 𖥂 Drone     | Electric motor / propeller sounds |
| 𖦏 Gunshot   | Impulsive firearm discharge       |
| 🛩️ Jetplane | Jet engine and aircraft noise     |
| 🚚 Vehicle  | Ground vehicle engine sounds      |

The system uses a **CNN‑BiGRU** deep learning model trained on synthetically generated multi‑label mixtures. A temporal detection processor (sliding window + merging) provides event‑level localisation. The front end is a military‑themed **Streamlit** web application with radar animation, timeline visualisation, and detailed detection logs.

---

## Features

- 🎯 **Multi‑label classification** – detects multiple overlapping sounds in the same audio segment.
- ⏱️ **Temporal localisation** – sliding window (1 s segments, 0.5 s hop) with intelligent merging.
- 📊 **Comprehensive visualisation** – timeline bars, per‑class metrics, tactical log format.
- 🎙️ **Two input modes** – file upload (WAV/MP3/OGG) or live microphone recording.
- ⚡ **Real‑time capable** – GPU inference: 0.0052 RTF; CPU: 0.0148 RTF.
- 🔧 **Optimisable thresholds** – per‑class thresholds (e.g., gunshot 0.40, jetplane 0.55) improve F1 by 3.6%.

---

## Demo

_The interface includes a rotating radar, audio controls, detection timeline, and detailed event logs._

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster inference

### Clone the repository

```bash
git clone https://github.com/sadhikasahana/sound-threat-detection-system.git
cd sound-threat-detection-system
```

```bash
pip install -r requirements.txt
```

## Usage

### 1.Run the web application

```bash
streamlit run app.py
```

The app will open in your default browser.

### 2.Load Audio

- Upload a file (WAV, MP3, OGG) or record from your microphone (max 60 s).

- Audio info displayed: duration, loudness, noise activity

### 3. Run analysis

- Click 🔍 RUN ANALYSIS. The system will:

- Split the audio into overlapping 1‑second segments.

- Extract mel‑spectrograms (128×259) for each segment.

- Run the CNN‑BiGRU model.

- Merge segment detections (gap tolerance 0.3 s).

- Display results: summary metrics, timeline, detection table, and log.

### 4. Interpret results

- Timeline – coloured bars show when each class was detected. Hover (if supported) or refer to the table for exact times.

- Detection log – formatted like a tactical report, listing start/end times and confidence.

- Per‑class statistics – counts, total time, average/max confidence

### 5. Export Results

- Click "💾 SAVE RESULTS" to export findings

## Data Flow

```
Audio Input
    ↓
[File Upload / Recording]
    ↓
[Audio Normalization]
    ↓
[Segment Creation (Overlapping)]
    ↓
[Mel Spectrogram Extraction]
    ↓
[Model Inference on Each Segment]
    ↓
[Temporal Detection Merging]
    ↓
[Results Aggregation]
    ↓
[Visualization & Statistics]
```
