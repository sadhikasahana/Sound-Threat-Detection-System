import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import io
from PIL import Image
from inference_service import Detection


class RadarVisualizer:
    
    @staticmethod
    def create_timeline_visualization(detections: List[Detection],
                                     audio_duration: float,
                                     figsize: Tuple[int, int] = (14, 6)) -> Image.Image:
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='#02a85d')
        
        class_colors = {
            'drone': '#00ff00',
            'gunshot': '#ff0000',
            'jetplane': '#00ffff',
            'vehicle': '#ffff00'
        }
        
        class_positions = {
            'drone': 3,
            'gunshot': 2,
            'jetplane': 1,
            'vehicle': 0
        }
        
        detections_by_class = {}
        for det in detections:
            if det.class_name not in detections_by_class:
                detections_by_class[det.class_name] = []
            detections_by_class[det.class_name].append(det)
        
        for cls, class_dets in detections_by_class.items():
            y_pos = class_positions[cls]
            color = class_colors[cls]
            
            for det in class_dets:
                duration = det.end_time - det.start_time
                rect = patches.Rectangle(
                    (det.start_time, y_pos - 0.3),
                    duration, 0.6,
                    linewidth=2, edgecolor=color, facecolor=color, alpha=0.7
                )
                ax.add_patch(rect)
                
                confidence_text = f"{det.peak_confidence:.2f}"
                ax.text(det.start_time + duration/2, y_pos, confidence_text,
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       color='#000000')
        
        ax.set_xlim(0, audio_duration)
        ax.set_ylim(-0.5, 3.5)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['Vehicle', 'Jetplane', 'Gunshot', 'Drone'],
                          color='#02a85d', fontweight='bold')
        ax.set_xlabel('Time (seconds)', color='#02a85d', fontweight='bold', fontsize=11)
        ax.set_title('DETECTION TIMELINE', color='#02a85d', fontweight='bold', fontsize=14)
        
        ax.set_facecolor("#0d1710")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#02a85d')
        ax.spines['bottom'].set_color('#02a85d')
        ax.tick_params(colors='#02a85d', labelsize=10)
        ax.grid(True, alpha=0.2, color='#02a85d', linestyle='--', axis='x')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#0d1710', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        return img


class StatisticsCalculator:
    
    @staticmethod
    def calculate_stats(detections: List[Detection], audio_duration: float) -> Dict:
        
        stats = {
            'total_detections': len(detections),
            'total_duration': audio_duration,
            'classes_detected': set(),
            'detection_density': 0.0,
            'per_class': {}
        }
        
        class_times = {
            'drone': 0.0,
            'gunshot': 0.0,
            'jetplane': 0.0,
            'vehicle': 0.0
        }
        
        class_counts = {
            'drone': 0,
            'gunshot': 0,
            'jetplane': 0,
            'vehicle': 0
        }
        
        class_confidences = {
            'drone': [],
            'gunshot': [],
            'jetplane': [],
            'vehicle': []
        }
        
        for det in detections:
            stats['classes_detected'].add(det.class_name)
            duration = det.end_time - det.start_time
            class_times[det.class_name] += duration
            class_counts[det.class_name] += 1
            class_confidences[det.class_name].append(det.peak_confidence)
        
        stats['classes_detected'] = sorted(list(stats['classes_detected']))
        
        for cls in ['drone', 'gunshot', 'jetplane', 'vehicle']:
            confidences = class_confidences[cls]
            stats['per_class'][cls] = {
                'count': class_counts[cls],
                'total_time': class_times[cls],
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'max_confidence': max(confidences) if confidences else 0.0,
                'detected': class_counts[cls] > 0
            }
        
        total_det_time = sum(class_times.values())
        stats['detection_density'] = total_det_time / audio_duration if audio_duration > 0 else 0.0
        
        return stats


class LogFormatter:
    
    @staticmethod
    def format_detection_log(detections: List[Detection]) -> str:
        
        if not detections:
            return "▶ NO DETECTIONS\n"
        
        log_lines = ["=" * 70]
        log_lines.append("TACTICAL DETECTION LOG".center(70))
        log_lines.append("=" * 70)
        
        sorted_dets = sorted(detections, key=lambda x: x.start_time)
        
        for i, det in enumerate(sorted_dets, 1):
            duration = det.end_time - det.start_time
            log_line = (
                f"[{i:02d}] {det.class_name.upper():10s} | "
                f"{det.start_time:6.2f}s - {det.end_time:6.2f}s ({duration:.2f}s) | "
                f"CONF: {det.peak_confidence:.2%}"
            )
            log_lines.append(log_line)
        
        log_lines.append("=" * 70)
        return "\n".join(log_lines)
    
    @staticmethod
    def format_stats(stats: Dict) -> str:
        text = []
        text.append("=" * 50)
        text.append("MISSION STATISTICS".center(50))
        text.append("=" * 50)
        text.append(f"Total Audio Duration: {stats['total_duration']:.2f}s")
        text.append(f"Total Detections: {stats['total_detections']}")
        text.append(f"Detection Density: {stats['detection_density']:.1%}")
        text.append(f"Classes Detected: {', '.join(stats['classes_detected']) or 'NONE'}")
        text.append("-" * 50)
        
        for cls in ['drone', 'gunshot', 'jetplane', 'vehicle']:
            info = stats['per_class'][cls]
            if info['detected']:
                text.append(f"\n▸ {cls.upper()}")
                text.append(f"  Occurrences: {info['count']}")
                text.append(f"  Total Time: {info['total_time']:.2f}s")
                text.append(f"  Avg Confidence: {info['avg_confidence']:.2%}")
                text.append(f"  Max Confidence: {info['max_confidence']:.2%}")
        
        text.append("\n" + "=" * 50)
        return "\n".join(text)


class MilitaryThemeCSS:
    
    @staticmethod
    def get_custom_css() -> str:
        return """
        <style>
            :root {
                --color-primary: #02a85d;
                --color-dark: #0d1710;
                --color-danger: #ff0000;
                --color-warning: #ffff00;
                --color-info: #00ffff;
            }
            
            .stApp {
                background-color: var(--color-dark);
            }
            
            .metric-card {
                background: linear-gradient(135deg, #02a85d 0%, #02a85d 100%);
                border: 2px solid var(--color-primary);
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
                color: var(--color-primary);
                font-family: 'Courier New', monospace;
            }
            
            .status-box {
                background: linear-gradient(135deg, #0d1710 0%, #1a1f3a 100%);
                border-left: 4px solid var(--color-primary);
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
                color: var(--color-primary);
                font-family: 'Courier New', monospace;
            }
            
            .status-active {
                border-left-color: var(--color-primary);
                box-shadow: inset 0 0 10px rgba(0, 255, 0, 0.1);
            }
            
            .status-danger {
                border-left-color: var(--color-danger);
                box-shadow: inset 0 0 10px rgba(255, 0, 0, 0.1);
                color: var(--color-danger);
            }
            
            .status-warning {
                border-left-color: var(--color-warning);
                color: var(--color-warning);
            }
            
            .status-info {
                border-left-color: var(--color-info);
                color: var(--color-info);
            }
            
            @keyframes scan {
                0% { transform: rotateZ(0deg); }
                100% { transform: rotateZ(360deg); }
            }
            
            .scanning {
                animation: scan 2s linear infinite;
            }
            
            .alert-success {
                color: var(--color-primary);
                font-weight: bold;
            }
            
            .alert-danger {
                color: var(--color-danger);
                font-weight: bold;
            }
            
            .stButton > button {
                background: linear-gradient(135deg, #0d1710 0%, #02a85d 100%);
                border: 2px solid var(--color-primary);
                color: #ffffff !important;
                font-weight: bold;
                font-family: 'Courier New', monospace;
                box-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                box-shadow: 0 0 20px rgba(0, 255, 0, 0.6);
                transform: scale(1.05);
            }
            
            h1, h2, h3 {
                color: var(--color-primary);
                font-family: 'Courier New', monospace;
                text-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
            }
            
            p, span, div {
                color: var(--color-primary);
            }
            
            .stButton > button p {
                color: #ffffff !important;
            }
            
            .stSlider {
                color: var(--color-primary);
            }
            
            .stContainer {
                background: linear-gradient(135deg, #0d1710 0%, #02a85d 100%);
                border: 1px solid rgba(0, 255, 0, 0.2);
                border-radius: 8px;
                padding: 15px;
            }
        </style>
        """
