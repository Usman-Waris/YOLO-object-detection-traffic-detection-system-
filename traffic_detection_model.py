"""
🔥 ADVANCED YOLOv11 TRAFFIC ANALYTICS PIPELINE
- Object Detection + Tracking (ByteTrack)
- Unique Vehicle Counting
- Frame-wise Analytics
- Traffic Density Estimation
- CSV Export (detections + summary)
"""


import os
import sys
import cv2
import pandas as pd
from ultralytics import YOLO

# ================= CONFIG =================
VIDEO_PATH = "object detection sample.mp4"
MODEL_NAME = "yolo11s.pt"

SAVE_VIDEO = True
SAVE_CSV = True
CSV_DIR = "results"

SHOW_PREVIEW = False
CONF = 0.25
# =========================================


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def main():

    if not os.path.exists(VIDEO_PATH):
        print("❌ Video not found")
        return

    print("🚀 Loading model...")
    model = YOLO(MODEL_NAME)

    print("🎥 Running tracking...")

    results = model.track(
        source=VIDEO_PATH,
        stream=True,
        save=SAVE_VIDEO,
        show=SHOW_PREVIEW,
        conf=CONF,
        tracker="bytetrack.yaml"
    )

    # ====== TRACKING STORAGE ======
    unique_ids = {
        "car": set(),
        "person": set(),
        "motorbike": set(),
        "bus": set(),
        "truck": set()
    }

    detections = []
    frame_counts = []

    frame_idx = 0

    for r in results:
        frame_idx += 1

        counts = {k: 0 for k in unique_ids.keys()}

        if r.boxes is None:
            continue

        img_h, img_w = r.orig_shape

        for box in r.boxes:
            cls_id = int(box.cls.item())
            label = r.names[cls_id]

            track_id = int(box.id.item()) if box.id is not None else -1

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf.item())

            # ===== COUNTING =====
            if label in counts:
                counts[label] += 1
                if track_id != -1:
                    unique_ids[label].add(track_id)

            # ===== SAVE DETECTIONS =====
            detections.append({
                "frame": frame_idx,
                "label": label,
                "track_id": track_id,
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

        # save per frame counts
        frame_counts.append({
            "frame": frame_idx,
            **counts
        })

    # ================= FINAL SUMMARY =================
    summary = {
        "cars": len(unique_ids["car"]),
        "persons": len(unique_ids["person"]),
        "bikes": len(unique_ids["motorbike"]),
        "buses": len(unique_ids["bus"]),
        "trucks": len(unique_ids["truck"])
    }

    total_vehicles = (
        summary["cars"] +
        summary["bikes"] +
        summary["buses"] +
        summary["trucks"]
    )

    # Traffic logic
    if total_vehicles < 20:
        traffic = "Low"
    elif total_vehicles < 50:
        traffic = "Medium"
    else:
        traffic = "High"

    print("\n🚦 FINAL TRAFFIC REPORT")
    print(summary)
    print("Traffic Level:", traffic)

    # ================= SAVE CSV =================
    if SAVE_CSV:
        make_dir(CSV_DIR)

        pd.DataFrame(detections).to_csv(f"{CSV_DIR}/detections.csv", index=False)
        pd.DataFrame(frame_counts).to_csv(f"{CSV_DIR}/frame_counts.csv", index=False)

        summary_df = pd.DataFrame([{
            **summary,
            "traffic_level": traffic,
            "total_vehicles": total_vehicles
        }])

        summary_df.to_csv(f"{CSV_DIR}/summary.csv", index=False)

        print("📊 CSV files saved in:", CSV_DIR)

    print("✅ DONE")


if __name__ == "__main__":
    main()