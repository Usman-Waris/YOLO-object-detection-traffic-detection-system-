import streamlit as st
import cv2
import pandas as pd
import os
import numpy as np
from ultralytics import YOLO

# --- Page Config ---
st.set_page_config(page_title="Pro Traffic Analytics AI", layout="wide")
st.title("🚦 Advanced Traffic Intelligence & Analytics Dashboard")

# --- Model Loading ---
@st.cache_resource
def load_model():
    return YOLO("yolo11s.pt")

model = load_model()

# --- Sidebar & File Upload ---
uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])

if uploaded_video:
    temp_file_path = "temp_video_upload.mp4"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_video.read())
    
    if os.path.exists(temp_file_path):
        # --- Storage & Metrics (Updated Labels to Match YOLO Classes) ---
        unique_ids = {
            "car": set(), 
            "person": set(), 
            "motorcycle": set(), # Changed from motorbike to motorcycle
            "bus": set(), 
            "truck": set()
        }
        detections_log = []
        chart_data = [] 

        st.sidebar.header("📊 Real-Time Counters")
        car_stat = st.sidebar.empty()
        bike_stat = st.sidebar.empty()
        bus_stat = st.sidebar.empty()
        truck_stat = st.sidebar.empty()
        person_stat = st.sidebar.empty()
        traffic_status = st.sidebar.empty()
        
        st.subheader("📺 Live Processing Feed")
        col_vid, col_chart = st.columns([2, 1]) 
        
        frame_placeholder = col_vid.empty()
        chart_placeholder = col_chart.empty()

        # --- Running Tracking ---
        results = model.track(
            source=temp_file_path,
            stream=True,
            conf=0.25,
            tracker="bytetrack.yaml"
        )

        for idx, r in enumerate(results):
            frame = r.plot()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            current_counts = {k: 0 for k in unique_ids.keys()}

            if r.boxes and r.boxes.id is not None:
                ids = r.boxes.id.int().cpu().tolist()
                clss = r.boxes.cls.int().cpu().tolist()
                confs = r.boxes.conf.cpu().tolist()

                for id, cls, conf in zip(ids, clss, confs):
                    label = r.names[cls] # YOLO gives "motorcycle", "car", etc.
                    if label in unique_ids:
                        unique_ids[label].add(id)
                        current_counts[label] += 1
                        
                        detections_log.append({
                            "frame": idx,
                            "label": label,
                            "track_id": id,
                            "confidence": round(conf, 2)
                        })

            # --- Live Graph Logic ---
            if idx % 10 == 0:
                total_now = sum(current_counts.values())
                chart_data.append({"Frame": idx, "Live Traffic Load": total_now})
                chart_placeholder.line_chart(pd.DataFrame(chart_data).set_index("Frame"))

            # Update UI (Sidebar Metrics)
            car_stat.metric("Cars 🚗", len(unique_ids["car"]))
            bike_stat.metric("Motorcycles 🏍️", len(unique_ids["motorcycle"]))
            bus_stat.metric("Buses 🚌", len(unique_ids["bus"]))
            truck_stat.metric("Trucks 🚚", len(unique_ids["truck"]))
            person_stat.metric("Persons 🚶", len(unique_ids["person"]))

            frame_placeholder.image(frame, use_column_width=True)

        # --- Final Analytics Logic ---
        total_objects = sum(len(v) for v in unique_ids.values())
        level = "Low" if total_objects < 20 else "Medium" if total_objects < 50 else "High"
        traffic_status.subheader(f"Status: {level}")

        # --- ADVANCED REPORTS SECTION ---
        st.divider()
        st.header("📈 Post-Analysis Insights")
        
        tab1, tab2, tab3 = st.tabs(["📋 Summary Table", "📊 Traffic Flow Graph", "📥 Export Data"])
        
        with tab1:
            summary_df = pd.DataFrame({
                "Category": ["Cars", "Motorcycles", "Buses", "Trucks", "Persons", "Total Detected"],
                "Count": [
                    len(unique_ids["car"]), 
                    len(unique_ids["motorcycle"]), 
                    len(unique_ids["bus"]),
                    len(unique_ids["truck"]),
                    len(unique_ids["person"]), 
                    total_objects
                ]
            })
            st.table(summary_df)

        with tab2:
            st.write("Video ke doran traffic density ka graph:")
            full_chart_df = pd.DataFrame(chart_data)
            if not full_chart_df.empty:
                st.area_chart(full_chart_df.set_index("Frame"))

        with tab3:
            col_a, col_b = st.columns(2)
            if detections_log:
                df_det = pd.DataFrame(detections_log)
                col_a.download_button("Download Detailed CSV", df_det.to_csv(index=False).encode('utf-8'), "traffic_analytics.csv", "text/csv")
                col_b.download_button("Download Summary CSV", summary_df.to_csv(index=False).encode('utf-8'), "summary_report.csv", "text/csv")

        # Cleanup
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    else:
        st.error("Error processing video.")
