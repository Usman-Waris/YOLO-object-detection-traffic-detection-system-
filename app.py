import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO

st.title("🚦 Smart Traffic Analysis System")

model = YOLO("yolo11s.pt")

uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4"])

if uploaded_video:

    with open("temp.mp4", "wb") as f:
        f.write(uploaded_video.read())

    results = model.track(
        source="temp.mp4",
        stream=True,
        conf=0.25,
        tracker="bytetrack.yaml"
    )

    unique_ids = {
        "car": set(),
        "person": set(),
        "motorbike": set()
    }

    frame_placeholder = st.empty()

    for r in results:
        frame = r.plot()

        if r.boxes:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                label = r.names[cls_id]

                if box.id is not None:
                    track_id = int(box.id.item())

                    if label in unique_ids:
                        unique_ids[label].add(track_id)

        frame_placeholder.image(frame, channels="BGR")

    # Final Results
    cars = len(unique_ids["car"])
    persons = len(unique_ids["person"])
    bikes = len(unique_ids["motorbike"])

    total = cars + bikes

    if total < 20:
        traffic = "Low"
    elif total < 50:
        traffic = "Medium"
    else:
        traffic = "High"

    st.subheader("📊 Final Report")

    st.write(f"Cars: {cars}")
    st.write(f"Bikes: {bikes}")
    st.write(f"Persons: {persons}")
    st.write(f"Traffic Level: {traffic}")