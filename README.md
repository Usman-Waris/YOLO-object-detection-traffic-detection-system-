An end-to-end Computer Vision-powered traffic analysis system that converts raw video streams into meaningful, real-time insights. This project leverages modern object detection techniques to monitor, analyze, and visualize traffic patterns efficiently.

🚀 **Try the Live App:**  
👉 [Click here to launch the app](https://4yxmjv9nmatreqyrohx4hy.streamlit.app/)

🚀 Overview

This web application processes video input and performs real-time multi-class object detection and tracking to extract actionable traffic data. It is designed for smart city applications, traffic monitoring, and logistics optimization.

✨ Features

🔍 Multi-Class Detection
Detects and tracks:
🚗 Cars
🏍 Bikes
🚶 Pedestrians
🚛 Trucks

Powered by YOLOv11 for high-speed, high-accuracy detection

📈 Dynamic Traffic Intensity Mapping
Real-time visualization of traffic density
Graph-based insights for congestion patterns
Helps in quick decision-making

🧾 Automated Data Logging
Records every detected object
Structured data storage using Pandas
Generates detailed analytics tables
📤 Exportable Insights

Download processed data in:
CSV format
Excel format
Useful for further analysis and reporting
🛠 Tech Stack
Category	Technologies Used
Core	Python, OpenCV
Model	YOLOv11
Web Framework	Streamlit
Data Analysis	Pandas
Visualization	Matplotlib, Plotly

📦 Installation
# Clone the repository
git clone https://github.com/your-username/traffic-intelligence-app.git

# Navigate into the project folder
cd traffic-intelligence-app

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

▶️ Usage
streamlit run app.py
Upload a video or use a live stream
The app will:
Detect objects in real-time
Display traffic analytics
Log data automatically

📊 Output
Real-time annotated video feed
Traffic graphs & insights
Downloadable datasets (CSV/Excel)

🎯 Use Cases
🚦 Smart City Traffic Monitoring
🚚 Logistics & Fleet Optimization
🛣 Highway Traffic Analysis
🚓 Road Safety & Surveillance

🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

📜 License

This project is licensed under the MIT License.

💡 Future Improvements
Vehicle speed estimation
Number plate recognition
Multi-camera integration
Cloud deployment (AWS/GCP)

👨‍💻 Author

Usman Waris
