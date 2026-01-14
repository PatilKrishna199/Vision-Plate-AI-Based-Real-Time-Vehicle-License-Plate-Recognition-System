**License Plate Detection & Recognition from Video (Real-Time)**
📌 Project Overview

This project focuses on detecting and recognizing car license plates from video in real time using Computer Vision and OCR techniques. It processes video frames, detects number plate regions, and extracts alphanumeric characters using Tesseract OCR.

The system is designed to work efficiently on recorded videos or live camera feeds and is suitable for traffic monitoring, surveillance systems, parking automation, and smart city applications.

🎯 Objectives

Detect vehicle license plates from video frames

Extract the number plate region accurately

Recognize alphanumeric characters from the plate

Display the detected plate and recognized text in real time

🧠 System Architecture
Video Input → Frame Preprocessing → Plate Detection → Plate ROI Extraction
→ OCR Processing → Text Recognition → Output Display

⚙️ Features

✅ Real-time license plate detection from video

✅ Robust contour-based plate localization

✅ Image preprocessing for better OCR accuracy

✅ Tesseract OCR for character recognition

✅ Live bounding box and text overlay

✅ Skips invalid or noisy plate regions automatically

🛠️ Technologies Used

Programming Language: Python

Libraries & Tools:

OpenCV (cv2)

NumPy

Pytesseract (OCR)

Tesseract OCR Engine

Environment: Python Virtual Environment (venv)

📂 Project Structure
AIT Project/
│
├── model/
│   ├── binary_128_0.50_ver3.pb
│   └── binary_128_0.50_labels_ver2.txt
│
├── main.py                # Original CNN-based pipeline
├── main_tesseract.py      # Tesseract OCR-based pipeline
├── PlateFinder.py         # License plate detection logic
├── OCR.py                 # CNN OCR logic (optional)
├── test_small.mp4         # Sample video
├── venv/                  # Virtual environment
└── README.md

▶️ How It Works
1️⃣ Video Capture

Reads frames continuously from a video file or camera using OpenCV.

2️⃣ Plate Detection

Applies edge detection and contour analysis.

Filters contours using aspect ratio and area constraints.

Identifies rectangular regions likely to be number plates.

3️⃣ Plate Preprocessing

Converts to grayscale.

Applies blurring and adaptive thresholding.

Enhances text visibility for OCR.

4️⃣ OCR Recognition

Uses Tesseract OCR with a character whitelist.

Extracts alphanumeric license plate text.

5️⃣ Output Display

Displays detected plate region.

Shows recognized text on video frames.

Prints detected plate numbers in terminal.

🚀 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/license-plate-detection.git
cd license-plate-detection

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install opencv-python numpy pytesseract

4️⃣ Install Tesseract OCR

Download from: https://github.com/UB-Mannheim/tesseract/wiki

Set path in code:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

▶️ Run the Project
python main_tesseract.py


Press q to exit the video window.

📊 Sample Output

Live video with bounding box around license plate

Detected license plate number printed in terminal

Plate image displayed in a separate window

Example:

Detected Plate: MH12AB1234

📈 Applications

Traffic surveillance systems

Automated toll collection

Parking management

Vehicle access control

Smart city infrastructure

Law enforcement monitoring

🔮 Future Enhancements

Integrate YOLO/Deep Learning-based plate detection

Support multiple plates per frame

Improve OCR accuracy using CNN models

Deploy as a web application (Flask / FastAPI)

Real-time webcam and CCTV stream support

👨‍💻 Author

Krishna Patil
CSE | AI & Data Science Enthusiast
Pimpri Chinchwad University

⭐ Acknowledgements

OpenCV Community

Tesseract OCR

Python Open-Source Ecosystem
