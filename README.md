🧠 PART 1: FACE DATA COLLECTION (Backend)

✅ File: face_data_capture.py (Your first script)
This script collects face images of a person and saves them for training.

🔧 Backend Logic
• Camera Initialization: Starts video capture using OpenCV.
• Face Detection: Uses Haarcascade to detect faces in real time.
• Face Capture & Resizing: Captures faces every 3rd frame, resizes to 50x50 pixels.
• Multithreading: Speeds up processing by resizing faces in parallel threads.
• Data Saving:
◦ Flattens images and stores them in faces_data.pkl using pickle.
◦ Stores corresponding user names in names.pkl.

🎯 Output
• Trained image data and names saved in data/ directory.
• These are used later for recognition and attendance.


🧠 PART 2: FACE RECOGNITION & ATTENDANCE
(Backend + Real-time UI)
✅ File: face_recognition_attendance.py (Your second script)
This script uses the saved face data to recognize users in real time and mark attendance.

🔧 Backend Logic
• Loads Data: Reads faces_data.pkl and names.pkl.
• Trains Classifier: Uses KNeighborsClassifier from scikit-learn.
• Camera Feed: Captures live video using OpenCV.
• Face Detection: Detects faces and predicts their identity using the trained model.
• Attendance CSV: Creates/updates a CSV file like Attendance_25-05-2025.csv
with name and timestamp.
• Text-to-Speech: Uses pyttsx3 to audibly confirm attendance.
• Custom UI Frame: Optionally overlays the live camera feed onto a background image
(background.png) for better visuals.

🎯 User Interaction
• Press 'o': Takes attendance and writes to CSV.
• Press 'q': Quits the application.

🎯 Output
• CSV attendance file per day inside Attendance/ folder.
• Visual window showing live video, recognized names, and drawing boxes around faces.


🧠 PART 3: ATTENDANCE DASHBOARD (Frontend)
✅ File: streamlit_dashboard.py (Your third script)
This is a web-based dashboard using Streamlit to view attendance records.

🖥 Frontend Logic (via Streamlit)
• Automatically Detects Date: Gets today's date to show that day's attendance.
• Checks if File Exists: If not, creates a new file with headers.
• Loads CSV: If file exists, loads attendance data into a DataFrame.
• Displays Data: Shows the DataFrame (NAME, TIME) on a web page.

✅ Streamlit UI
• st.write() is used to:
◦ Show messages (e.g., “Attendance file created”).
◦ Display a table of attendance data.

🎯 Output
• A clean and modern web interface showing who was present and when.


🔄 Full Workflow Integration
dashboard.
Steps
1 face_data_capture.py - You collect faces and names of users.
2 face_recognition_attendance.py - System uses webcam to recognize users and mark
attendance.
3 Attendance/ folder - Stores CSV files for each day with name & time.
4 streamlit_dashboard.py - Loads today’s CSV and displays it on a web
