## 📌 Overview

HoloMed is a real-time, contactless medical visualization system that detects abnormalities from CT scan data and renders them as interactive 3D models. The system integrates AI-based tumor detection with 3D reconstruction and visualization, enabling intuitive medical analysis without physical interaction.


## 🧪 Workflow

CT Scan (DICOM)
→ Preprocessing & Normalization
→ AI Model (Tumor Detection)
→ Segmentation Mask
→ 3D Reconstruction (Marching Cubes)
→ Brain + Tumor Visualization


## ⚙️ Installation & Setup

### 1. Clone Repository

git clone https://github.com/YOUR_USERNAME/holomed-ai.git
cd holomed-ai

---

### 2. Create Virtual Environment

python -m venv venv
venv\Scripts\activate

---

### 3. Install Dependencies

pip install -r requirements.txt

---

### 4. Add Dataset

Place CT scan DICOM files inside:
data/

---

### 5. Run Backend

python main.py

---

### 6. Run Visualization

python -m http.server 8000

Open in browser:
http://localhost:8000/viewer.html

---

## 📁 Project Structure

<img width="436" height="374" alt="image" src="https://github.com/user-attachments/assets/87a36880-f798-47f7-8364-cc17e5965a7a" />

---

## ⚠️ Important Notes

* Dataset (DICOM files) is not included due to size constraints.
* Model file (model.h5) is not uploaded due to GitHub size limits.
* Place required files locally before running the project.

