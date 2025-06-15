# 🔍 Face Similarity Finder

A simple **Streamlit web app** that allows users to upload their own image and compare it against a gallery of photos to find the most visually similar faces using facial recognition and Euclidean distance. Built using `dlib`, `numpy`, and `streamlit`.

---

## Demo

**Run Locally** – see setup instructions below.

---

## How It Works

1. The user uploads a **target image** (a selfie or portrait).
2. The user uploads a **zip file** of multiple face images (gallery).
3. The app detects and encodes all faces using a pre-trained 128D face embedding model.
4. It compares the target image to each gallery face using **Euclidean distance**.
5. Returns and displays the **top N matches** under a user-defined threshold.

---

## Features

- Upload a **face image** to search
- Upload a **zip of gallery photos**
- Automatically detects faces using HOG-based detection
- Calculates **similarity using Euclidean distance**
- Allows users to **enter custom threshold**
- Displays the **top 5 similar faces** sorted by closeness

---

## Tech Stack

- **Python 3.9+**
- **Streamlit** – UI framework
- **dlib** – face detection & encoding
- **NumPy** – numerical computing
- **Pillow** – image processing

---

## Installation & Usage

### 1. Clone this repository

```bash
git clone https://github.com/bhoomikaagrawal16/face-similarity-app.git
cd face-similarity-finder
```

### 2. (Optional) Create a virtual environment

```bash
conda create -n face_similarity_env python=3.9
conda activate face_similarity_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```
---
## Similarity Threshold

You can type in a custom **similarity threshold** (default: `0.6`) in the app.

- **Lower values** (e.g. `0.4`) mean stricter matching — fewer but more accurate results.
- **Higher values** (e.g. `0.7`) allow looser matches — more results but possibly less relevant.

| Distance Range | Interpretation       |
|----------------|----------------------|
| `0.2 – 0.4`     | Likely the same person |
| `0.4 – 0.6`     | Possibly the same      |
| `> 0.6`         | Likely different people |

---
## 📸 Sample Usage

- Set the similarity threshold
  
  <img width="555" alt="{1C037E35-7AEA-414E-BF4B-79204E4B6D70}" src="https://github.com/user-attachments/assets/e2229837-db00-4d6d-ae99-6f09014c4ff3" />

- Upload your own face image like `me.jpg`

  <img width="551" alt="{3CD8C25A-4BA3-4572-9182-7D55EEA50394}" src="https://github.com/user-attachments/assets/81b4e8a8-0f13-4bf7-b4ed-49f89e771106" />

- Upload a zip file of gallery photos (e.g., `photos.zip`)

  <img width="539" alt="{D59CD381-1ADC-4079-A450-92A38C45CC06}" src="https://github.com/user-attachments/assets/5d94c9f2-03f0-4d22-9398-688dd6d8f538" />

- The app detects faces in the gallery and compares them to your image
- It returns the **top 5 visually similar faces**, sorted by similarity

  <img width="219" alt="{49B5679A-EB9D-46E1-AF15-19B0175FEC4F}" src="https://github.com/user-attachments/assets/f26c432c-a282-4095-ba61-ebd62e77f9cd" />

---

## 👤 Author

**Bhoomika Agrawal**  
[🔗 GitHub](https://github.com/bhoomikaagrawal16)  
[🔗 LinkedIn](https://www.linkedin.com/in/bhoomikaagrawal)
