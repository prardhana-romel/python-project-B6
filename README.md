# Deepfake Video Detector

This project is a Streamlit web app for detecting deepfakes in uploaded video files.  
It is powered by custom neural network models using PyTorch, Torch Geometric, OpenCV, and MTCNN for face detection.

### 🚀 Live Demo

You can try the app instantly here:  
[https://python-project-b6-k3qoban5bfzg7mp8sxftvw.streamlit.app/](https://python-project-b6-k3qoban5bfzg7mp8sxftvw.streamlit.app/)

---

## 📁 Project structure

- `app.py` — Main Streamlit web app code
- `model_definitions.py`, `my_models.py` — Deep learning and graph neural network model code
- `requirements.txt` — All dependencies required to run this project

---

## ⚡ Model File Handling (funet_a_full.pth)

The main model file (`funet_a_full.pth`) is **too large to be uploaded to GitHub**.

- Instead, the app will automatically download it from Google Drive when needed using `gdown`.
- No manual download is required.

**How it works:**  
- If the model file is not present, it downloads from a Google Drive link via `gdown` (see top of `app.py` for the code).

**Google Drive Model Link:**  
- (Optionally paste your Google Drive shareable link here for reference.)

---

## 💾 Installation

To run locally:
1. Clone the repo
2. Run:
    ```
    pip install -r requirements.txt
    streamlit run app.py
    ```
The app will download the model file automatically.

---

## 📝 Submission Links

- **Streamlit app:**  
  [https://python-project-b6-k3qoban5bfzg7mp8sxftvw.streamlit.app/](https://python-project-b6-k3qoban5bfzg7mp8sxftvw.streamlit.app/)
- **GitHub repo:**  
  [https://github.com/prardhana-romel/python-project-B6](https://github.com/prardhana-romel/python-project-B6)

---

## 🔒 .gitignore

The following file is excluded from GitHub by `.gitignore`:
