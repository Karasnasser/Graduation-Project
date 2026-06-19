# Kidney Disease Prediction System - Technical Specifications

## 1. Project Overview
This project is a comprehensive AI-powered system for detecting, staging, and managing Chronic Kidney Disease (CKD). It combines traditional Machine Learning, Deep Learning, Computer Vision (OCR), and Large Language Models (RAG) to provide a holistic diagnostic tool.

## 2. Technology Stack

### Core Frameworks & Languages
*   **Language:** Python 3.10+
*   **Web Framework:** FastAPI (High-performance async API)
*   **Server:** Uvicorn (ASGI server)

### Artificial Intelligence & Machine Learning
*   **Machine Learning (Scikit-Learn, XGBoost):**
    *   Random Forest Classifier
    *   XGBoost (Extreme Gradient Boosting)
    *   SVM (Support Vector Machine) with RBF kernel
*   **Deep Learning (TensorFlow/Keras):**
    *   Custom Multi-layer Perceptron (MLP) networks
    *   Multi-task Learning Architecture (Simultaneous CKD detection & Staging)
    *   Features: Batch Normalization, Dropout Regularization, Adam Optimizer
*   **LLM & RAG (Retrieval-Augmented Generation):**
    *   **LLM:** Google Gemini Pro (`google-generativeai`)
    *   **Vector Database:** ChromaDB (for storing medical knowledge embeddings)
    *   **Embeddings:** Default ChromaDB embeddings / Gemini embeddings

### Computer Vision & OCR
*   **Medical Imaging (CT Scans):** MobileNetV2 (Transfer Learning CNN) for 4-class prediction (Tumor, Cyst, Stone, Normal).
*   **Primary OCR Engine:** EasyOCR (Deep learning-based OCR)
*   **Fallback OCR Engine:** Tesseract (via `pytesseract`)
*   **Image Processing:** OpenCV (`cv2`)
    *   Adaptive Thresholding
    *   Noise Removal (Bilateral Filtering)
    *   Skew Correction
    *   Contrast Enhancement (CLAHE)

### Data Processing & Utilities
*   **Data Manipulation:** Pandas, NumPy
*   **PDF Generation:** ReportLab / FPDF2
*   **Environment Management:** python-dotenv

## 3. Model Architectures & Details

### A. Ensemble Machine Learning Model
An ensemble approach is used to maximize prediction accuracy.
*   **Components:**
    1.  **Random Forest:** 100 estimators, max depth 10, balanced class weights.
    2.  **XGBoost:** 100 estimators, max depth 6, learning rate 0.1.
    3.  **SVM:** RBF kernel, probability enabled for confidence scoring.
*   **Input Features:** 24 clinical features (age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane).

### B. Deep Learning Model (Neural Network)
Designed to capture complex non-linear relationships in the medical data.
*   **Architecture:**
    *   **Input Layer:** Matches feature dimension (24 features).
    *   **Hidden Block 1:** 128 Neurons (ReLU) + BatchNormalization + Dropout(0.3).
    *   **Hidden Block 2:** 64 Neurons (ReLU) + BatchNormalization + Dropout(0.3).
    *   **Hidden Block 3:** 32 Neurons (ReLU).
    *   **Output:** Sigmoid (Binary Classification) or Softmax (Multi-class Staging).
*   **Multi-task Capability:** Can predict presence of disease AND stage (G1-G5) simultaneously via branching output layers.

### C. RAG (Medical Chatbot)
Provides intelligent explanations of lab results.
*   **Knowledge Base:** Contains KDIGO guidelines, CKD-EPI equations, and dietary recommendations.
*   **Retrieval:** Semantic search finds the most relevant medical guidelines based on user queries.
*   **Generation:** Gemini Pro synthesizes the retrieved guidelines with patient specific data (eGFR, Age, etc.) to give a personalized, medically-grounded answer.

### D. Medical Vision Model (CT Scans)
Classifies kidney CT images to assist radiologists.
*   **Architecture:** MobileNetV2 with pre-trained ImageNet weights.
*   **Custom Head:** GlobalAveragePooling2D -> Dense(128) -> Dropout(0.5) -> Dense(4, Softmax).
*   **Classes:** Normal, Cyst, Stone, Tumor.

## 4. Key Features & Modules

### 1. Lab Result Scanner (OCR)
*   **Function:** Extracts medical values directly from photos of lab reports.
*   **Pipeline:**
    1.  Image Preprocessing (Resize -> Grayscale -> Denoise -> Deskew).
    2.  Text Extraction (EasyOCR).
    3.  Regex Parsing (Identifies "Creatinine", "eGFR", "Albumin" patterns).
    4.  Structured Data JSON output.

### 2. Kidney Staging Engine (Algo)
*   **Function:** Deterministic calculation of kidney function.
*   **Guidelines:** Based on **KDIGO 2012** Clinical Practice Guidelines.
*   **Calculators:**
    *   **eGFR:** CKD-EPI 2021 Creatinine Equation.
    *   **Risk Level:** Heatmap based on GFR Stage (G1-G5) and Albuminuria (A1-A3).

### 3. Automated Report Generator
*   Generates professional PDF medical reports containing:
    *   Patient Demographics.
    *   Calculated eGFR & Staging.
    *   Risk Assessment Color Coding.
    *   Actionable Recommendations.

## 5. API Structure (Backend)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Predict CKD probability using ML/DL models. |
| `POST` | `/predict/image` | Upload lab image -> OCR -> Auto-Prediction. |
| `POST` | `/predict/ct` | Upload CT Scan image -> MobileNetV2 Classification. |
| `POST` | `/stage` | Calculate GFR Stage & Risk based on values. |
| `POST` | `/chat` | RAG-based medical Q&A about kidney health. |
| `POST` | `/diet/plan` | Generate personalized 7-day diet plan. |
| `POST` | `/report` | Generate PDF health report. |
| `GET` | `/health` | System health check. |

## 6. Deployment Requirements
*   **OS:** Windows / Linux / MacOS
*   **Python:** 3.10+
*   **RAM:** 4GB+ (Recommended 8GB for OCR & DL)
*   **Dependencies:** See `requirements.txt`
