# Kidney Disease Prediction System -- Architecture

---

## 1. System Overview (Layered Architecture)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#0d1117', 'primaryTextColor': '#e6edf3', 'lineColor': '#58a6ff', 'fontSize': '13px'}, 'flowchart': {'rankSpacing': 50, 'nodeSpacing': 30}}}%%

graph TB
    subgraph L1["LAYER 1 -- CLIENTS"]
        C1["Web Dashboard -- Streamlit"]
        C2["REST API Clients"]
        C3["CLI -- main.py"]
    end

    subgraph L2["LAYER 2 -- API GATEWAY -- FastAPI"]
        direction LR
        E1["/predict"]
        E2["/predict/image"]
        E3["/stage"]
        E4["/chat"]
        E5["/alerts"]
        E6["/report"]
    end

    subgraph L3["LAYER 3 -- PROCESSING"]
        direction LR
        P1["Feature Engineer"]
        P2["Clinical Binning"]
        P3["Feature Alignment"]
    end

    subgraph L4["LAYER 4 -- AI ENGINE"]
        direction LR
        M1["RF"]
        M2["XGB"]
        M3["SVM"]
        M4["DNN"]
        META["Meta-Learner"]
    end

    subgraph L5["LAYER 5 -- ANALYSIS"]
        direction LR
        S1["CKD Staging"]
        S2["Risk Assessor"]
        S3["SHAP Explainer"]
    end

    subgraph L6["LAYER 6 -- SERVICES"]
        direction LR
        OCR["OCR Pipeline"]
        RAG["RAG Medical Q/A"]
        ALERT["Smart Alerts"]
    end

    subgraph L7["LAYER 7 -- OUTPUT"]
        direction LR
        O1["JSON Response"]
        O2["PDF Report"]
        O3["Chat Answer"]
    end

    L1 --> L2
    L2 --> L3
    L3 --> L4
    M1 & M2 & M3 & M4 --> META
    L4 --> L5
    L5 --> L7
    L6 --> L7

    style L1 fill:#0d1117,stroke:#58a6ff,stroke-width:2px,color:#e6edf3
    style L2 fill:#161b22,stroke:#a371f7,stroke-width:2px,color:#e6edf3
    style L3 fill:#0d1117,stroke:#3fb950,stroke-width:2px,color:#e6edf3
    style L4 fill:#161b22,stroke:#f0883e,stroke-width:2px,color:#e6edf3
    style L5 fill:#0d1117,stroke:#d29922,stroke-width:2px,color:#e6edf3
    style L6 fill:#161b22,stroke:#58a6ff,stroke-width:1px,color:#e6edf3
    style L7 fill:#0d1117,stroke:#3fb950,stroke-width:2px,color:#e6edf3
```

---

## 2. Training Pipeline (Offline)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#0d1117', 'primaryTextColor': '#e6edf3', 'lineColor': '#58a6ff', 'fontSize': '12px'}, 'flowchart': {'rankSpacing': 40}}}%%

flowchart LR
    A["Raw Datasets\n3 CSV/XLSX"] --> B["DataLoader\nMerge + Clean"]
    B --> C["Imputation\nKNN k=5"]
    C --> D["Encoding\nLabel + Scale"]
    D --> E["Clinical Binning\n6 features x 4 bins"]
    E --> F["One-Hot Encoding\ndrop_first"]
    F --> G["Interaction Features\nRatios + Products"]
    G --> H["Feature Selection\nSelectKBest + PCA"]
    H --> I["Split\n80/20 Stratified"]
    I --> J["Train 4 Models\nRF + XGB + SVM + DNN"]
    J --> K["Meta-Learner\n5-Fold Stacking"]
    K --> L["Save Models\n+ Feature Names"]

    style A fill:#161b22,stroke:#58a6ff,stroke-width:1px,color:#e6edf3
    style E fill:#161b22,stroke:#3fb950,stroke-width:2px,color:#e6edf3
    style F fill:#161b22,stroke:#3fb950,stroke-width:2px,color:#e6edf3
    style J fill:#161b22,stroke:#f0883e,stroke-width:2px,color:#e6edf3
    style K fill:#161b22,stroke:#f0883e,stroke-width:2px,color:#e6edf3
    style L fill:#161b22,stroke:#d29922,stroke-width:2px,color:#e6edf3
```

---

## 3. Inference Pipeline (Real-Time)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#0d1117', 'primaryTextColor': '#e6edf3', 'lineColor': '#58a6ff', 'fontSize': '12px'}, 'flowchart': {'rankSpacing': 40}}}%%

flowchart TB
    REQ["API Request\nJSON or Image"]
    
    subgraph INPUT["Step 1 -- Input Processing"]
        direction LR
        JSON["JSON Data"]
        IMG["Image Upload"] --> OCR_STEP["OCR Extraction"]
    end

    subgraph TRANSFORM["Step 2 -- Feature Transformation"]
        direction TB
        T1["Map to CKD_FEATURE_ORDER"]
        T2["Apply create_categorical_bins"]
        T3["Pad missing columns with 0"]
        T4["Reorder to trained schema"]
        T1 --> T2 --> T3 --> T4
    end

    subgraph PREDICT["Step 3 -- Model Prediction"]
        direction LR
        ENS["Ensemble\nWeighted Voting"]
        CONF["Confidence\nAgreement + Distance"]
    end

    subgraph ANALYZE["Step 4 -- Clinical Analysis"]
        direction LR
        STAGE["CKD Stage\neGFR + KDIGO"]
        RISK["Risk Score\n0-100 Composite"]
        SHAP["SHAP\nTop Factors"]
    end

    subgraph RESPONSE["Step 5 -- Response Assembly"]
        RESULT["prediction + probability + confidence\n+ stage + risk + alerts + xai_explanation"]
    end

    REQ --> INPUT
    JSON --> TRANSFORM
    OCR_STEP --> TRANSFORM
    TRANSFORM --> PREDICT
    ENS --> CONF
    PREDICT --> ANALYZE
    ANALYZE --> RESPONSE

    style INPUT fill:#161b22,stroke:#58a6ff,stroke-width:1px,color:#e6edf3
    style TRANSFORM fill:#161b22,stroke:#3fb950,stroke-width:2px,color:#e6edf3
    style PREDICT fill:#161b22,stroke:#f0883e,stroke-width:2px,color:#e6edf3
    style ANALYZE fill:#161b22,stroke:#d29922,stroke-width:1px,color:#e6edf3
    style RESPONSE fill:#161b22,stroke:#3fb950,stroke-width:2px,color:#e6edf3
```

---

## 4. Module Map

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#0d1117', 'primaryTextColor': '#e6edf3', 'lineColor': '#58a6ff', 'fontSize': '11px'}, 'flowchart': {'rankSpacing': 35}}}%%

graph TB
    subgraph ENTRY["Entry Points"]
        direction LR
        MAIN["main.py"]
        API["api.py"]
        ST["streamlit_app.py"]
    end

    subgraph CORE["src/ Packages"]
        direction LR
        subgraph PP["preprocessing/"]
            PP1["DataLoader"]
            PP2["FeatureEngineer"]
        end
        subgraph MD["models/"]
            MD1["MLModels"]
            MD2["DLModel"]
            MD3["EnsembleModel"]
            MD4["StagingModel"]
        end
        subgraph STG["staging/"]
            STG1["GFRCalculator"]
            STG2["RiskAssessor"]
        end
    end

    subgraph SVC["src/ Services"]
        direction LR
        subgraph EX["explainability/"]
            EX1["SHAPExplainer"]
        end
        subgraph OC["ocr/"]
            OC1["LabImageExtractor"]
        end
        subgraph RG["rag/"]
            RG1["GeminiRAG"]
        end
        subgraph MN["monitoring/"]
            MN1["SmartAlertEngine"]
            MN2["LongitudinalMonitor"]
        end
    end

    ENTRY --> CORE
    ENTRY --> SVC
    CORE --> SVC

    style ENTRY fill:#0d1117,stroke:#a371f7,stroke-width:2px,color:#e6edf3
    style CORE fill:#161b22,stroke:#3fb950,stroke-width:1px,color:#e6edf3
    style SVC fill:#161b22,stroke:#d29922,stroke-width:1px,color:#e6edf3
```

---

## 5. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| API | FastAPI + Uvicorn | Async REST with Swagger |
| ML | Scikit-learn, XGBoost | RF, SVM, Gradient Boosting |
| DL | TensorFlow / Keras | Neural Network |
| OCR | EasyOCR | Lab report text extraction |
| RAG | Gemini + ChromaDB | Medical Q/A |
| XAI | SHAP | Feature attribution |
| Reports | FPDF | PDF generation |
| Data | Pandas, NumPy | ETL + feature engineering |
| Deploy | Docker Compose | Containerization |
| UI | Streamlit | Web dashboard |
