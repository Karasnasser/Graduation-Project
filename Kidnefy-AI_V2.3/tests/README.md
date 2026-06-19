# Tests Directory

This directory is reserved for automated tests to ensure the reliability and correctness of the Kidney Disease Prediction System.

## Purpose
-   **Unit Tests:** Test individual functions (e.g., `calculate_egfr`, `preprocess_image`).
-   **Integration Tests:** Test how modules work together (e.g., Image -> OCR -> Prediction).
-   **Regression Tests:** Ensure new changes don't break existing features.

## How to Run Tests
1.  Install dependencies:
    ```bash
    pip install pytest
    ```
2.  Run all tests:
    ```bash
    pytest
    ```

## Structure (Recommended)
-   `test_preprocessing.py`: Tests for data loading and cleaning.
-   `test_models.py`: Tests for model training and prediction.
-   `test_ocr.py`: Tests for image processing.
-   `test_api.py`: Tests for API endpoints.
