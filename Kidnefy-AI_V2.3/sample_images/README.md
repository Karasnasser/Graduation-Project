# Sample Images Directory

This directory is intended for storing **sample lab report images** to test the Optical Character Recognition (OCR) system.

## Usage
1.  Place your medical lab report images here (e.g., `.jpg`, `.png`).
2.  Run the prediction command pointing to an image in this folder:
    ```bash
    python main.py predict --image sample_images/my_report.jpg
    ```

## Expected Format
-   Images should be clear and readable.
-   Ideally contain standard kidney function test results (Creatinine, Urea, etc.).
-   Supported formats: JPG, PNG, JPEG.
