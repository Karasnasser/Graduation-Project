"""Reorganize project structure - move files to proper directories."""
import shutil
import os
from pathlib import Path

root = Path(__file__).parent

# Create target directories
(root / "tests").mkdir(exist_ok=True)
(root / "scripts").mkdir(exist_ok=True)
(root / "docs").mkdir(exist_ok=True)

# Test files -> tests/
test_files = [
    "test_all_ai_tasks.py",
    "test_chatbot.py",
    "test_client.py",
    "test_ocr.py",
    "test_simple.py",
    "test_smart_alerts_standalone.py",
    "verify_integration.py",
    "verify_staging.py",
]

# Utility scripts -> scripts/
script_files = [
    "add_knowledge.py",
    "analyze_datasets.py",
    "analyze_new_dataset.py",
    "inspect_datasets.py",
    "list_models.py",
    "read_pdf.py",
    "chatbot_ui.py",
    "pdf_content.txt",
    "train_staging.py",
]

# Documentation -> docs/
doc_files = [
    "API_DOCUMENTATION.md",
    "TECHNICAL_SPECIFICATIONS.md",
    "PROJECT_HANDOVER.md",
    "PROJECT_DETAILS_AR.md",
    "PROJECT_ARCHITECTURE_AR.md",
    "PROJECT_SYSTEM_ANALYSIS_AR.md",
    "PROJECT_REPORT_AR.md",
    "SMART_ALERTS.md",
    "FUTURE_IMPROVEMENTS.md",
]

moved = 0
for f in test_files:
    src = root / f
    if src.exists():
        shutil.move(str(src), str(root / "tests" / f))
        print(f"  tests/{f}")
        moved += 1

for f in script_files:
    src = root / f
    if src.exists():
        shutil.move(str(src), str(root / "scripts" / f))
        print(f"  scripts/{f}")
        moved += 1

for f in doc_files:
    src = root / f
    if src.exists():
        shutil.move(str(src), str(root / "docs" / f))
        print(f"  docs/{f}")
        moved += 1

# Remove empty reports/ directory if it exists and is empty
reports_dir = root / "reports"
if reports_dir.exists() and not any(reports_dir.iterdir()):
    reports_dir.rmdir()
    print("  Removed empty reports/ directory")

print(f"\nDone! Moved {moved} files.")
