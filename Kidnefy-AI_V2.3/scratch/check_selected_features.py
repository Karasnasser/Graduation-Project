import joblib
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

path = Path("models")
if (path / "feature_selector.joblib").exists():
    selector = joblib.load(path / "feature_selector.joblib")
    if (path / "full_feature_names.joblib").exists():
        full_features = joblib.load(path / "full_feature_names.joblib")
        mask = selector.get_support()
        selected = [f for f, m in zip(full_features, mask) if m]
        print("Selected features count:", len(selected))
        print("Selected features:")
        for f in selected:
            print(f" - {f}")
        print("\nDropped features:")
        for f in [f for f, m in zip(full_features, mask) if not m]:
            print(f" - {f}")
    else:
        print("full_feature_names.joblib not found")
else:
    print("feature_selector.joblib not found")
