"""Reuse native-chip review; identity/learned terms subtract for relative scores."""
import importlib.util
from pathlib import Path
import run as experiment

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('relative_case_review', HERE.parent / 'learned_edge_metric/analyze_cases.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
review.HERE = HERE
review.OUT = HERE / 'analysis'
review.experiment = experiment

if __name__ == '__main__':
    review.main()
