import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.sna.build_graphs import build_and_export_graphs
from src.sna.feature_engineering import extract_sna_features

def run_full_sna_pipeline():
    print("=" * 70)
    print("CYBERSHIELD AI — PHASE 5: COMPLETE SNA PIPELINE")
    print("=" * 70)

    # Step 3: Build & Export NetworkX graphs
    build_and_export_graphs()

    # Step 4: Extract Node & Dyad SNA features
    extract_sna_features()

    print("\nPhase 5 pipeline execution complete.")

if __name__ == "__main__":
    run_full_sna_pipeline()
