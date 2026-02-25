import pandas as pd
from config import DATA_FILE, OUTPUT_DIR, SAMPLE_SIZE
from models import InsuranceModeler
from visualizer import InsuranceVisualizer

def run_analytics():
    try:
        df = pd.read_csv(DATA_FILE)
        if SAMPLE_SIZE:
            df = df.head(SAMPLE_SIZE)
    except FileNotFoundError:
        print(f" {DATA_FILE} file not found:")
        return

    modeler = InsuranceModeler(df)
    viz = InsuranceVisualizer(OUTPUT_DIR)

    stats = modeler.get_group_statistics()
    print("\nGroup statistics (Smoker vs Non-smoker)")
    print(stats)

    regression_model = modeler.run_regression()
    print("\nMULTIPLE REGRESSION INDICATORS.")
    print(regression_model.summary())

    print("\nGraphhics...")
    viz.plot_correlations(modeler.get_correlation_matrix())
    viz.plot_distributions(df)
    viz.plot_regression_residuals(regression_model)

    print(f"\n Outouts: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    run_analytics()