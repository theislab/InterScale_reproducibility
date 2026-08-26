from pathlib import Path
import pandas as pd


def merge_seed_results(
    input_dir: str = "results_seeds", output_file: str = "combined_results.csv"
):
    path = Path(input_dir)
    csv_files = sorted(path.glob("*metrics_seed_*.csv"))

    if not csv_files:
        print(f"No results files found in {input_dir}")
        return

    df_list = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(df_list, ignore_index=False)

    combined_df.to_csv(output_file, index=False)
    print(f"Aggregated {len(csv_files)} seed runs into '{output_file}':")
    print(combined_df)


if __name__ == "__main__":
    merge_seed_results()