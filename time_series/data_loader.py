import pandas as pd
from pathlib import Path

#  DATA LOADER

class QuarterlyDataLoader:
    def __init__(self, filepath: str):
        self.path = Path(filepath)

    def load(self) -> pd.Series:
        df = pd.read_excel(self.path, skiprows=1, header=0)
        df.columns = ["date", "value"]
        df = df.dropna(subset=["date", "value"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        s = df.set_index("date")["value"].astype(float)
        s.index = pd.PeriodIndex(s.index, freq="Q")
        return s

