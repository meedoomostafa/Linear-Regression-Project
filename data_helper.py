import pandas as pd
import numpy as np

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].values  # All columns except last
        y = data.iloc[:, -1].values  # Last column
        return {'x': X, 'y': y}
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

