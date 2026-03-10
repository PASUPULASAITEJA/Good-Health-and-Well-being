import warnings
from sklearn.datasets import fetch_openml, load_breast_cancer
import pandas as pd

warnings.filterwarnings('ignore')

with open("datasets_info.txt", "w") as f:
    f.write("--- Breast Cancer ---\n")
    bc = load_breast_cancer(as_frame=True)
    f.write(", ".join(bc.frame.columns) + "\n")
    
    f.write("\n--- Heart Disease ---\n")
    hd = fetch_openml(name='heart', version=1, as_frame=True, parser='auto')
    f.write(", ".join(hd.frame.columns) + "\n")
