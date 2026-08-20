import warnings; warnings.filterwarnings("ignore")
import importlib.util, pandas as pd
spec = importlib.util.spec_from_file_location("hp", "hits-preprocess.py")
hp = importlib.util.module_from_spec(spec); spec.loader.exec_module(hp)
df = pd.DataFrame({
    "SMILES_Structure_Parent": ["CCO", "CCN", "c1ccccc1O", "CC(=O)O", "CCCC"],
    "Measurement_Value": ["4.3", "3.0", "5.1", "2.2", "6.0"],
    "Measurement_Unit": ["uM"] * 5,
    "Test": ["Solubility"] * 5,
    "Test_Type": ["pH7.4"] * 5})
p = hp.Preprocessor(df=df, task_type="classification", task="sol", threshold=4.0,
                    convert_units=False, correct_pH=False, scale_activity=False)
print(p.preprocess()[["Measurement_Value", "Classification_label"]].to_string(index=False))
