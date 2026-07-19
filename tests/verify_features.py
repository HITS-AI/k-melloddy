"""Exercise every data.py preprocessing feature to capture ground-truth expected values
for the software-certification test-criteria document."""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
# Allow running from anywhere: put the repo root (parent of tests/) on sys.path
# so `import data` resolves without needing PYTHONPATH set.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import data as D

def hr(t): print("\n" + "="*70 + f"\n{t}\n" + "="*70)

# ---------------------------------------------------------------- 1. SMILES std
hr("1. SMILES standardization (Preprocessor.preprocess_compound)")
pp = D.Preprocessor(
    df=pd.DataFrame({"smiles_structure_parent":["CC(=O)Oc1ccccc1C(=O)O"],
                     "measurement_value":[1.0]}),
    task_type="regression", task="t",
    convert_units=False, correct_pH=False, scale_activity=False)
cases = {
    "aspirin sodium salt":   "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]",
    "13C-isotope aspirin":   "[13CH3]C(=O)Oc1ccccc1C(=O)O",
    "invalid":               "C1CC1C(",
}
for name, smi in cases.items():
    try:
        std, scaf = pp.preprocess_compound(smi)
        print(f"{name:22s} {smi!r:44s} -> std={std!r}  scaffold={scaf!r}")
    except Exception as e:
        print(f"{name:22s} {smi!r:44s} -> ERROR {type(e).__name__}: {e}")
# batch handles invalid -> None (no crash); df size must match list length
pp_batch = D.Preprocessor(
    df=pd.DataFrame({"smiles_structure_parent":["CCO","C1CC1C(","c1ccccc1O"],
                     "measurement_value":[1.0,2.0,3.0]}),
    task_type="regression", task="t",
    convert_units=False, correct_pH=False, scale_activity=False)
pp_batch.preprocess_compounds(["CCO", "C1CC1C(", "c1ccccc1O"])
print("batch Standardized_SMILES:", list(pp_batch.df["Standardized_SMILES"]))

# ---------------------------------------------------------------- 2. Unit -> SI
hr("2. Unit SI conversion (UnitConverter.convert_to_si)")
uc = D.UnitConverter()
for val, unit in [(1.0,"uM"),(1.0,"nM"),(1.0,"hr"),(30.0,"min"),
                  (5.0,"ug/mL"),(2.5,"ng*h/mL"),(1.0,"10-6 cm/s"),
                  (0.8,"ratio"),(50.0,"%"),(1.0,"fold"),(3.0,"WeirdUnit")]:
    v,u,orig = uc.convert_to_si(val, unit)
    print(f"{val:>6} {unit:12s} -> {v!r:24} {u}")
print("normalize 'uM' vs 'um':", uc.normalize_unit_string("uM"), "/", uc.normalize_unit_string("um"))
# comparison operator handling via column API
dfu = pd.DataFrame({"measurement_value":[">5","<0.1","2.0"], "measurement_unit":["uM","uM","uM"]})
dfu = uc.convert_column_to_si(dfu,"measurement_value","measurement_unit","v_si","u_si")
print(dfu[["measurement_value","v_si","u_si"]].to_string(index=False))

# ---------------------------------------------------------------- 3. pH correct
hr("3. pH correction (pHCorrector.correct_activity)")
phc = D.pHCorrector(method="all", target_pH=7.4)
r = phc.correct_activity(100.0, 4.0, smiles="CC(=O)Oc1ccccc1C(=O)O")
print("activity=100 @pH4.0 ->", {k:round(v,4) if isinstance(v,float) else v for k,v in r.items()})
r2 = phc.correct_activity(100.0, 7.4, smiles="CCO")
print("activity=100 @pH7.4 (==target, unchanged) ->", r2)
r3 = phc.correct_activity(np.nan, 5.0)
print("NaN activity passthrough ->", r3)

# ---------------------------------------------------------------- 4. task type
hr("4. recognize_task_type")
for name, series in {
    "continuous->regression": ["4.3","3.0","5.1","2.2","6.0"],
    "binary 0/1->classification": ["0","1","0","1"],
    "text->classification": ["active","inactive","active"],
    "censored numeric->regression": [">5","4.3","3.0","<2.1","6.0"],
}.items():
    df = pd.DataFrame({"measurement_value": series})
    print(f"{name:32s} -> {D.recognize_task_type(df)}")

# ---------------------------------------------------------------- 5. quorum
hr("5. Training quorum (DataInspector.satisfy_training_quorum + TrainingQuorumError)")
insp = D.DataInspector(df=pd.DataFrame({"smiles_structure_parent":["CCO"],
                                        "measurement_value":["1"]}))
# regression: 60 rows, 40 uncensored -> pass; 40 rows -> fail
reg_ok = pd.DataFrame({"measurement_value":[str(i) for i in range(40)] + [">1"]*20,
                       "measurement_relation":["="]*40 + [">"]*20})
reg_bad = pd.DataFrame({"measurement_value":[str(i) for i in range(40)],
                        "measurement_relation":["="]*40})
print("regression 60 rows/40 uncensored ->", insp.satisfy_training_quorum(reg_ok,"regression"))
print("regression 40 rows              ->", insp.satisfy_training_quorum(reg_bad,"regression"))
cls_ok  = pd.DataFrame({"measurement_value":["active"]*25 + ["inactive"]*25})
cls_bad = pd.DataFrame({"measurement_value":["active"]*25 + ["inactive"]*10})
print("classification 25/25 ->", insp.satisfy_training_quorum(cls_ok,"classification"))
print("classification 25/10 ->", insp.satisfy_training_quorum(cls_bad,"classification"))
try:
    raise D.TrainingQuorumError("regression")
except D.TrainingQuorumError as e:
    print("TrainingQuorumError(regression):", e)

# ---------------------------------------------------------------- 6. endpoint map
hr("6. Endpoint mapping (ManualFormatConverter)")
conv = D.ManualFormatConverter(D.ManualConversionConfig(min_similarity=0.55, prefer_exact=True))
for ep in ["Permeability | Caco-2 | Papp",
           "CYP_Inhibition | CYP2C9 | IC50",
           "CYP_Inhibition | CYP2D6 | IC50",
           "CYP_Inhibition | CYP1A1 | IC50",
           "CYP_Inhibition | CYP3A4_MDZ | IC50",
           "Toxicity | hERG | IC50",
           "Toxicity | Cytotoxicity | CC50",
           "Solubility | Solubility | Solubility"]:
    print(f"{ep:42s} -> {conv.match_endpoint(ep)!r}")

# ---------------------------------------------------------------- 7. gist matrix
hr("7. preprocess_to_gist (end-to-end, data_sample.csv)")
gm = D.preprocess_to_gist("input_data/data_sample.csv", endpoint_mapper="manual", fill_missing=False)
print("shape:", gm.shape, "| columns[:6]:", list(gm.columns[:6]))
print("n endpoint columns:", len([c for c in gm.columns if c!="smiles"]))
nonnull = {c:int(gm[c].notna().sum()) for c in gm.columns if c!="smiles"}
print("populated endpoints:", {k:v for k,v in nonnull.items() if v>0})

# ---------------------------------------------------------------- 8. preprocess_dataframe
hr("8. preprocess_dataframe (long-format, end-to-end)")
long_df = pd.DataFrame({
    "smiles_structure_parent":["CCO","CCN","c1ccccc1O","CC(=O)O","CCCC"],
    "measurement_value":["4.3","3.0","5.1","2.2","6.0"],
    "measurement_unit":["uM"]*5,
    "test":["Solubility"]*5, "test_type":["pH7.4"]*5,
})
out = D.preprocess_dataframe(long_df, task_type="regression", task="sol",
                             convert_units=True, correct_pH=True, scale_activity=True)
print("output columns:", list(out.columns))
print(out.to_string(index=False))

# ---------------------------------------------------------------- 9. special chars
hr("9. Special char replacement + column normalization (DataInspector)")
raw = pd.DataFrame({"smiles_structure_parent":["CCO"],
                    "Measurment_Value":["5"],      # legacy typo
                    "Test_Species":["Human"],       # legacy rename
                    "Measurement_Unit":["μM"], # micro sign
                    "Measurement_Temp":["25°C"]})
insp2 = D.DataInspector(df=raw)
print("normalized columns:", list(insp2.df.columns))
print("μM ->", repr(insp2.df["measurement_unit"].iloc[0]),
      "| 25°C ->", repr(insp2.df["measurement_temp"].iloc[0]))

print("\nALL CHECKS COMPLETED")
