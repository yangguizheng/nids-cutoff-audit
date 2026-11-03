import importlib.util
import pandas as pd

spec = importlib.util.spec_from_file_location('ff', 'scripts_v2/figures_final.py')
ff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ff)


sm = pd.read_csv("results_v2/wp4d_summary.csv")
sm20 = pd.read_csv("results_v2/wp4d_summary_unsw_s20_val.csv")
sm = pd.concat([sm[sm.setting != "unsw_pair"], sm20], ignore_index=True)
clean = pd.read_csv("results_v2/wp4_clean_run.csv")

ff.REPORT = []
ff.fig4(sm, clean, "results_v2/r3/Fig4.pdf")

with open("results_v2/r3/COLLISION_REPORT_Fig4.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(ff.REPORT) + "\n")
print("Fig4.pdf written (R9: validation rules hollow + dagger + note)")
print("\n--- COLLISION REPORT (R9 / Fig4) ---")
for line in ff.REPORT:
    print("  " + line)
