import importlib.util
spec = importlib.util.spec_from_file_location('st', 'scripts_v2/supplemental_tables.py')
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)
st.OUT = "results_v2/r3/Supplemental_Tables.xlsx"
st.main()
print("WROTE", st.OUT)
