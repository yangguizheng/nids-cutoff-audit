import openpyxl
NEW = "results_v2/r3/Supplemental_Tables.xlsx"
wb = openpyxl.load_workbook(NEW, data_only=True)

for name in ["S8", "S10", "S13", "S6"]:
    ws = wb[name]
    print("====", name, "dims", ws.dimensions, "====")
    for i, row in enumerate(ws.iter_rows(min_row=1, max_row=4, values_only=True), 1):

        vals = list(row)
        while vals and vals[-1] is None:
            vals.pop()
        print("  r%d:" % i, vals[:8])
    print()
