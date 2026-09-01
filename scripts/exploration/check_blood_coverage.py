"""Checks per-test fill rate (coverage) of the tiered blood panel among linked admissions."""
import duckdb

LABEVENTS = "data/raw/mimic-iv/labevents.csv"
RESOLVED_LINKS = "data/interim/resolved_study_links.csv"

TIER1 = {"WBC": 51301, "RBC": 51279, "Hemoglobin": 51222, "Hematocrit": 51221,
         "Platelets": 51265, "Neutrophils": 51256, "Lymphocytes": 51244, "Monocytes": 51254}
TIER2 = {"CRP": 50889, "ESR": 51288, "Albumin": 50862, "LDH": 50954}
TIER3 = {"Sodium": 50983, "Potassium": 50971, "Glucose": 50931, "Creatinine": 50912, "Urea": 51006}
ALL_TESTS = {**TIER1, **TIER2, **TIER3}

con = duckdb.connect()
con.execute(f"""
    CREATE TEMP TABLE linked_hadm AS
    SELECT DISTINCT hadm_id FROM read_csv_auto('{RESOLVED_LINKS}')
""")
n_hadm = con.execute("SELECT COUNT(*) FROM linked_hadm").fetchone()[0]
print(f"Linked admissions: {n_hadm}\n")

itemid_list = ",".join(str(v) for v in ALL_TESTS.values())

df = con.execute(f"""
    SELECT l.itemid, COUNT(DISTINCT l.hadm_id) AS n_covered
    FROM read_csv_auto('{LABEVENTS}') l
    WHERE l.hadm_id IN (SELECT hadm_id FROM linked_hadm)
      AND l.itemid IN ({itemid_list})
    GROUP BY l.itemid
""").df()

coverage = dict(zip(df["itemid"], df["n_covered"]))

print(f"{'Test':<12} {'itemid':<8} {'covered':<10} {'coverage %':<10}")
for name, itemid in ALL_TESTS.items():
    n = coverage.get(itemid, 0)
    print(f"{name:<12} {itemid:<8} {n:<10} {100*n/n_hadm:.1f}%")
