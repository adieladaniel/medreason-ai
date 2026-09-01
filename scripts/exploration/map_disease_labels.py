"""
Resolves ambiguous multi-admission study links (nearest admittime wins), then maps
linked hadm_ids to the 5 target disease labels via ICD-9/ICD-10 code ranges.
"""
import duckdb

ADMISSIONS = "data/raw/mimic-iv/admissions.csv"
DIAGNOSES = "data/raw/mimic-iv/diagnoses_icd.csv"
LINKS = "data/interim/cxr_admission_links.csv"

con = duckdb.connect()

con.execute(f"CREATE TEMP TABLE links AS SELECT * FROM read_csv_auto('{LINKS}')")

# tie-break: for studies matching >1 admission, keep the admission whose admittime
# is closest to study_datetime
con.execute("""
    CREATE TEMP TABLE resolved AS
    SELECT * FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY study_id
                ORDER BY ABS(EPOCH(study_datetime) - EPOCH(admittime))
            ) AS rn
        FROM links
    )
    WHERE rn = 1
""")

n_resolved = con.execute("SELECT COUNT(*) FROM resolved").fetchone()[0]
print(f"Resolved studies after tie-break: {n_resolved}")

con.execute(f"""
    CREATE TEMP TABLE diag AS
    SELECT hadm_id, icd_code, icd_version
    FROM read_csv_auto('{DIAGNOSES}', types={{'icd_code': 'VARCHAR'}})
    WHERE hadm_id IN (SELECT DISTINCT hadm_id FROM resolved)
""")

# ICD-9 / ICD-10 code ranges for the 5 target classes
con.execute("""
    CREATE TEMP TABLE labeled AS
    SELECT
        hadm_id,
        MAX(CASE WHEN icd_version = 9  AND icd_code BETWEEN '480'  AND '4869' THEN 1
                  WHEN icd_version = 10 AND icd_code BETWEEN 'J12' AND 'J189' THEN 1
                  ELSE 0 END) AS pneumonia,
        MAX(CASE WHEN icd_version = 9  AND icd_code BETWEEN '010'  AND '0189' THEN 1
                  WHEN icd_version = 10 AND icd_code BETWEEN 'A15' AND 'A19Z' THEN 1
                  ELSE 0 END) AS tuberculosis,
        MAX(CASE WHEN icd_version = 10 AND icd_code = 'U071' THEN 1
                  ELSE 0 END) AS covid19,
        MAX(CASE WHEN icd_version = 9  AND icd_code BETWEEN '1620' AND '1629' THEN 1
                  WHEN icd_version = 10 AND icd_code BETWEEN 'C34' AND 'C34Z' THEN 1
                  ELSE 0 END) AS lung_cancer
    FROM diag
    GROUP BY hadm_id
""")

print("\n=== Per-disease positive hadm_id counts (multi-label, not mutually exclusive) ===")
row = con.execute("""
    SELECT
        SUM(pneumonia) AS pneumonia,
        SUM(tuberculosis) AS tuberculosis,
        SUM(covid19) AS covid19,
        SUM(lung_cancer) AS lung_cancer,
        SUM(CASE WHEN pneumonia + tuberculosis + covid19 + lung_cancer = 0 THEN 1 ELSE 0 END) AS no_target_dx
    FROM labeled
""").fetchone()
print(f"Pneumonia:    {row[0]}")
print(f"Tuberculosis: {row[1]}")
print(f"COVID-19:     {row[2]}")
print(f"Lung Cancer:  {row[3]}")
print(f"No target diagnosis code present (candidate 'Normal' pool, unverified): {row[4]}")
print(f"Total labeled hadm_ids: {n_resolved if False else con.execute('SELECT COUNT(*) FROM labeled').fetchone()[0]}")

overlap = con.execute("""
    SELECT SUM(pneumonia + tuberculosis + covid19 + lung_cancer > 1) FROM labeled
""").fetchone()[0]
print(f"\nhadm_ids positive for >1 target disease simultaneously: {overlap}")

con.execute("""
    COPY labeled TO 'data/interim/disease_labels.csv' (HEADER, DELIMITER ',')
""")
print("\nSaved to data/interim/disease_labels.csv")
