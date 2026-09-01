"""
Merges ICD-based (admission-level) and report-based (study-level) signals into
final validated single labels for the 3-class target (Normal, Pneumonia, Lung Cancer).

Rule: Pneumonia/Cancer require BOTH an ICD code on the admission AND a positive
report mention on that specific study (reduces noise from unrelated admission codes).
Normal requires the report to show No Finding AND the admission has none of the
target-disease codes (including excluded Tuberculosis, to avoid mislabeling a
genuinely diseased-but-off-target patient as Normal).
"""
import duckdb

DIAGNOSES = "data/raw/mimic-iv/diagnoses_icd.csv"
RESOLVED_LINKS = "data/interim/resolved_study_links.csv"
REPORT_LABELS = "data/interim/report_labels.csv"
OUTPUT = "data/interim/final_study_labels.csv"

con = duckdb.connect()

con.execute(f"CREATE TEMP TABLE resolved AS SELECT * FROM read_csv_auto('{RESOLVED_LINKS}')")
con.execute(f"CREATE TEMP TABLE report_labels AS SELECT * FROM read_csv_auto('{REPORT_LABELS}')")
con.execute(f"""
    CREATE TEMP TABLE diag AS
    SELECT hadm_id, icd_code, icd_version
    FROM read_csv_auto('{DIAGNOSES}', types={{'icd_code': 'VARCHAR'}})
    WHERE hadm_id IN (SELECT DISTINCT hadm_id FROM resolved)
""")

con.execute("""
    CREATE TEMP TABLE icd_labels AS
    SELECT
        hadm_id,
        MAX(CASE WHEN icd_version = 9  AND icd_code BETWEEN '480'  AND '4869' THEN 1
                  WHEN icd_version = 10 AND icd_code BETWEEN 'J12' AND 'J189' THEN 1
                  ELSE 0 END) AS icd_pneumonia,
        MAX(CASE WHEN icd_version = 9  AND icd_code BETWEEN '010'  AND '0189' THEN 1
                  WHEN icd_version = 10 AND icd_code BETWEEN 'A15' AND 'A19Z' THEN 1
                  ELSE 0 END) AS icd_tb,
        MAX(CASE WHEN icd_version = 9  AND icd_code BETWEEN '1620' AND '1629' THEN 1
                  WHEN icd_version = 10 AND icd_code BETWEEN 'C34' AND 'C34Z' THEN 1
                  ELSE 0 END) AS icd_cancer
    FROM diag
    GROUP BY hadm_id
""")

con.execute("""
    CREATE TEMP TABLE merged AS
    SELECT
        r.subject_id, r.study_id, r.hadm_id,
        COALESCE(i.icd_pneumonia, 0) AS icd_pneumonia,
        COALESCE(i.icd_tb, 0) AS icd_tb,
        COALESCE(i.icd_cancer, 0) AS icd_cancer,
        rl.pneumonia_mentioned,
        rl.cancer_mentioned,
        rl.no_finding
    FROM resolved r
    LEFT JOIN icd_labels i ON r.hadm_id = i.hadm_id
    LEFT JOIN report_labels rl ON r.study_id = rl.study_id
""")

con.execute("""
    CREATE TEMP TABLE final AS
    SELECT *,
        CASE
            WHEN icd_pneumonia = 1 OR pneumonia_mentioned THEN 'Pneumonia'
            WHEN icd_cancer = 1 OR cancer_mentioned THEN 'Lung Cancer'
            WHEN icd_pneumonia = 0 AND icd_cancer = 0 AND icd_tb = 0 AND no_finding THEN 'Normal'
            ELSE 'Excluded'
        END AS final_label
    FROM merged
""")

dist = con.execute("SELECT final_label, COUNT(*) FROM final GROUP BY 1 ORDER BY 2 DESC").df()
print("=== Final label distribution (study-level) ===")
print(dist)

con.execute(f"""
    COPY final TO '{OUTPUT}' (HEADER, DELIMITER ',')
""")
print(f"\nSaved to {OUTPUT}")
