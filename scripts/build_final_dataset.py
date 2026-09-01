"""
Assembles the final multimodal training table: one row per study, with the
primary chest X-ray image path, radiology report path, tiered blood feature
values (closest to study_datetime), and the validated final disease label.
"""
import duckdb

FINAL_LABELS = "data/interim/final_study_labels.csv"
RESOLVED_LINKS = "data/interim/resolved_study_links.csv"
METADATA = "data/raw/mimic-cxr/mimic-cxr-dataset/metadata.csv"
LABEVENTS = "data/raw/mimic-iv/labevents.csv"
OUTPUT = "data/processed/final_dataset.csv"

IMAGES_ROOT = "data/raw/mimic-cxr/mimic-cxr-dataset/official_data_iccv_final/files"
REPORTS_ROOT = "data/raw/mimic-cxr/mimic-cxr-dataset/mimic-cxr-reports/files"

TIER1 = {"wbc": 51301, "rbc": 51279, "hemoglobin": 51222, "hematocrit": 51221,
         "platelets": 51265, "neutrophils": 51256, "lymphocytes": 51244, "monocytes": 51254}
TIER2 = {"crp": 50889, "esr": 51288, "albumin": 50862, "ldh": 50954}
TIER3 = {"sodium": 50983, "potassium": 50971, "glucose": 50931, "creatinine": 50912, "urea": 51006}
ALL_TESTS = {**TIER1, **TIER2, **TIER3}

con = duckdb.connect()

con.execute(f"""
    CREATE TEMP TABLE samples AS
    SELECT f.subject_id, f.study_id, f.hadm_id, f.final_label, r.study_datetime
    FROM read_csv_auto('{FINAL_LABELS}') f
    JOIN read_csv_auto('{RESOLVED_LINKS}') r ON f.study_id = r.study_id
    WHERE f.final_label != 'Excluded'
""")
n_samples = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
print(f"Samples to assemble: {n_samples}")

# pick one primary image per study: prefer PA > AP > LATERAL > LL > anything else
con.execute(f"""
    CREATE TEMP TABLE primary_image AS
    SELECT subject_id, study_id, dicom_id, ViewPosition FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY study_id
                ORDER BY CASE ViewPosition
                    WHEN 'PA' THEN 1 WHEN 'AP' THEN 2
                    WHEN 'LATERAL' THEN 3 WHEN 'LL' THEN 4
                    ELSE 5 END
            ) AS rn
        FROM read_csv_auto('{METADATA}')
        WHERE study_id IN (SELECT study_id FROM samples)
    )
    WHERE rn = 1
""")

itemid_to_name = {v: k for k, v in ALL_TESTS.items()}
itemid_list = ",".join(str(v) for v in ALL_TESTS.values())

con.execute(f"""
    CREATE TEMP TABLE labs AS
    SELECT hadm_id, itemid, charttime, valuenum
    FROM read_csv_auto('{LABEVENTS}')
    WHERE hadm_id IN (SELECT DISTINCT hadm_id FROM samples)
      AND itemid IN ({itemid_list})
      AND valuenum IS NOT NULL
""")
print(f"Relevant lab rows pulled: {con.execute('SELECT COUNT(*) FROM labs').fetchone()[0]}")

# closest-to-study_datetime value per (study, test) -- join labs to each sample's study_datetime per hadm_id
con.execute("""
    CREATE TEMP TABLE labs_ranked AS
    SELECT s.study_id, l.itemid, l.valuenum,
        ROW_NUMBER() OVER (
            PARTITION BY s.study_id, l.itemid
            ORDER BY ABS(EPOCH(l.charttime) - EPOCH(s.study_datetime))
        ) AS rn
    FROM samples s
    JOIN labs l ON s.hadm_id = l.hadm_id
""")

pivot_cols = ", ".join(
    f"MAX(CASE WHEN itemid = {itemid} THEN valuenum END) AS {name}"
    for name, itemid in ALL_TESTS.items()
)
con.execute(f"""
    CREATE TEMP TABLE labs_wide AS
    SELECT study_id, {pivot_cols}
    FROM labs_ranked
    WHERE rn = 1
    GROUP BY study_id
""")

con.execute(f"""
    CREATE TEMP TABLE final_table AS
    SELECT
        s.subject_id, s.study_id, s.hadm_id, s.final_label,
        printf('{IMAGES_ROOT}/p%s/p%s/s%s/%s.jpg',
            substr(CAST(s.subject_id AS VARCHAR), 1, 2), CAST(s.subject_id AS VARCHAR),
            CAST(s.study_id AS VARCHAR), pi.dicom_id) AS image_path,
        printf('{REPORTS_ROOT}/p%s/p%s/s%s.txt',
            substr(CAST(s.subject_id AS VARCHAR), 1, 2), CAST(s.subject_id AS VARCHAR),
            CAST(s.study_id AS VARCHAR)) AS report_path,
        lw.* EXCLUDE (study_id)
    FROM samples s
    LEFT JOIN primary_image pi ON s.study_id = pi.study_id
    LEFT JOIN labs_wide lw ON s.study_id = lw.study_id
""")

missing_image = con.execute("SELECT COUNT(*) FROM final_table WHERE image_path LIKE '%None%' OR image_path IS NULL").fetchone()[0]
print(f"Samples missing a primary image: {missing_image}")

dist = con.execute("SELECT final_label, COUNT(*) FROM final_table GROUP BY 1 ORDER BY 2 DESC").df()
print("\n=== Final dataset label distribution ===")
print(dist)

con.execute(f"COPY final_table TO '{OUTPUT}' (HEADER, DELIMITER ',')")
print(f"\nSaved to {OUTPUT}")
