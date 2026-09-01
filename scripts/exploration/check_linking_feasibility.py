"""
Tests whether MIMIC-CXR studies can be linked to MIMIC-IV admissions via:
  subject_id + study_datetime  -->  admissions[admittime, dischtime]  -->  hadm_id
Then checks hadm_id coverage in labevents.csv and diagnoses_icd.csv.
"""
import duckdb

CXR_METADATA = "data/raw/mimic-cxr/mimic-cxr-dataset/metadata.csv"
ADMISSIONS = "data/raw/mimic-iv/admissions.csv"
LABEVENTS = "data/raw/mimic-iv/labevents.csv"
DIAGNOSES = "data/raw/mimic-iv/diagnoses_icd.csv"

con = duckdb.connect()

total_raw_studies = con.execute(f"""
    SELECT COUNT(DISTINCT study_id) FROM read_csv_auto('{CXR_METADATA}')
""").fetchone()[0]

con.execute(f"""
    CREATE TEMP TABLE studies AS
    SELECT DISTINCT
        subject_id,
        study_id,
        make_timestamp(
            CAST(StudyDate / 10000 AS INT),
            CAST((StudyDate % 10000) / 100 AS INT),
            CAST(StudyDate % 100 AS INT),
            CAST(FLOOR(StudyTime / 10000) AS INT),
            CAST(FLOOR(StudyTime % 10000 / 100) AS INT),
            StudyTime % 100
        ) AS study_datetime
    FROM read_csv_auto('{CXR_METADATA}', types={{'StudyDate': 'BIGINT', 'StudyTime': 'DOUBLE'}})
    -- a handful of rows have malformed StudyDate/StudyTime (column-shift artifacts); exclude before parsing
    WHERE CAST((StudyDate % 10000) / 100 AS INT) BETWEEN 1 AND 12
      AND CAST(StudyDate % 100 AS INT) BETWEEN 1 AND 31
      AND CAST(FLOOR(StudyTime / 10000) AS INT) BETWEEN 0 AND 23
      AND CAST(FLOOR(StudyTime % 10000 / 100) AS INT) BETWEEN 0 AND 59
      AND StudyTime % 100 BETWEEN 0 AND 60
""")

kept_studies = con.execute("SELECT COUNT(DISTINCT study_id) FROM studies").fetchone()[0]
dropped_studies = total_raw_studies - kept_studies

con.execute(f"""
    CREATE TEMP TABLE admissions AS
    SELECT subject_id, hadm_id, admittime, dischtime
    FROM read_csv_auto('{ADMISSIONS}')
""")

con.execute("""
    CREATE TEMP TABLE linked AS
    SELECT s.subject_id, s.study_id, s.study_datetime, a.hadm_id, a.admittime, a.dischtime
    FROM studies s
    JOIN admissions a
      ON s.subject_id = a.subject_id
     AND s.study_datetime BETWEEN a.admittime AND a.dischtime
""")

total_studies, total_subjects = con.execute(
    "SELECT COUNT(*), COUNT(DISTINCT subject_id) FROM studies"
).fetchone()

matched_studies, matched_subjects, matched_hadm = con.execute(
    "SELECT COUNT(DISTINCT study_id), COUNT(DISTINCT subject_id), COUNT(DISTINCT hadm_id) FROM linked"
).fetchone()

ambiguous = con.execute("""
    SELECT COUNT(*) FROM (
        SELECT study_id FROM linked GROUP BY study_id HAVING COUNT(*) > 1
    )
""").fetchone()[0]

print("=== Study <-> Admission linking ===")
print(f"Studies dropped due to malformed StudyDate/StudyTime: {dropped_studies} / {total_raw_studies}")
print(f"Total CXR studies (distinct study_id): {total_studies}")
print(f"Total CXR subjects (distinct subject_id): {total_subjects}")
print(f"Studies with >=1 matching admission: {matched_studies} ({100*matched_studies/total_studies:.1f}%)")
print(f"Subjects with >=1 matching admission: {matched_subjects} ({100*matched_subjects/total_subjects:.1f}%)")
print(f"Distinct hadm_id resolved: {matched_hadm}")
print(f"Studies matching MORE than one admission (ambiguous): {ambiguous}")

diag_hadm = con.execute(f"""
    SELECT COUNT(DISTINCT hadm_id) FROM read_csv_auto('{DIAGNOSES}')
""").fetchone()[0]

diag_coverage = con.execute(f"""
    SELECT COUNT(DISTINCT l.hadm_id)
    FROM linked l
    WHERE l.hadm_id IN (SELECT DISTINCT hadm_id FROM read_csv_auto('{DIAGNOSES}'))
""").fetchone()[0]

print("\n=== Diagnosis coverage (diagnoses_icd.csv) ===")
print(f"Linked hadm_ids with >=1 diagnosis row: {diag_coverage} / {matched_hadm} ({100*diag_coverage/matched_hadm:.1f}%)")

lab_coverage = con.execute(f"""
    SELECT COUNT(DISTINCT l.hadm_id)
    FROM linked l
    WHERE l.hadm_id IN (SELECT DISTINCT hadm_id FROM read_csv_auto('{LABEVENTS}') WHERE hadm_id IS NOT NULL)
""").fetchone()[0]

print("\n=== Blood-lab coverage (labevents.csv) ===")
print(f"Linked hadm_ids with >=1 lab row: {lab_coverage} / {matched_hadm} ({100*lab_coverage/matched_hadm:.1f}%)")

con.execute("""
    COPY linked TO 'data/interim/cxr_admission_links.csv' (HEADER, DELIMITER ',')
""")
print("\nSaved matched links to data/interim/cxr_admission_links.csv")
