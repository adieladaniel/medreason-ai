"""Persists the tie-broken (one admission per study) study<->admission links to disk."""
import duckdb

LINKS = "data/interim/cxr_admission_links.csv"

con = duckdb.connect()
con.execute(f"CREATE TEMP TABLE links AS SELECT * FROM read_csv_auto('{LINKS}')")

con.execute("""
    CREATE TEMP TABLE resolved AS
    SELECT subject_id, study_id, study_datetime, hadm_id, admittime, dischtime FROM (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY study_id
                ORDER BY ABS(EPOCH(study_datetime) - EPOCH(admittime))
            ) AS rn
        FROM links
    )
    WHERE rn = 1
""")

n = con.execute("SELECT COUNT(*) FROM resolved").fetchone()[0]
print(f"Resolved study-level links: {n}")

con.execute("""
    COPY resolved TO 'data/interim/resolved_study_links.csv' (HEADER, DELIMITER ',')
""")
print("Saved to data/interim/resolved_study_links.csv")
