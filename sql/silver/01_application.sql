-- Applicants with a known outcome (TARGET). One row per SK_ID_CURR.
-- Cleaning rules:
--   * DAYS_EMPLOYED = 365243 is a sentinel (mostly pensioners) -> NULL + flag
--   * CODE_GENDER 'XNA' -> NULL
SELECT
    * EXCLUDE (_source_file, _loaded_at) REPLACE (
        CAST(SK_ID_CURR AS BIGINT)                                   AS SK_ID_CURR,
        CAST(TARGET AS INTEGER)                                      AS TARGET,
        CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL ELSE DAYS_EMPLOYED END AS DAYS_EMPLOYED,
        NULLIF(CODE_GENDER, 'XNA')                                   AS CODE_GENDER
    ),
    CAST(DAYS_EMPLOYED = 365243 AS INTEGER)                          AS FLAG_DAYS_EMPLOYED_ANOM
FROM bronze_application
