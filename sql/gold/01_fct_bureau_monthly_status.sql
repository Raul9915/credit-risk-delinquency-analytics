-- One row per bureau credit x month with a known status.
-- MONTHS_ON_BOOK = months since the credit was opened (DAYS_CREDIT / 30.4375).
-- VINTAGE_COHORT = how long before the application the credit was opened
-- (the dataset has no calendar dates, so cohorts are relative).
WITH base AS (
    SELECT
        bb.SK_ID_BUREAU,
        b.SK_ID_CURR,
        b.CREDIT_TYPE_GROUP,
        bb.MONTHS_BALANCE,
        bb.STATUS,
        bb.STATUS_ORDER,
        bb.STATUS_LABEL,
        bb.MONTHS_BALANCE - CAST(ROUND(b.DAYS_CREDIT / 30.4375) AS INTEGER) AS MONTHS_ON_BOOK,
        FLOOR(-b.DAYS_CREDIT / 365.25)                                     AS YEARS_BEFORE_APP
    FROM silver_bureau_balance bb
    JOIN silver_bureau b USING (SK_ID_BUREAU)
    WHERE bb.STATUS_ORDER IS NOT NULL          -- drop 'X' (unknown)
)
SELECT
    * EXCLUDE (YEARS_BEFORE_APP),
    CASE
        WHEN YEARS_BEFORE_APP < 1 THEN '0-1y before app'
        WHEN YEARS_BEFORE_APP < 2 THEN '1-2y before app'
        WHEN YEARS_BEFORE_APP < 3 THEN '2-3y before app'
        WHEN YEARS_BEFORE_APP < 5 THEN '3-5y before app'
        ELSE '5y+ before app'
    END AS VINTAGE_COHORT
FROM base
WHERE MONTHS_ON_BOOK >= 0
