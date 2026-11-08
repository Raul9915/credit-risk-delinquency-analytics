-- Monthly status of each bureau credit. MONTHS_BALANCE is relative to the
-- application date (0 = application month, -1 = one month before, ...).
-- STATUS: C closed, X unknown, 0 no DPD, 1 = 1-30 DPD, 2 = 31-60, 3 = 61-90,
--         4 = 91-120, 5 = 120+ DPD or sold / written off.
SELECT
    CAST(SK_ID_BUREAU AS BIGINT)  AS SK_ID_BUREAU,
    CAST(MONTHS_BALANCE AS INTEGER) AS MONTHS_BALANCE,
    CAST(STATUS AS VARCHAR)       AS STATUS,
    CASE CAST(STATUS AS VARCHAR)
        WHEN '0' THEN 0 WHEN '1' THEN 1 WHEN '2' THEN 2
        WHEN '3' THEN 3 WHEN '4' THEN 4 WHEN '5' THEN 5
        WHEN 'C' THEN -1
        ELSE NULL                 -- 'X' = unknown
    END                           AS STATUS_ORDER,
    CASE CAST(STATUS AS VARCHAR)
        WHEN '0' THEN 'Current'
        WHEN '1' THEN 'DPD 1-30'
        WHEN '2' THEN 'DPD 31-60'
        WHEN '3' THEN 'DPD 61-90'
        WHEN '4' THEN 'DPD 91-120'
        WHEN '5' THEN 'DPD 120+'
        WHEN 'C' THEN 'Closed'
        ELSE 'Unknown'
    END                           AS STATUS_LABEL
FROM bronze_bureau_balance
