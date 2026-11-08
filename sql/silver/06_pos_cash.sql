-- Monthly snapshots of previous POS / cash loans with Home Credit.
SELECT
    CAST(SK_ID_PREV AS BIGINT) AS SK_ID_PREV,
    CAST(SK_ID_CURR AS BIGINT) AS SK_ID_CURR,
    CAST(MONTHS_BALANCE AS INTEGER) AS MONTHS_BALANCE,
    NAME_CONTRACT_STATUS,
    SK_DPD,
    SK_DPD_DEF
FROM bronze_pos_cash
