-- Previous credits reported to the Credit Bureau. One row per SK_ID_BUREAU.
SELECT
    CAST(SK_ID_CURR AS BIGINT)   AS SK_ID_CURR,
    CAST(SK_ID_BUREAU AS BIGINT) AS SK_ID_BUREAU,
    CREDIT_ACTIVE,
    CREDIT_TYPE,
    CASE
        WHEN CREDIT_TYPE IN ('Consumer credit', 'Credit card', 'Car loan', 'Mortgage', 'Microloan')
            THEN CREDIT_TYPE
        ELSE 'Other'
    END                          AS CREDIT_TYPE_GROUP,
    DAYS_CREDIT,
    CREDIT_DAY_OVERDUE,
    DAYS_CREDIT_ENDDATE,
    DAYS_ENDDATE_FACT,
    AMT_CREDIT_MAX_OVERDUE,
    CNT_CREDIT_PROLONG,
    AMT_CREDIT_SUM,
    AMT_CREDIT_SUM_DEBT,
    AMT_CREDIT_SUM_LIMIT,
    AMT_CREDIT_SUM_OVERDUE,
    DAYS_CREDIT_UPDATE,
    AMT_ANNUITY
FROM bronze_bureau
