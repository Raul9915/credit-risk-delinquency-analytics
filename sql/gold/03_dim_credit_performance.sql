-- One row per bureau credit: observation window and first month reaching default.
-- Only months in an open status (0-5) count as "at risk"; closed months do not.
SELECT
    SK_ID_BUREAU,
    ANY_VALUE(SK_ID_CURR)        AS SK_ID_CURR,
    ANY_VALUE(CREDIT_TYPE_GROUP) AS CREDIT_TYPE_GROUP,
    ANY_VALUE(VINTAGE_COHORT)    AS VINTAGE_COHORT,
    MIN(MONTHS_ON_BOOK)          AS FIRST_OBSERVED_MOB,
    MAX(MONTHS_ON_BOOK)          AS LAST_OBSERVED_MOB,
    MIN(MONTHS_ON_BOOK) FILTER (WHERE STATUS_ORDER >= ${default_status_min}) AS FIRST_DEFAULT_MOB,
    MAX(STATUS_ORDER)            AS WORST_STATUS_ORDER
FROM gold_fct_bureau_monthly_status
WHERE STATUS_ORDER >= 0
GROUP BY SK_ID_BUREAU
