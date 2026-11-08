-- Month-over-month delinquency transitions (month m -> m+1) for open credits.
-- Grain: credit type x from-state x to-state ('All' = every credit type).
WITH pairs AS (
    SELECT
        a.CREDIT_TYPE_GROUP,
        a.STATUS_LABEL AS FROM_STATE,
        a.STATUS_ORDER AS FROM_ORDER,
        b.STATUS_LABEL AS TO_STATE,
        b.STATUS_ORDER AS TO_ORDER
    FROM gold_fct_bureau_monthly_status a
    JOIN gold_fct_bureau_monthly_status b
      ON  b.SK_ID_BUREAU   = a.SK_ID_BUREAU
      AND b.MONTHS_BALANCE = a.MONTHS_BALANCE + 1
    WHERE a.STATUS_ORDER >= 0                  -- from-state must be an open credit
),
counts AS (
    SELECT
        COALESCE(CREDIT_TYPE_GROUP, 'All') AS CREDIT_TYPE,
        FROM_STATE, FROM_ORDER, TO_STATE, TO_ORDER,
        COUNT(*) AS N_TRANSITIONS
    FROM pairs
    GROUP BY GROUPING SETS (
        (CREDIT_TYPE_GROUP, FROM_STATE, FROM_ORDER, TO_STATE, TO_ORDER),
        (FROM_STATE, FROM_ORDER, TO_STATE, TO_ORDER)
    )
)
SELECT
    *,
    N_TRANSITIONS / SUM(N_TRANSITIONS) OVER (PARTITION BY CREDIT_TYPE, FROM_STATE) AS PROBABILITY
FROM counts
