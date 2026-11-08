-- Modeling table: one row per applicant (SK_ID_CURR) with TARGET.
-- Every feature describes information available at application time
-- (all history tables are, by construction, prior to the application).
WITH bureau_agg AS (
    SELECT
        SK_ID_CURR,
        COUNT(*)                                                     AS BUR_N_CREDITS,
        COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Active')             AS BUR_N_ACTIVE,
        COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Bad debt')           AS BUR_N_BAD_DEBT,
        MAX(DAYS_CREDIT)                                             AS BUR_DAYS_LAST_CREDIT,
        MIN(DAYS_CREDIT)                                             AS BUR_DAYS_FIRST_CREDIT,
        MAX(CREDIT_DAY_OVERDUE)                                      AS BUR_MAX_DAYS_OVERDUE,
        MAX(AMT_CREDIT_MAX_OVERDUE)                                  AS BUR_MAX_AMT_OVERDUE,
        SUM(AMT_CREDIT_SUM_DEBT)                                     AS BUR_TOTAL_DEBT,
        SUM(AMT_CREDIT_SUM)                                          AS BUR_TOTAL_CREDIT,
        SUM(AMT_CREDIT_SUM_OVERDUE)                                  AS BUR_TOTAL_OVERDUE,
        SUM(AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(AMT_CREDIT_SUM), 0)    AS BUR_DEBT_TO_CREDIT,
        SUM(CNT_CREDIT_PROLONG)                                      AS BUR_N_PROLONGATIONS
    FROM silver_bureau
    GROUP BY SK_ID_CURR
),
bureau_balance_agg AS (
    SELECT
        SK_ID_CURR,
        COUNT(*)                                                     AS BB_N_MONTHS,
        AVG(CAST(STATUS_ORDER > 0 AS DOUBLE))                        AS BB_SHARE_MONTHS_DPD,
        MAX(STATUS_ORDER)                                            AS BB_WORST_STATUS,
        COUNT(*) FILTER (WHERE STATUS_ORDER >= ${default_status_min}) AS BB_N_MONTHS_DEFAULT,
        COUNT(*) FILTER (WHERE STATUS_ORDER > 0 AND MONTHS_BALANCE >= -12) AS BB_N_DPD_LAST_12M
    FROM gold_fct_bureau_monthly_status
    WHERE STATUS_ORDER >= 0
    GROUP BY SK_ID_CURR
),
previous_agg AS (
    SELECT
        SK_ID_CURR,
        COUNT(*)                                                     AS PREV_N_APPLICATIONS,
        AVG(CAST(NAME_CONTRACT_STATUS = 'Approved' AS DOUBLE))       AS PREV_APPROVAL_RATE,
        AVG(CAST(NAME_CONTRACT_STATUS = 'Refused' AS DOUBLE))        AS PREV_REFUSAL_RATE,
        AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))                 AS PREV_CREDIT_TO_APPLICATION,
        MAX(DAYS_DECISION)                                           AS PREV_DAYS_LAST_DECISION
    FROM silver_previous_application
    GROUP BY SK_ID_CURR
),
installments_agg AS (
    SELECT
        SK_ID_CURR,
        COUNT(*)                                                     AS INST_N_PAYMENTS,
        AVG(CAST(DAYS_LATE > 0 AS DOUBLE))                           AS INST_SHARE_LATE,
        AVG(GREATEST(DAYS_LATE, 0))                                  AS INST_AVG_DAYS_LATE,
        MAX(DAYS_LATE)                                               AS INST_MAX_DAYS_LATE,
        AVG(PAYMENT_RATIO)                                           AS INST_AVG_PAYMENT_RATIO,
        AVG(CAST(PAYMENT_RATIO < 1 AS DOUBLE))                       AS INST_SHARE_UNDERPAID
    FROM silver_installments
    GROUP BY SK_ID_CURR
),
pos_agg AS (
    SELECT SK_ID_CURR,
           MAX(SK_DPD)                         AS POS_MAX_DPD,
           AVG(CAST(SK_DPD_DEF > 0 AS DOUBLE)) AS POS_SHARE_MONTHS_DPD_DEF
    FROM silver_pos_cash
    GROUP BY SK_ID_CURR
),
cc_agg AS (
    SELECT SK_ID_CURR,
           AVG(UTILIZATION)                    AS CC_AVG_UTILIZATION,
           MAX(UTILIZATION)                    AS CC_MAX_UTILIZATION,
           MAX(SK_DPD)                         AS CC_MAX_DPD
    FROM silver_credit_card
    GROUP BY SK_ID_CURR
)
SELECT
    a.*,
    -- engineered application ratios
    -a.DAYS_BIRTH / 365.25                             AS AGE_YEARS,
    -a.DAYS_EMPLOYED / 365.25                          AS YEARS_EMPLOYED,
    a.AMT_CREDIT  / NULLIF(a.AMT_INCOME_TOTAL, 0)      AS CREDIT_TO_INCOME,
    a.AMT_ANNUITY / NULLIF(a.AMT_INCOME_TOTAL, 0)      AS ANNUITY_TO_INCOME,
    a.AMT_ANNUITY / NULLIF(a.AMT_CREDIT, 0)            AS ANNUITY_TO_CREDIT,
    a.AMT_CREDIT  / NULLIF(a.AMT_GOODS_PRICE, 0)       AS CREDIT_TO_GOODS,
    (COALESCE(a.EXT_SOURCE_1, 0) + COALESCE(a.EXT_SOURCE_2, 0) + COALESCE(a.EXT_SOURCE_3, 0))
      / NULLIF((a.EXT_SOURCE_1 IS NOT NULL)::INT + (a.EXT_SOURCE_2 IS NOT NULL)::INT
               + (a.EXT_SOURCE_3 IS NOT NULL)::INT, 0) AS EXT_SOURCE_MEAN,
    b.* EXCLUDE (SK_ID_CURR),
    bb.* EXCLUDE (SK_ID_CURR),
    p.* EXCLUDE (SK_ID_CURR),
    i.* EXCLUDE (SK_ID_CURR),
    pos.* EXCLUDE (SK_ID_CURR),
    cc.* EXCLUDE (SK_ID_CURR),
    CAST(b.SK_ID_CURR IS NOT NULL AS INTEGER)          AS HAS_BUREAU_HISTORY
FROM silver_application a
LEFT JOIN bureau_agg         b   USING (SK_ID_CURR)
LEFT JOIN bureau_balance_agg bb  USING (SK_ID_CURR)
LEFT JOIN previous_agg       p   USING (SK_ID_CURR)
LEFT JOIN installments_agg   i   USING (SK_ID_CURR)
LEFT JOIN pos_agg            pos USING (SK_ID_CURR)
LEFT JOIN cc_agg             cc  USING (SK_ID_CURR)
