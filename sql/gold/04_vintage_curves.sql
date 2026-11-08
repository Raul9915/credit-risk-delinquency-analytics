-- Censoring-aware vintage curves (Kaplan-Meier style cumulative default rate).
--   hazard(k)  = credits defaulting at month-on-book k / credits at risk at k
--   cum_pd(k)  = 1 - prod_{j<=k} (1 - hazard(j))
-- At risk at k: entered observation at or before k, and neither defaulted nor
-- left observation before k. Handles credits observed for different lengths.
WITH credits AS (
    SELECT VINTAGE_COHORT, CREDIT_TYPE_GROUP AS CREDIT_TYPE, FIRST_OBSERVED_MOB AS ENTRY_K,
           COALESCE(FIRST_DEFAULT_MOB, LAST_OBSERVED_MOB) AS EXIT_K,
           CAST(FIRST_DEFAULT_MOB IS NOT NULL AS INTEGER) AS IS_EVENT
    FROM gold_dim_credit_performance
    UNION ALL
    SELECT VINTAGE_COHORT, 'All', FIRST_OBSERVED_MOB,
           COALESCE(FIRST_DEFAULT_MOB, LAST_OBSERVED_MOB),
           CAST(FIRST_DEFAULT_MOB IS NOT NULL AS INTEGER)
    FROM gold_dim_credit_performance
),
grid AS (
    SELECT g.VINTAGE_COHORT, g.CREDIT_TYPE, r.k AS MONTHS_ON_BOOK
    FROM (SELECT DISTINCT VINTAGE_COHORT, CREDIT_TYPE FROM credits) g
    CROSS JOIN range(0, ${vintage_max_months_on_book} + 1) r(k)
),
entries AS (
    SELECT VINTAGE_COHORT, CREDIT_TYPE, ENTRY_K AS MONTHS_ON_BOOK, COUNT(*) AS N_ENTRIES
    FROM credits GROUP BY ALL
),
exits AS (
    SELECT VINTAGE_COHORT, CREDIT_TYPE, EXIT_K AS MONTHS_ON_BOOK,
           COUNT(*) AS N_EXITS, SUM(IS_EVENT) AS N_EVENTS
    FROM credits GROUP BY ALL
),
joined AS (
    SELECT g.*,
           COALESCE(en.N_ENTRIES, 0) AS N_ENTRIES,
           COALESCE(ex.N_EXITS, 0)   AS N_EXITS,
           COALESCE(ex.N_EVENTS, 0)  AS N_EVENTS
    FROM grid g
    LEFT JOIN entries en USING (VINTAGE_COHORT, CREDIT_TYPE, MONTHS_ON_BOOK)
    LEFT JOIN exits   ex USING (VINTAGE_COHORT, CREDIT_TYPE, MONTHS_ON_BOOK)
),
at_risk AS (
    SELECT *,
           SUM(N_ENTRIES) OVER w_incl
             - COALESCE(SUM(N_EXITS) OVER w_excl, 0) AS N_AT_RISK
    FROM joined
    WINDOW w_incl AS (PARTITION BY VINTAGE_COHORT, CREDIT_TYPE ORDER BY MONTHS_ON_BOOK
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
           w_excl AS (PARTITION BY VINTAGE_COHORT, CREDIT_TYPE ORDER BY MONTHS_ON_BOOK
                      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
),
hazard AS (
    SELECT *, CASE WHEN N_AT_RISK > 0 THEN N_EVENTS / N_AT_RISK ELSE 0 END AS HAZARD
    FROM at_risk
)
SELECT
    VINTAGE_COHORT, CREDIT_TYPE, MONTHS_ON_BOOK, N_AT_RISK, N_EVENTS, HAZARD,
    1 - EXP(SUM(LN(GREATEST(1 - HAZARD, 1e-12))) OVER (
            PARTITION BY VINTAGE_COHORT, CREDIT_TYPE ORDER BY MONTHS_ON_BOOK
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) AS CUMULATIVE_DEFAULT_RATE
FROM hazard
