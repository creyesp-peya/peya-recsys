CREATE OR REPLACE TABLE `peya-food-and-groceries.user_fiorella_dirosario.user_attributes_sep2020_sep2021` AS
    WITH most_freq_attr AS (
        SELECT
            user.id as user_id
             , APPROX_TOP_COUNT(o.city.city_id, 1)[OFFSET(0)].value AS most_frequent_city
             , APPROX_TOP_COUNT(o.application, 1)[OFFSET(0)].value AS most_frequent_platform
        FROM `peya-bi-tools-pro.il_core.fact_orders` o
        WHERE registered_date BETWEEN "2020-09-01" AND "2021-09-30"
          AND o.order_status = "CONFIRMED"
          AND o.country.country_id = 3
        GROUP BY 1
    )
       , users_period AS
        (
            SELECT
                DISTINCT
                user_id
            FROM `peya-food-and-groceries.user_fiorella_dirosario.order_sep2020_sep2021`
        )
    SELECT
        up.user_id
         , freq.most_frequent_city as city_id
         , freq.most_frequent_platform as platform
         , us.segment
         ,last_order_date
         ,days_from_first_order
    FROM users_period up
             LEFT JOIN most_freq_attr freq ON freq.user_id = up.user_id
             LEFT JOIN `peya-growth-and-onboarding.automated_tables_reports.user_segments` us ON us.user_id = up.user_id
        AND DATE(us.date) = "2021-09-30"