DECLARE minimum_products INT64;
DECLARE minimum_orders INT64;

SET minimum_products = 5;
SET minimum_orders = 2;

WITH products_by_user AS (
    SELECT uo.user_id
         , COUNT(DISTINCT gtin)        as cant_products
         , COUNT(DISTINCT uo.order_id) as cant_orders
         , MAX(uo.order_id)            as last_order_id
    FROM `peya-food-and-groceries.user_fiorella_dirosario.order_sep2020_sep2021` as uo
             JOIN
         `peya-food-and-groceries.user_fiorella_dirosario.order_details_sep2020_sep2021` as od
         ON uo.order_id = od.order_id
    WHERE uo.user_id IS NOT NULL
      AND od.gtin IS NOT NULL
      AND od.has_gtin = 1
    GROUP BY 1
)
SELECT CAST(ua.user_id AS STRING)                    AS user_id
     , CAST(ua.city_id AS STRING)                    AS city_id
     , ua.platform
     , IF(ua.segment IS NULL, "Not set", ua.segment) AS segment
FROM `peya-food-and-groceries.user_fiorella_dirosario.attributes_sep2020_sep2021` ua
         LEFT JOIN products_by_user pbu ON pbu.user_id = ua.user_id
WHERE cant_products >= minimum_products
  AND cant_orders >= minimum_orders
