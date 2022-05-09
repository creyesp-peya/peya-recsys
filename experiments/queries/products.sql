DECLARE minimum_products INT64;
DECLARE minimum_orders INT64;

SET minimum_products = 5;
SET minimum_orders = 2;


WITH products_by_user AS (
    SELECT uo.user_id
         , COUNT(DISTINCT od.gtin)     as cant_products
         , COUNT(DISTINCT uo.order_id) as cant_orders
         , MAX(uo.order_id)            as last_order_id
    FROM `peya-food-and-groceries.user_fiorella_dirosario.order_sep2020_sep2021` as uo
             JOIN
         `peya-food-and-groceries.user_fiorella_dirosario.order_details_sep2020_sep2021` as od
         ON
             uo.order_id = od.order_id
    WHERE uo.user_id IS NOT NULL
      AND od.gtin IS NOT NULL
      AND od.has_gtin = 1
    GROUP BY 1
    HAVING cant_products >= minimum_products
       AND cant_orders >= minimum_orders
)
   , products AS (
    SELECT DISTINCT od.gtin
    FROM `peya-food-and-groceries.user_fiorella_dirosario.order_details_sep2020_sep2021` od
             JOIN
         `peya-food-and-groceries.user_fiorella_dirosario.order_sep2020_sep2021` uo ON uo.order_id = od.order_id
             JOIN
         products_by_user pbu ON pbu.user_id = uo.user_id
    WHERE od.gtin IS NOT NULL
      AND od.has_gtin = 1
)

SELECT CAST(pa.gtin AS STRING)                                        AS product_id
     , IF(pa.category_id IS NULL, "", CAST(pa.category_id AS STRING)) AS category_id
     , IF(pa.brand_id IS NULL, "", CAST(pa.brand_id AS STRING))       AS brand_id
     , CAST(pa.age AS STRING)                                         AS age
FROM `peya-food-and-groceries.user_fiorella_dirosario.product_attributes_sep2020_sep2021` pa
         JOIN products p ON p.gtin = pa.gtin
