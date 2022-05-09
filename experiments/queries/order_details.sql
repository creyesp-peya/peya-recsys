CREATE OR REPLACE TABLE ` peya-food-and-groceries.user_fiorella_dirosario.order_details_sep2020_sep2021 ` AS
SELECT o.order_id
     , CASE
           WHEN SPLIT(d.integration_code, "-")[SAFE_OFFSET(0)] IN ("D", "P")
               THEN LPAD(SPLIT(d.integration_code, "-")[SAFE_OFFSET(2)], 14, '0')
           WHEN SPLIT(d.integration_code, "-")[SAFE_OFFSET(0)] IN ("S", "G")
               THEN LPAD(SPLIT(d.integration_code, "-")[SAFE_OFFSET(3)], 14, '0')
           ELSE LPAD(dp.gtin, 14, '0')
    END                  AS gtin
     , CASE
           WHEN SPLIT(d.integration_code, "-")[SAFE_OFFSET(0)] IN ("D", "P", "S", "G") THEN 1
           WHEN dp.gtin IS NOT NULL THEN 1
           ELSE 0
    END                  as has_gtin
     , d.integration_code
     , d.product.product_id
     , d.product_name
     , o.restaurant.id   AS partner_id
     , o.restaurant.name AS partner_name
FROM `peya-bi-tools-pro.il_core.fact_orders` o
   , UNNEST(details) d
         LEFT JOIN `peya-bi-tools-pro.il_core.dim_product` dp ON dp.id = d.product.product_id
WHERE registered_date BETWEEN "2020-09-01" AND "2021-09-30"
  AND o.business_type.business_type_id = 2
  AND o.order_status = "CONFIRMED"
  AND o.country.country_id = 3