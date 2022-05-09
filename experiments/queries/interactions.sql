CREATE OR REPLACE TABLE ` peya-food-and-groceries.user_fiorella_dirosario.order_sep2020_sep2021 ` AS
SELECT o.user.id       as user_id
     , o.is_pre_order
     , o.pickup
     , o.discount_type
     , o.order_id
     , o.registered_at as timestamp
FROM `peya-bi-tools-pro.il_core.fact_orders` o
WHERE registered_date BETWEEN "2020-09-01" AND "2021-09-30"
  AND o.business_type.business_type_id = 2
  AND o.order_status = "CONFIRMED"
  AND o.country.country_id = 3