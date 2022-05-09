CREATE OR REPLACE TABLE `peya-food-and-groceries.user_fiorella_dirosario.product_attributes_sep2020_sep2021` AS
    WITH info AS (
        SELECT
            od.gtin as gtin
             , MIN(dp.date_created) as date_created
             , MIN(brand_id) as brand_id
             , MIN(cb.name) as brand_name
             , MIN(cb.group_id) as brand_group_id
             , MIN(cbg.name) as brand_group_name
             , MIN(category_id) as category_id
             , MIN(cc.name) as category_name
        FROM `peya-food-and-groceries.user_fiorella_dirosario.order_details_sep2020_sep2021` od
                 INNER JOIN `peya-bi-tools-pro.il_core.dim_product` dp ON dp.id = od.product_id
                 LEFT JOIN `peya-data-origins-pro.cl_catalogue.product` cp ON cp.gtin = od.gtin
                 LEFT JOIN `peya-data-origins-pro.cl_catalogue.brand` cb ON cb.id = cp.brand_id
                 LEFT JOIN `peya-data-origins-pro.cl_catalogue.category` cc ON cc.id = cp.category_id
                 LEFT JOIN `peya-data-origins-pro.cl_catalogue.brand_group` cbg ON cbg.id = cb.group_id
        GROUP BY 1
    )
    SELECT
        gtin
         , category_id
         , category_name
         , brand_id
         , brand_name
         , brand_group_id
         , brand_group_name
         , DATE_DIFF("2021-09-30", DATE(date_created), DAY) as age
    FROM info