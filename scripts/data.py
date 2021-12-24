from google.cloud import bigquery
from google.cloud.bigquery_storage_v1beta1 import BigQueryStorageClient
import numpy as np
import pandas as pd


def markets():
    MINIMUM_PRODUCTS = 5
    MINIMUM_ORDERS = 2
    data_project_id = "peya-food-and-groceries"
    data_dataset_id = "user_fiorella_dirosario"
    data_table_orders = "order_sep2020_sep2021"
    data_table_order_details = "order_details_sep2020_sep2021"
    data_table_users = "attributes_sep2020_sep2021"
    data_table_products = "product_attributes_sep2020_sep2021"

    interaction_query_train = f"""
    DECLARE minimum_products INT64;
    DECLARE minimum_orders INT64;
    
    SET minimum_products = {MINIMUM_PRODUCTS};
    SET minimum_orders = {MINIMUM_ORDERS};
    
    WITH products_by_user AS (
        SELECT
            uo.user_id
          , COUNT(DISTINCT gtin) as cant_products
          , COUNT(DISTINCT uo.order_id) as cant_orders
          , MAX(uo.order_id) as last_order_id
        FROM 
          `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
        JOIN 
          `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
        ON 
          uo.order_id = od.order_id  
        WHERE 
          uo.user_id IS NOT NULL
          AND od.gtin IS NOT NULL
          AND od.has_gtin = 1
        GROUP BY 1
    )
    SELECT DISTINCT
        CAST(uo.user_id AS STRING) AS user_id
      , CAST(od.gtin AS STRING) AS product_id
      --, uo.order_id
      --, uo.timestamp
    FROM 
      `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
    JOIN 
      `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
    ON uo.order_id = od.order_id
    LEFT JOIN products_by_user pbu ON pbu.user_id = uo.user_id
    WHERE uo.user_id IS NOT NULL
      AND od.gtin IS NOT NULL
      AND od.has_gtin = 1
      AND cant_products >= minimum_products
      AND cant_orders >= minimum_orders
      AND uo.order_id != pbu.last_order_id 
    """

    interaction_query_test = f"""
    DECLARE minimum_products INT64;
    DECLARE minimum_orders INT64;
    
    SET minimum_products = {MINIMUM_PRODUCTS};
    SET minimum_orders = {MINIMUM_ORDERS};
    
    WITH products_by_user AS (
        SELECT
            uo.user_id
          , COUNT(DISTINCT gtin) as cant_products
          , COUNT(DISTINCT uo.order_id) as cant_orders
          , MAX(uo.order_id) as last_order_id
        FROM 
          `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
        JOIN 
          `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
        ON 
          uo.order_id = od.order_id  
        WHERE 
          uo.user_id IS NOT NULL
          AND od.gtin IS NOT NULL
          AND od.has_gtin = 1
        GROUP BY 1
    )
    SELECT DISTINCT
        CAST(uo.user_id AS STRING) AS user_id
      , CAST(od.gtin AS STRING) AS product_id
      --, uo.order_id
      --, uo.timestamp
    FROM 
      `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
    JOIN 
      `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
    ON
      uo.order_id = od.order_id
    LEFT JOIN 
      products_by_user pbu 
    ON 
      pbu.user_id = uo.user_id
    WHERE 
      uo.user_id IS NOT NULL
      AND od.gtin IS NOT NULL
      AND od.has_gtin = 1
      AND cant_products >= minimum_products
      AND cant_orders >= minimum_orders
      AND uo.order_id = pbu.last_order_id 
    """

    product_query = f"""
    DECLARE minimum_products INT64;
    DECLARE minimum_orders INT64;
    
    SET minimum_products = {MINIMUM_PRODUCTS};
    SET minimum_orders = {MINIMUM_ORDERS};
    
    
    WITH products_by_user AS (
        SELECT
            uo.user_id
          , COUNT(DISTINCT od.gtin) as cant_products
          , COUNT(DISTINCT uo.order_id) as cant_orders
          , MAX(uo.order_id) as last_order_id
        FROM 
          `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
        JOIN 
          `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
        ON 
          uo.order_id = od.order_id  
        WHERE 
          uo.user_id IS NOT NULL
          AND od.gtin IS NOT NULL
          AND od.has_gtin = 1
        GROUP BY 1
        HAVING 
          cant_products >= minimum_products AND cant_orders >= minimum_orders
    )
    , products AS (
      SELECT DISTINCT
        od.gtin
      FROM
        `{data_project_id}.{data_dataset_id}.{data_table_order_details}` od
      JOIN 
        `{data_project_id}.{data_dataset_id}.{data_table_orders}` uo ON uo.order_id = od.order_id
      JOIN 
        products_by_user pbu ON pbu.user_id = uo.user_id
      WHERE od.gtin IS NOT NULL AND od.has_gtin = 1
    )
    
    SELECT 
        CAST(pa.gtin AS STRING) AS product_id
      , IF(pa.category_id IS NULL, "", CAST(pa.category_id AS STRING))  AS category_id
      , IF(pa.brand_id IS NULL, "", CAST(pa.brand_id AS STRING)) AS brand_id
      , CAST(pa.age AS STRING) AS age
    FROM 
      `{data_project_id}.{data_dataset_id}.{data_table_products}` pa
    JOIN products p ON p.gtin = pa.gtin
    """

    user_query = f"""
    DECLARE minimum_products INT64;
    DECLARE minimum_orders INT64;
    
    SET minimum_products = {MINIMUM_PRODUCTS};
    SET minimum_orders = {MINIMUM_ORDERS};
    
    WITH products_by_user AS (
        SELECT
        uo.user_id
        , COUNT(DISTINCT gtin) as cant_products
        , COUNT(DISTINCT uo.order_id) as cant_orders
        , MAX(uo.order_id) as last_order_id
        FROM 
        `{data_project_id}.{data_dataset_id}.{data_table_orders}` as uo
        JOIN 
        `{data_project_id}.{data_dataset_id}.{data_table_order_details}` as od
        ON uo.order_id = od.order_id  
        WHERE uo.user_id IS NOT NULL
        AND od.gtin IS NOT NULL
        AND od.has_gtin = 1
        GROUP BY 1
    )
    SELECT 
        CAST(ua.user_id AS STRING) AS user_id
      , CAST(ua.city_id AS STRING) AS city_id
      , ua.platform
      , IF(ua.segment IS NULL, "Not set", ua.segment) AS segment
    FROM 
      `{data_project_id}.{data_dataset_id}.{data_table_users}` ua
    LEFT JOIN products_by_user pbu ON pbu.user_id = ua.user_id
    WHERE cant_products >= minimum_products
          AND cant_orders >= minimum_orders
    """

    project_id = "peya-data-analyt-factory-stg"  #@param ["peya-data-analyt-factory-stg", "peya-food-and-groceries", "peya-growth-and-onboarding"]
    client = bigquery.client.Client(project=project_id)
    bq_storage_client = BigQueryStorageClient()

    interactions_train = (
        client.query(interaction_query_train)
            .result()
            .to_arrow(bqstorage_client=bq_storage_client)
            .to_pandas()
    )

    interactions_test = (
        client.query(interaction_query_test)
            .result()
            .to_arrow(bqstorage_client=bq_storage_client)
            .to_pandas()
    )

    users = (
        client.query(user_query)
            .result()
            .to_arrow(bqstorage_client=bq_storage_client)
            .to_pandas()
    )
    products = (
        client.query(product_query)
            .result()
            .to_arrow(bqstorage_client=bq_storage_client)
            .to_pandas()
    )

    user_ids = users["user_id"].unique().tolist()
    product_ids = products["product_id"].unique().tolist()