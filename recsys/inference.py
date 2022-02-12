import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
import pandas as pd


def brute_force(model, item_ds):
    index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
    index.index_from_dataset(
        tf.data.Dataset.zip((item_ds.map(lambda x: x["item_id"]).batch(100), item_ds.batch(100).map(model.candidate_model)))
    )

    return index


def scann(model, item_ds):
    scann_index = tfrs.layers.factorized_top_k.ScaNN(model.query_model)
    scann_index.index_from_dataset(
        tf.data.Dataset.zip(
            (item_ds.map(lambda x: x["item_id"]).batch(100), item_ds.batch(100).map(model.candidate_model)))
    )
    return scann_index


def inference(model, item_ds, user_ids):
    step = 1000
    len_queries = len(user_ids)
    partial_result = []
    index = brute_force(model, item_ds)
    for k in range(0, len_queries, step):
        rec_score, rec_products = index.call(tf.constant([user_ids[k:k + step]]))
        partial_result.append([rec_score.numpy(), rec_products.numpy()])

    recommendation = np.concatenate([k[1] for k in partial_result], axis=1).squeeze()
    recommendation = pd.DataFrame(recommendation, index=user_ids).stack().reset_index()
    recommendation.columns = ["user_id", "rank", "item_id"]
    recommendation['item_id'] = recommendation['item_id'].apply(lambda x: x.decode())

    return recommendation


def inference_with_exclusion(model, products_ds, user_ids, items_exclusion_list=None):
    if not items_exclusion_list:
        items_exclusion_list = []

    step = 1000
    len_queries = len(user_ids)
    partial_result = []
    index = brute_force(model, products_ds)
    for k in range(0, len_queries, step):
        rec_score, rec_products = index.query_with_exclusions(
            tf.constant([user_ids[k:k + step]]),
            exclusions=tf.constant([items_exclusion_list]))
        partial_result.append([rec_score.numpy(), rec_products.numpy()])

    recommendation = np.concatenate([k[1] for k in partial_result], axis=1).squeeze()
    recommendation = pd.DataFrame(recommendation, index=user_ids).stack().reset_index()
    recommendation.columns = ["user_id", "rank", "product_id"]
    recommendation['product_id'] = recommendation['product_id'].apply(lambda x: x.decode())

    return recommendation
