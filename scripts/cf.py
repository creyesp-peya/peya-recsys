import os

import numpy as np
import tensorflow as tf

from recsys.model import TwoTowerCF


def two_tower_cf():
    data_path = "../data/cf/train_*.csv"

    tf.random.set_seed(42)
    interactions_train_ds = tf.data.experimental.make_csv_dataset(
        data_path,
        batch_size=8192,
        num_epochs=1,
        column_defaults=[tf.int32, tf.string],
        num_parallel_reads=20,
        shuffle_buffer_size=10000
    ).prefetch(tf.data.AUTOTUNE).cache()
    user_ds = interactions_train_ds.map(lambda x: x["user_id"]).unique()
    item_ds = interactions_train_ds.map(lambda x: x["product_id"]).unique()
    user_ids = np.unique(np.concatenate(list(user_ds.as_numpy_iterator()))).tolist()
    item_ids = np.unique(np.concatenate(list(item_ds.as_numpy_iterator()))).astype(str).tolist()

    dataset = interactions_train_ds.map(lambda x: {"user_id": x["user_id"], "item_id": x["product_id"]})

    model = TwoTowerCF(user_ids, item_ids)
    model.query_model.summary()
    model.candidate_model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    model.fit(dataset, epochs=1, steps_per_epoch=1)

    predicted_user = model.query_model.predict(dataset.map(lambda x: {"user_id": x["user_id"]}).take(1))
    print(f"Shape of output user model: {predicted_user.shape}")

    output_path = "models/two_tower_cf/"
    query_model_path = os.path.join(output_path, "query_model")
    model.query_model.save(query_model_path)
    candidate_model_path = os.path.join(output_path, "candidate_model")
    model.candidate_model.save(candidate_model_path)


if __name__ == "__main__":
    two_tower_cf()
