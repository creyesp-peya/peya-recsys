import os

import tensorflow as tf

from recsys import inference
from recsys.export_model import export_index
from recsys.model import TwoTowerCF


def two_tower_cf():
    users = tf.random.uniform(shape=[1000], minval=1, maxval=10, dtype=tf.int32)
    items = tf.random.uniform(shape=[1000], minval=1, maxval=1000, dtype=tf.int32)
    user_ids = tf.unique(users).y
    product_ids = tf.unique(items).y

    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "user_id": users,
            "item_id": items
        }
    )
    dataset = dataset.batch(8)

    model = TwoTowerCF(user_ids, product_ids, norm_embedding=True)
    model.query_model.summary()
    model.candidate_model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    model.fit(dataset, epochs=1)

    predicted_user = model.query_model.predict(dataset.map(lambda x: {"user_id": x["user_id"]}).take(1))
    print(f"Shape of output user model: {predicted_user.shape}")

    output_path = "models/test/two_tower_cf/"
    query_model_path = os.path.join(output_path, "query_model")
    model.query_model.save(query_model_path)

    product_ds = tf.data.Dataset.from_tensor_slices({"item_id": product_ids})
    index = inference.scann(model, product_ds)
    print(index.call(tf.constant([98]), 5))

    index_model_path = os.path.join(output_path, "index_model", "1")
    export_index(index_model_path, index, signature="concat")

    index_loaded_model = tf.saved_model.load(index_model_path)
    infer_signature = index_loaded_model.signatures['serving_default']
    prediction = infer_signature(tf.constant([133]))
    print(prediction)


if __name__ == "__main__":
    two_tower_cf()
