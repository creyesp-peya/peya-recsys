import os

import tensorflow as tf

from recsys.model import TwoTowerCF


def two_tower_cf():
    users = tf.random.uniform(shape=[100], minval=1, maxval=10, dtype=tf.int32)
    products = tf.random.uniform(shape=[100], minval=1, maxval=1000, dtype=tf.int32)
    user_ids = tf.unique(users).y
    product_ids = tf.unique(products).y

    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "user_id": users,
            "product_id": products
        }
    )
    dataset = dataset.batch(8)

    model = TwoTowerCF(user_ids, product_ids, norm_embedding=True)
    model.user_model.summary()
    model.product_model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    model.fit(dataset, epochs=1)

    predicted_user = model.user_model.predict(dataset.map(lambda x: {"user_id": x["user_id"]}).take(1))
    print(f"Shape of output user model: {predicted_user.shape}")

    output_path = "models/test/two_tower_cf/"
    query_model_path = os.path.join(output_path, "query_model")
    if not os.path.exists(query_model_path):
        os.makedirs(query_model_path)
    model.user_model.save(query_model_path)


if __name__ == "__main__":
    two_tower_cf()
