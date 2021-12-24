import tensorflow as tf
from model import TwoTowerCF, TwoTowerCFContext


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

    model = TwoTowerCF(user_ids, product_ids)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    history = model.fit(dataset, epochs=5)


def two_tower_context():
    users = tf.random.uniform(shape=[100], minval=1, maxval=10, dtype=tf.int32)
    dow = tf.random.uniform(shape=[100], minval=0, maxval=6, dtype=tf.int32)
    hod = tf.random.uniform(shape=[100], minval=0, maxval=23, dtype=tf.int32)
    gtin = tf.random.uniform(shape=[100], minval=10000000, maxval=100000000, dtype=tf.int32)
    category = tf.random.uniform(shape=[100], minval=0, maxval=30, dtype=tf.int32)
    brand = tf.random.uniform(shape=[100], minval=0, maxval=10, dtype=tf.int32)
    user_ids = tf.unique(users).y
    product_ids = tf.unique(gtin).y
    category_ids = tf.unique(category).y
    brand_ids = tf.unique(brand).y

    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "user_id": users,
            "dow": dow,
            "hod": hod,
            "gtin_id": gtin,
            "category_id": category,
            "brand_id": brand,
        }
    )
    dataset = dataset.batch(32)

    model = TwoTowerCFContext.from_parameters(
        user_ids=user_ids,
        gtin_ids=product_ids,
        category_ids=category_ids,
        brand_ids=brand_ids,
    )
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1), run_eagerly=True)
    history = model.fit(dataset, epochs=5)


if __name__ == "__main__":
    # two_tower_cf()
    two_tower_context()
