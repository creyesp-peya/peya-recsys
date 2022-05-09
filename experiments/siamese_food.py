# https://medium.com/@prabhnoor0212/siamese-network-keras-31a3a8f37d04
# https://keras.io/examples/vision/siamese_network/
# https://arxiv.org/pdf/1503.03832.pdf
import os

import tensorflow as tf

BATCH_SIZE = 1024

products = tf.data.TextLineDataset("../data/product.csv").batch(BATCH_SIZE)

file_path_patter = "../data/pairs/"
files = [file_path_patter + k for k in os.listdir(file_path_patter)]
cutoff = int(len(files) * 0.9)
# train_files = [k for k in files[:cutoff]]
# val_files = [k for k in files[cutoff:]]
train_files = ["../data/pairs/dataset-00000.csv"]
val_files = ["../data/pairs/dataset-00001.csv"]

column_name = {
    "item_a": tf.string,
    "product_name_a": tf.string,
    "product_description_a": tf.string,
    "section_name_a": tf.string,
    "item_b": tf.string,
    "product_name_b": tf.string,
    "product_description_b": tf.string,
    "section_name_b": tf.string,
    "target": tf.string
}
target_name = "target"
train_ds = tf.data.experimental.make_csv_dataset(
    train_files,
    batch_size=BATCH_SIZE,
    column_defaults=column_name.values(),
    select_columns=column_name.keys(),
    label_name=target_name,
    field_delim=";",
    ignore_errors=True
)
train_ds = train_ds.map(lambda x, y: (
    {"product_a": x["product_name_a"], "product_b": x["product_name_b"]},
    tf.strings.to_number(y, out_type=tf.int32)
)).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.experimental.make_csv_dataset(
    val_files,
    batch_size=BATCH_SIZE,
    column_defaults=column_name.values(),
    select_columns=column_name.keys(),
    label_name=target_name,
    field_delim=";",
    ignore_errors=True
)
val_ds = val_ds.map(lambda x, y: (
    {"product_a": x["product_name_a"], "product_b": x["product_name_b"]},
    tf.strings.to_number(y, out_type=tf.int32)
)).prefetch(tf.data.AUTOTUNE)

next(products.take(1).as_numpy_iterator())
next(train_ds.take(1).as_numpy_iterator())

MAX_VOCABULARY = 10000
MAX_SEQUENCE_LENGTH = 32

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCABULARY,
    output_mode='int',
    # output_sequence_length=MAX_SEQUENCE_LENGTH,
    split="whitespace",
    standardize="lower_and_strip_punctuation",
    ngrams=1,
    name="text_vectorization",
)
vectorize_layer.adapt(products)

embedding = tf.keras.layers.Embedding(
    input_dim=len(vectorize_layer.get_vocabulary()),
    output_dim=64,
    mask_zero=True,  # Use masking to handle the variable sequence lengths
    name="embedding"
)
bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), name="bidirectional")
dense1 = tf.keras.layers.Dense(64, activation="relu", name="dense")
dotted = tf.keras.layers.Dot(axes=1, normalize=False, name="similarity")

input_a = tf.keras.Input(shape=(None,), dtype=tf.string, name="product_a")
input_b = tf.keras.Input(shape=(None,), dtype=tf.string, name="product_b")

output_a = dense1(bidirectional(embedding(vectorize_layer(input_a))))
output_b = dense1(bidirectional(embedding(vectorize_layer(input_b))))
output = dotted([output_a, output_b])

model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.AUC()],
)
model.fit(
    train_ds,
    epochs=2,
    steps_per_epoch=100,
    validation_data=val_ds,
    validation_steps=30,
)
# test_loss, test_acc = model.evaluate(test_dataset)
# sample_text = ('The movie was cool. The animation and the graphics '
#                'were out of this world. I would recommend this movie.')
# predictions = model.predict(np.array([sample_text]))
