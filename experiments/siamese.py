# https://medium.com/@prabhnoor0212/siamese-network-keras-31a3a8f37d04
# https://keras.io/examples/vision/siamese_network/
# https://arxiv.org/pdf/1503.03832.pdf

import tensorflow as tf

examples = {
    "product_a": [
        "Hamburguesa Don de Burguesa con Papas Fritas",
        "Hamburguesa con Queso Cheddar y Panceta con Papas Fritas",
        "Pizza Muzzarella",
        "Bondiola a la Pizza al plato con Papas Fritas o mini ensalada de L y T",
    ],
    "product_b": [
        "Hamburguesa Solari Solari con Papas Fritas",
        "Pizza Fugazzetta",
        "Pizza Fugazzetta",
        "Pizza Fugazzetta",
    ],
    "target": [
        1,
        0,
        1,
        0,
    ]
}

text_dataset = tf.data.Dataset.from_tensor_slices(examples)
X = text_dataset.map(lambda x: {"product_a": x["product_a"], "product_b": x["product_b"]}).batch(4).prefetch(tf.data.AUTOTUNE)
y = text_dataset.map(lambda x: x["target"]).batch(4).prefetch(tf.data.AUTOTUNE)
dataset = tf.data.Dataset.zip((X, y))
vocab = text_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices([x["product_a"], x["product_b"]])).batch(4)
next(vocab.take(1).as_numpy_iterator())
next(X.batch(2).take(1).as_numpy_iterator())
next(y.batch(2).take(1).as_numpy_iterator())

MAX_VOCABULARY = 1000
MAX_SEQUENCE_LENGTH = 32

input_a = tf.keras.Input(shape=(None,), dtype=tf.string, name="product_a")
input_b = tf.keras.Input(shape=(None,), dtype=tf.string, name="product_b")

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCABULARY,
    output_mode='int',
    # output_sequence_length=MAX_SEQUENCE_LENGTH,
    split="whitespace",
    standardize="lower_and_strip_punctuation",
    ngrams=1,
    name="text_vectorization",
)
vectorize_layer.adapt(vocab)

embedding = tf.keras.layers.Embedding(
    input_dim=len(vectorize_layer.get_vocabulary()),
    output_dim=64,
    mask_zero=True,  # Use masking to handle the variable sequence lengths
    name="embedding"
)
bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), name="bidirectional")
dense1 = tf.keras.layers.Dense(64, activation="relu", name="dense")
dotted = tf.keras.layers.Dot(axes=1, name="similarity")

output_a = dense1(bidirectional(embedding(vectorize_layer(input_a))))
output_b = dense1(bidirectional(embedding(vectorize_layer(input_b))))
output = dotted([output_a, output_b])


model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.AUC()],
)
model.fit(
    dataset,
    epochs=1,
    # validation_data=test_dataset,
    # validation_steps=30
)
# test_loss, test_acc = model.evaluate(test_dataset)
# sample_text = ('The movie was cool. The animation and the graphics '
#                'were out of this world. I would recommend this movie.')
# predictions = model.predict(np.array([sample_text]))
