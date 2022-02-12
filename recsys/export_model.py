import tensorflow as tf


def export_index(output_path, index, signature="concat"):
    @tf.function(input_signature=[tf.TensorSpec([None, ], dtype=tf.int32)])
    def signature_default(input_1):
        score, products = index.call(input_1)
        return {
            "score": score,
            "items": products,
        }

    @tf.function(input_signature=[tf.TensorSpec([None, ], dtype=tf.int32)])
    def signature_concat(input_1):
        score, products = index.call(input_1)
        return tf.transpose(tf.concat([tf.as_string(products), tf.as_string(score)], axis=0))

    tf.saved_model.save(
        index,
        output_path,
        options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"]),
        signatures={
            'serving_default': signature_concat if signature == "concat" else signature_default,
            'serving_base': signature_default,
            'serving_concat': signature_concat,
        }
    )
