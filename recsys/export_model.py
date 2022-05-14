import logging

import tensorflow as tf


def export_index(output_path, index, use_scann=False):
    @tf.function(input_signature=[tf.TensorSpec([None, ], dtype=tf.int32, name="user_id")])
    def signature_default(user_id):
        score, products = index.call(user_id)
        return {
            "scores": score,
            "items": products,
        }

    logging.info(f"Save model to: {output_path}")
    tf.saved_model.save(
        index,
        output_path,
        options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"]) if use_scann else None,
        signatures={
            'serving_default': signature_default,
        }
    )


def export_tf_model(output_path, model):
    tf.keras.models.save_model(
        model,
        output_path,
    )
