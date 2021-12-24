from typing import Dict, Text
import os

import tensorflow as tf
import tensorflow_recommenders as tfrs


class TwoTowerCF(tfrs.Model):

    def __init__(
            self,
            user_ids: tf.keras.Model,
            product_ids: tf.keras.Model,
    ):
        super().__init__()
        self.embedding_dimension = 32
        self.user_model = self._set_user_model(user_ids)
        self.product_model = self._set_user_model(product_ids)
        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=None
        )

    def _set_user_model(self, user_ids):
        inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.int32, name="user_id")
        lookup = tf.keras.layers.IntegerLookup(
            vocabulary=tf.convert_to_tensor(user_ids),
            mask_token=None
        )
        embedding = tf.keras.layers.Embedding(len(user_ids) + 1, self.embedding_dimension)
        outputs = embedding(lookup(inputs))
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    def _set_product_model(self, product_ids):
        inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.int32, name="product_id")
        lookup = tf.keras.layers.IntegerLookup(
            vocabulary=tf.convert_to_tensor(product_ids),
            mask_token=None
        )
        embedding = tf.keras.layers.Embedding(len(product_ids) + 1, self.embedding_dimension)
        outputs = embedding(lookup(inputs))

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compute_loss(
            self,
            features: Dict[Text, tf.Tensor],
            training=False
    ) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        product_embeddings = self.product_model(features["product_id"])
        compute_metrics = False if training else True
        loss = self.retrieval_task.call(
            user_embeddings,
            product_embeddings,
            compute_metrics=compute_metrics
        )
        return loss


class QueryModel(tf.keras.Model):

    def __init__(self, user_ids: list):
        super().__init__()
        self.user_ids = user_ids
        self.model = self._set_model()

    def _set_model(self):
        user_input = tf.keras.Input(shape=(), dtype=tf.int32, name="user_id")
        user_lookup = tf.keras.layers.IntegerLookup(vocabulary=self.user_ids)
        user_emb = tf.keras.layers.Embedding(len(self.user_ids) + 1, 32)
        user_output = user_emb(user_lookup(user_input))

        dow_input = tf.keras.Input(shape=(), dtype=tf.int32, name="dow")
        # dow_lookup = tf.keras.layers.IntegerLookup(vocabulary=[k for k in range(7)])
        # dow_emb = tf.keras.layers.Embedding(8, 4)
        dow_onehot = tf.keras.layers.CategoryEncoding(num_tokens=7, output_mode="one_hot")
        # dow_output = dow_emb(dow_lookup(dow_input))
        dow_output = dow_onehot(dow_input)

        hod_input = tf.keras.Input(shape=(), dtype=tf.int32, name="hod")
        hod_lookup = tf.keras.layers.IntegerLookup(vocabulary=[k for k in range(24)])
        hod_emb = tf.keras.layers.Embedding(24, 4)
        hod_output = hod_emb(hod_lookup(hod_input))

        output = tf.concat([
            user_output,
            dow_output,
            hod_output,
        ], axis=1)
        return tf.keras.Model(inputs=[user_input, dow_input, hod_input], outputs=output)

    def call(self, inputs: Dict[Text, tf.Tensor]):
        return self.model(inputs)


class CandidateModel(tf.keras.Model):

    def __init__(
            self,
            gtin_ids: list,
            category_ids: list,
            brand_ids: list,
            age_mean: float = None,
            age_var: float = None
    ):
        super().__init__()
        self.gtin_ids = [str(k) for k in gtin_ids]
        self.category_ids = category_ids
        self.brand_ids = brand_ids
        self.age_mean = age_mean
        self.age_var = age_var
        self.model = self._set_model()

    def _set_model(self):
        gtin_input = tf.keras.Input(shape=(), dtype=tf.string, name="gtin_id")
        gtin_lookup = tf.keras.layers.StringLookup(vocabulary=self.gtin_ids)
        gtin_emb = tf.keras.layers.Embedding(len(self.gtin_ids) + 1, 32)

        cat_input = tf.keras.Input(shape=(), dtype=tf.int32, name="category_id")
        cat_lookup = tf.keras.layers.IntegerLookup(vocabulary=self.category_ids)
        cat_emb = tf.keras.layers.Embedding(len(self.category_ids) + 1, 6)

        brand_input = tf.keras.Input(shape=(), dtype=tf.int32, name="brand_id")
        brand_lookup = tf.keras.layers.IntegerLookup(vocabulary=self.brand_ids)
        brand_emb = tf.keras.layers.Embedding(len(self.brand_ids) + 1, 10)

        # name_input = tf.keras.Input(shape=(), dtype=tf.string, name="product_name")
        # name_tokenizer = tf.keras.layers.TextVectorization()
        # name_emb = tf.keras.layers.Embedding(input_dim=10_000, output_dim=32, mask_zero=True)
        # name_pooling = tf.keras.layers.GlobalAveragePooling1D()

        # age_input = tf.keras.Input(shape=(), dtype=tf.int32, name="product_age")
        # age_norm = tf.keras.layers.Normalization(mean=self.age_mean, variance=self.age_var)
        # tf.reshape(self.normalized_age(inputs["age"]), (-1, 1))

        output = tf.concat([
            gtin_emb(gtin_lookup(gtin_input)),
            cat_emb(cat_lookup(cat_input)),
            brand_emb(brand_lookup(brand_input)),
        ], axis=1)

        model = tf.keras.Model(
            inputs=[
                gtin_input,
                cat_input,
                brand_input,
            ],
            outputs=output
        )
        return model

    def call(self, inputs: Dict[Text, tf.Tensor]):
        return self.model(inputs)


class TwoTowerCFContext(tfrs.models.Model):

    def __init__(
            self,
            query_model: tf.keras.Model,
            candidate_model: tf.keras.Model,
    ):
        super().__init__()
        query_input = query_model.model.inputs
        query_dense = tf.keras.layers.Dense(32)
        query_output = query_dense(query_model.model(query_input))
        self.query_model = tf.keras.Model(
            inputs=query_input,
            outputs=query_output
        )

        self.candidate_model = tf.keras.Model(
            inputs=candidate_model.model.inputs,
            outputs=tf.keras.layers.Dense(32)(candidate_model.model(candidate_model.model.inputs))
        )
        self.task = tfrs.tasks.Retrieval()

    @classmethod
    def from_parameters(
            cls,
            user_ids: list,
            gtin_ids: list,
            category_ids: list,
            brand_ids: list,
            age_mean: float = None,
            age_var: float = None,
    ):
        query_model = QueryModel(user_ids)
        candidate_model = CandidateModel(gtin_ids, category_ids, brand_ids, age_mean, age_var)
        return cls(query_model, candidate_model)

    def compute_loss(
            self,
            features: Dict[Text, tf.Tensor],
            training=False,
    ):
        query_embeddings = self.query_model({
            "user_id": features["user_id"],
            "dow": features["dow"],
            "hod": features["hod"],
        })
        candidate_embeddings = self.candidate_model({
            "gtin_id": tf.strings.as_string(features["gtin_id"]),
            "category_id": features["category_id"],
            "brand_id": features["brand_id"],
            # "age": features["age"],
        })
        compute_metrics = False if training else True
        loss = self.task.call(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            compute_metrics=compute_metrics,
        )
        return loss
