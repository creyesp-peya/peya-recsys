class MarketsModel(tfrs.Model):

    def __init__(
            self,
            user_model: tf.keras.Model,
            product_model: tf.keras.Model,
            task: tf.keras.layers.Layer
    ):
        super().__init__()
        self.user_model = user_model
        self.product_model = product_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(
            self,
            features: Dict[Text, tf.Tensor],
            training=False
    ) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        positive_product_embeddings = self.product_model(features["product_id"])
        compute_metrics = False if training else True
        return self.task(user_embeddings, positive_product_embeddings,
                         compute_metrics=compute_metrics)


embedding_dimension = 32

user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=tf.convert_to_tensor(user_ids), mask_token=None),
    tf.keras.layers.Embedding(len(user_ids) + 1, embedding_dimension)
])

product_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=tf.convert_to_tensor(product_ids), mask_token=None),
    tf.keras.layers.Embedding(len(product_ids) + 1, embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
    candidates=products_ds.batch(128).map(product_model)
)

task = tfrs.tasks.Retrieval(
    metrics=metrics
)
