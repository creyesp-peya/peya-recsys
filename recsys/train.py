import argparse
import json
import logging
import os
os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "3"  # NOQA: E402
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # NOQA: E402

import dask.dataframe as dd
import tensorflow as tf

from recsys import export_model
from recsys.data import create_dataset, gcs_uri_to_fuse_path
from recsys.inference import brute_force, scann
from recsys.model import build_and_compile_model
from recsys.utils import get_distribution_strategy, is_chief


def train_tensorflow_model(
        training_data: str,
        validation_data: str,
        file_pattern: str,
        model_params: dict,
        model_dir: str,
):
    """Train a Tensorflow Keras model.

    Args:
        training_data (str): Training data as a csv file in GCS
        validation_data (str): Validation data as a csv file in GCS
        file_pattern (str): Read data from one or more files. If empty, then
            training and validation data is read from single file respectively.
            For multiple files, use a pattern e.g. "files-*.csv".
        model_params (dict): Dictionary of following training parameters
            batch_size: int (defaults to 100)
            epochs: int (defaults to 5)
            learning rate: float (defaults to 0.001)
            hidden_units: list(tuple) (defaults to [(10, 'relu')])
                Example - [(64, "relu"), (32, "elu")]
                creates 1st dense layer with 64 nodes & activation function as relu
                & 2nd dense layer with 32 nodes & activation function as elu
                Reference (activation functions):
                    https://www.tensorflow.org/api_docs/python/tf/keras/activations
            loss_fn: str (defaults to MeanSquaredError)
                Regression:
                    MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
                Classification:
                    BinaryCrossentropy,
                    CategoricalCrossentropy, SparseCategoricalCrossentropy
                Reference:
                    https://www.tensorflow.org/api_docs/python/tf/keras/losses
            optimizer: str (defaults to Adam)
                Supported values:
                    Adam, Adadelta, Adamax, Adagrad,
                    Ftrl, RMSprop, SGD
                Reference:
                    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
            metrics: list (defaults to ['Accuracy'])
                Reference:
                    https://www.tensorflow.org/api_docs/python/tf/keras/metrics
            distribute strategy: str (defaults to single)
                Supported values:
                    Without GPU: single
                    With GPU: single, mirror, multi
            early_stopping_epochs: int (defaults to 5)
                No of epochs to check for training loss convergence
        model_dir (str): output directory used to save model
    """
    devices = tf.config.list_physical_devices()
    logging.info(f"The current training job is using the following physical devices: {devices}")
    # prepare model params
    default_model_params = dict(
        batch_size=1024 * 1,
        epochs=1,
        steps_per_epoch=10,
        loss_fn="MeanSquaredError",
        optimizer="Adam",
        learning_rate=0.001,
        metrics=["Accuracy"],
        hidden_units=[(10, "relu")],
        distribute_strategy="single",
        early_stopping_epochs=5,
        embedding_dimension=128,
    )
    # merge dictionaries by overwriting default_model_params if provided in model_params
    model_params = {**default_model_params, **model_params}
    logging.info(f"Using model hyper-parameters: {model_params}")

    train_ds = create_dataset(training_data, model_params, file_pattern)
    valid_ds = create_dataset(validation_data, model_params, file_pattern)

    if file_pattern:
        input_data = os.path.join(training_data, file_pattern)
    else:
        input_data = training_data
    train_dd = dd.read_csv(input_data, blocksize=25e6, dtype={"item_id": str, "user_id": int})
    user_ids = train_dd["user_id"].unique().compute().to_list()
    item_ids = train_dd["item_id"].unique().compute().to_list()

    item_ds = tf.data.Dataset.from_tensor_slices(item_ids)

    train_features = list(train_ds.element_spec.keys())
    valid_features = list(valid_ds.element_spec.keys())
    logging.info(f"Training feature names: {train_features}")
    logging.info(f"Validation feature names: {valid_features}")

    if len(train_features) != len(valid_features):
        raise RuntimeError(f"No. of training features != # validation features")

    strategy = get_distribution_strategy(model_params["distribute_strategy"])
    with strategy.scope():
        tf_model = build_and_compile_model(user_ids, item_ids, model_params)

    logging.info("Use early stopping")
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss", mode="min", patience=model_params["early_stopping_epochs"]
    )

    logging.info("Fit model...")
    history = tf_model.fit(
        train_ds,
        batch_size=model_params["batch_size"],
        steps_per_epoch=model_params["steps_per_epoch"],
        epochs=model_params["epochs"],
        validation_data=valid_ds,
        callbacks=[callback],
    )

    # only persist output files if current worker is chief
    if is_chief(strategy):
        # ensure to change GCS to local mount path
        model_dir = gcs_uri_to_fuse_path(model_dir)
        model_dir.mkdir(exist_ok=True, parents=True)

        query_model_path = os.path.join(model_dir, "query_model")
        candidate_model_path = os.path.join(model_dir, "candidate_model")
        index_scann_model_path = os.path.join(model_dir, "index_scann_model")
        index_bruteforce_model_path = os.path.join(model_dir, "index_bruteforce_model")

        export_model.export_tf_model(query_model_path, tf_model.query_model)
        export_model.export_tf_model(candidate_model_path, tf_model.candidate_model)
        export_model.export_index(
            index_scann_model_path,
            scann(tf_model.query_model, tf_model.candidate_model, item_ds),
            use_scann=True
        )
        export_model.export_index(
            index_bruteforce_model_path,
            brute_force(tf_model.query_model, tf_model.candidate_model, item_ds),
            use_scann=False
        )

        metrics_path = model_dir.parent / "train_metrics.json"
        logging.info(f"Save metrics to: {metrics_path}")
        with open(metrics_path, "w") as fp:
            json.dump(history.history, fp)


def create_arg_parser() -> argparse.ArgumentParser:
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument(
        "--training_data",
        type=str,
        required=True,
        help="Path to file or folder containing training file(s) in CSV format.",
    )
    parser_arg.add_argument(
        "--validation_data",
        type=str,
        required=True,
        help="Path to file or folder containing validation file(s) in CSV format.",
    )
    parser_arg.add_argument(
        "--file_pattern",
        type=str,
        required=False,
        help="If train./valid. are folders, specify the file pattern to filter files "
             "e.g. 'files-*.csv'.",
    )
    parser_arg.add_argument(
        "--model_dir", required=False, type=str, default=os.environ.get("AIP_MODEL_DIR")
    )
    parser_arg.add_argument(
        "--model_params",
        type=json.loads,
        required=False,
        default={},
        help="""Hyper-parameters for model training in JSON format containing:
        'batch_size' (int): Batch size defaulting to 100,
        'epochs'(int): Number of epochs defaulting to 5,
        'learning_rate' (float): Learning rate defaulting to 0.001,
        'hidden_units' (list(tuple)):
            List of tuples containing (# hidden units, activation)
            defaulting to [(10, 'relu')]. Example: [(64, 'relu'), (32, 'elu')]
            creates 1st dense layer containing 64 hidden units with relu activation
            & 2nd dense layer containing 32 hidden units with elu action
            (see https://www.tensorflow.org/api_docs/python/tf/keras/activations)
        'loss_fn' (str): Loss function defaulting to 'MeanSquaredError'.
            Regression losses: 'MeanSquaredError', 'MeanAbsoluteError',
            'MeanAbsolutePercentageError'.
            Classification losses: 'BinaryCrossentropy', 'CategoricalCrossentropy',
            'SparseCategoricalCrossentropy'.
            (see https://www.tensorflow.org/api_docs/python/tf/keras/losses),
        'optimizer' (str): Optimizer defaults to 'Adam'. Supported:
            'Adam', 'Adadelta', 'Adamax', 'Adagrad', 'Ftrl', 'RMSprop', 'SGD'
            (see https://www.tensorflow.org/api_docs/python/tf/keras/optimizers,
        'metrics' (list): List of metrics defaulting to ['Accuracy'].
            (see https://www.tensorflow.org/api_docs/python/tf/keras/metrics),
        'distribute_strategy' (str): Distribution strategy defaulting to 'single'.
            Supported: 'single', 'mirror', 'multi'.
        'early_stopping_epochs' (int): Number of epochs to check for training loss
            convergence defaulting to 5.""",
    )
    return parser_arg


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    if args.model_dir is None:
        exit(parser.print_usage())

    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"arg dataset {args.training_data}...")
    train_tensorflow_model(
        training_data=args.training_data,
        validation_data=args.validation_data,
        file_pattern=args.file_pattern,
        model_params=args.model_params,
        model_dir=args.model_dir,
    )
