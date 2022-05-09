import logging
import os
from pathlib import Path

import tensorflow as tf
from google.cloud import storage
from tensorflow.data import Dataset

from recsys.config import PROJECT_ID


def gcs_uri_to_fuse_path(
        gcs_uri: str, gcs_prefix: str = "gs://", fuse_prefix: str = "/gcs/"
) -> Path:
    """Change "gs://" to "/gcs/" in GCS URI to support FUSE in Vertex AI. See:
    https://cloud.google.com/vertex-ai/docs/training/code-requirements#fuse

    Args:
        gcs_uri (str): GCS URI
        gcs_prefix (str): expected GCS prefix
        fuse_prefix (str): expected FUSE prefix

    Returns:
        Path: FUSE compatible path
    """
    if gcs_uri.startswith(gcs_prefix):
        return Path(fuse_prefix + gcs_uri[len(gcs_prefix):])
    return Path(gcs_uri)



def process_gcs_uri(uri: str) -> (str, str, str, str):
    '''
    Receives a Google Cloud Storage (GCS) uri and breaks it down to the scheme, bucket, path and file

            Parameters:
                    uri (str): GCS uri

            Returns:
                    scheme (str): uri scheme
                    bucket (str): uri bucket
                    path (str): uri path
                    file (str): uri file
    '''
    url_arr = uri.split("/")
    if "." not in url_arr[-1]:
        file = ""
    else:
        file = url_arr.pop()
    scheme = url_arr[0]
    bucket = url_arr[2]
    path = "/".join(url_arr[3:])
    path = path[:-1] if path.endswith("/") else path

    return scheme, bucket, path, file


def download_file_from_gcs(source_file: str, output_folder: str):
    scheme, bucket, path, file = process_gcs_uri(source_file)
    if scheme != "gs:":
        raise ValueError("URI scheme must be gs")

    # Upload the model to GCS
    b = storage.Client(project=PROJECT_ID).bucket(bucket)
    blob = b.blob(os.path.join(path, file))
    output_file = os.path.join(output_folder, file)
    blob.download_to_filename(output_file)

    return output_file


def create_dataset(
        input_data: str, model_params: dict, file_pattern: str = "", prefetch: bool = True
) -> Dataset:
    """Create a TF Dataset from input csv files.

    Args:
        input_data (Input[Dataset]): Train/Valid data in CSV format
        label_name (str): Name of column containing the labels
        model_params (dict): model parameters
        file_pattern (str): Read data from one or more files. If empty, then
            training and validation data is read from single file respectively.
            For multiple files, use a pattern e.g. "files-*.csv".
    Returns:
        dataset (TF Dataset): TF dataset where each element is a (features, labels)
            tuple that corresponds to a batch of CSV rows
    """

    # shuffle & shuffle_buffer_size added to rearrange input data
    # passed into model training
    # num_rows_for_inference is for auto detection of datatypes
    # while creating the dataset.
    # If a float column has a high proportion of integer values (0/1 etc),
    # the method wrongly detects it as a tf.int32 which fails during training time,
    # hence the high hardcoded value (default is 100)

    if file_pattern:
        input_data = os.path.join(input_data, file_pattern)

    logging.info(f"Creating dataset from CSV file(s) at {input_data}...")

    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=str(input_data),
        batch_size=model_params["batch_size"],
        num_epochs=1,
        column_defaults=[tf.int32, tf.string],
        shuffle=False,
        shuffle_buffer_size=1000,
        num_rows_for_inference=20000,
        num_parallel_reads=tf.data.AUTOTUNE,
        sloppy=True,
    )
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
