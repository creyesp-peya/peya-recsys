import logging

import tensorflow as tf


def get_distribution_strategy(distribute_strategy: str) -> tf.distribute.Strategy:
    """Set distribute strategy based on input string.

    Args:
        distribute_strategy (str): single, mirror or multi
    Returns:
        strategy (tf.distribute.Strategy): distribution strategy
    """
    logging.info(f"Distribution strategy: {distribute_strategy}")

    # Single machine, single compute device
    if distribute_strategy == "single":
        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    # Single machine, multiple compute device
    elif distribute_strategy == "mirror":
        strategy = tf.distribute.MirroredStrategy()
    # Multiple machine, multiple compute device
    elif distribute_strategy == "multi":
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        raise RuntimeError(f"Distribute strategy: {distribute_strategy} not supported")
    return strategy


def is_chief(strategy: tf.distribute.Strategy) -> bool:
    """Determine whether current worker is the chief (master). See more info:
    - https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
    - https://www.tensorflow.org/api_docs/python/tf/distribute/cluster_resolver/ClusterResolver # noqa: E501

    Args:
        strategy (tf.distribute.Strategy): strategy

    Returns:
        is_chief (bool): True if worker is chief, otherwise False
    """
    cr = strategy.cluster_resolver
    return (cr is None) or (cr.task_type == "worker" and cr.task_id == 0)
