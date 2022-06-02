"""Models for the classification of the acquisition parameters
"""
from typing import Callable, Collection, Union

import tensorflow as tf
from tensorflow.keras import Model, layers

from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.segbasisnet import SegBasisNet


def simple_model(
    inputs: tf.keras.Input,
    label_shapes: Collection[Union[tuple, int]],
    n_conv=5,
    global_pool=False,
) -> tf.keras.Model:
    """Build a simple model for classification

    Parameters
    ----------
    inputs : tf.keras.Input
        The input to the network
    label_shapes : Collection[Union[tuple, int]]
        The shapes of the labels, a tuple implies classification with the label shape
        and an int implies a regression task

    Returns
    -------
    tf.keras.Model
        The resulting keras model
    """
    # first convolution
    x = layers.Conv2D(filters=3, kernel_size=3, padding="same", name="input_conv")(inputs)
    x = layers.BatchNormalization(name="input_conv/bn")(x)
    x = layers.Activation("elu", name="input_conv/act")(x)

    # scale down multiple times
    for i in range(n_conv - 1):
        filters = 6 * (2 ** (i + 1))
        x = layers.Conv2D(
            filters=filters, kernel_size=3, padding="same", name=f"red{i}/conv"
        )(x)
        x = layers.BatchNormalization(name=f"red{i}/bn")(x)
        x = layers.Activation("elu", name=f"red{i}/act")(x)
        x = layers.MaxPool2D(pool_size=(2, 2), padding="same", name=f"red{i}/pool")(x)

    # get the different outputs
    outputs = []

    for i, shape in enumerate(label_shapes):
        if shape == 1:
            name = f"output_{i}"
        else:
            name = f"final/label_{i}_conv"
        out = layers.Conv2D(filters=shape, kernel_size=1, activation=None, name=name)(x)
        if global_pool:
            out = tf.keras.layers.GlobalMaxPooling2D(name=f"final/label_{i}_maxpool")(out)
        if shape == 1:
            pred = out
        elif shape > 1:
            pred = tf.keras.layers.Softmax(name=f"output_{i}", axis=-1)(out)
        else:
            raise ValueError("Shape cannot be 0")
        outputs.append(pred)

    return Model(inputs=inputs, outputs=tuple(outputs))


class SimpleModel(SegBasisNet):
    """Generate the simple model

    Parameters
    ----------
    loss : str
        The loss that should be used
    label_shapes : Collection[Union[tuple, int]]
        The shapes of the labels, a tuple implies classification with the label shape
        and an int implies a regression task
    is_training : bool
        If training should be performed
    do_finetune : bool
        If fine tuning should be done
    model_path : str
        The path where the files are located
    regularize : Tuple[bool, str, Any]
        The regularizer to use, first element signals if it should be used, the
        second the type and the third the parameters, by default (True, "L2", 0.00001)
    """

    def __init__(
        self,
        loss_name: str,
        tasks,
        label_shapes: Collection[Union[tuple, int]],
        is_training=True,
        do_finetune=False,
        model_path="",
        regularize=(True, "L2", 0.00001),
        **kwargs,
    ):
        super().__init__(
            loss_name,
            tasks,
            is_training,
            do_finetune,
            model_path,
            regularize,
            label_shapes=label_shapes,
            clipping_value=None,
            **kwargs,
        )

        self.divisible_by = 1

    @staticmethod
    def get_name():
        return "SimpleModel"

    def _build_model(self) -> Model:
        """Builds Model"""

        return simple_model(
            inputs=self.inputs["x"],
            label_shapes=self.options["label_shapes"],
            n_conv=self.options["n_conv"],
        )

    def get_hyperparameter_dict(self):
        """This function reads the hyperparameters from options and writes them to a dict of
        hyperparameters, which can then be read using tensorboard.

        Returns
        -------
        dict
            the hyperparameters as a dictionary
        """
        hyp = {
            "dimension": self.options.get("rank"),
            "loss": self.options["loss_name"],
            "n_conv": self.options["n_conv"],
        }
        hyperparameters = {key: str(value) for key, value in hyp.items()}
        return hyperparameters

    def _set_up_inputs(self):
        """setup the inputs. Inputs are taken from the config file."""
        ndim = len(cfg.train_input_shape) - 1
        input_shape = [None] * ndim + cfg.train_input_shape[-1:]
        self.inputs["x"] = tf.keras.Input(
            shape=input_shape,
            batch_size=cfg.batch_size_train,
            dtype=cfg.dtype,
            name="input",
        )
        self.options["out_channels"] = 1

    def get_loss(self, loss_name: str, task="segmentation") -> Callable:
        loss_func = super().get_loss(loss_name, task)
        # because the network is fully convolutional, the spatial dimension need ot be reduced
        if task in ("classification", "regression"):
            assert hasattr(loss_func, "__call__")
            axes = tuple(range(1, self.options["rank"] + 1))

            def loss_func_red(y, y_pred):
                # scale y up to the size of y_pred
                y_up = y
                for ax in axes:
                    y_up = tf.expand_dims(y_up, ax)
                    y_up = tf.repeat(y_up, repeats=y_pred.shape[ax], axis=ax)
                loss_val = loss_func(y_up, y_pred)
                return loss_val

            if isinstance(loss_name, str):
                loss_func_red.name = loss_name
            return loss_func_red
        else:
            return loss_func

    def get_metric(self, metric, task="segmentation"):
        metric_func = super().get_metric(metric, task)
        if task in ("classification", "regression"):
            assert hasattr(metric_func, "__call__")
            axes = tuple(range(1, self.options["rank"] + 1))

            def metric_func_red(y, y_pred):
                # scale y up to the size of y_pred
                y_up = y
                for ax in axes:
                    y_up = tf.expand_dims(y_up, ax)
                    y_up = tf.repeat(y_up, repeats=y_pred.shape[ax], axis=ax)
                metric_val = metric_func(y_up, y_pred)
                return metric_val

            metric_func_red.name = metric_func.name
            return metric_func_red
        else:
            return metric_func
