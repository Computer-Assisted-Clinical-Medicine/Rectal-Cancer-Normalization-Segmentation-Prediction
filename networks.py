"""Models for the classification of the acquisition parameters
"""
from pathlib import Path
from typing import Collection, Union

import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers

from SegmentationNetworkBasis import config as cfg
from SegmentationNetworkBasis.segbasisnet import SegBasisNet


def simple_model(
    inputs: tf.keras.Input,
    label_shapes: Collection[Union[tuple, int]],
    n_conv=5,
    global_pool=True,
) -> tf.keras.Model:
    """Build a simple model for classification

    Parameters
    ----------
    inputs : tf.keras.Input
        The input to the network
    label_shapes : Collection[Union[tuple, int]]
        The shapes of the labels, a tuple implies classification with the label shape
        and an int imples a regression task

    Returns
    -------
    tf.keras.Model
        The resulting keras model
    """
    # first convolution
    x = layers.Conv2D(filters=12, kernel_size=3, padding="same", name="input_conv")(inputs)
    x = layers.BatchNormalization(name="input_conv/bn")(x)
    x = layers.Activation("elu", name="input_conv/act")(x)

    # scale down multiple times
    for i in range(n_conv):
        filters = 12 * (2 ** (i + 1))
        x = layers.Conv2D(
            filters=filters, kernel_size=3, padding="same", name=f"red{i}/conv"
        )(x)
        x = layers.BatchNormalization(name=f"red{i}/bn")(x)
        x = layers.Activation("elu", name=f"red{i}/act")(x)
        x = layers.SpatialDropout2D(rate=0.2, name=f"red{i}/dropout")(x)
        x = layers.MaxPool2D(pool_size=(2, 2), padding="same", name=f"red{i}/pool")(x)

    # get the different outputs
    outputs = []

    for i, shape in enumerate(label_shapes):
        out = layers.Conv2D(filters=shape, kernel_size=1, activation=None)(x)
        out = tf.keras.layers.Dropout(0.2)(out)
        out = tf.keras.layers.BatchNormalization()(out)
        if global_pool:
            out = tf.keras.layers.GlobalMaxPooling2D()(out)
        if shape == 1:
            final_act = None
        elif shape > 1:
            final_act = "softmax"
        else:
            raise ValueError("Shape cannot be 0")
        pred = tf.keras.layers.Activation(final_act, name=f"output_{i}")(out)
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
        and an int imples a regression task
    is_training : bool
        If training should be performed
    do_finetune : bool
        If finetuning should be done
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

    @staticmethod
    def get_name():
        return "SimpleModel"

    def _build_model(self) -> Model:
        """Builds Model"""

        # only do global pooling when training the model
        return simple_model(
            inputs=self.inputs["x"],
            label_shapes=self.options["label_shapes"],
        )

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

    def apply(self, version, application_dataset, filename, apply_path):

        n_outputs = len(self.model.outputs)
        results = []
        for sample in application_dataset(filename):
            # add batch dimension
            res = self.model(sample.reshape((1,) + sample.shape))
            results.append(res)

        # separate into multiple lists
        output = [[row[out] for row in results] for out in range(n_outputs)]
        # and concatenate them
        output = [tf.concat(out, 0).numpy().squeeze() for out in output]

        # write the output to a file
        output_df = pd.DataFrame(columns=pd.RangeIndex(n_outputs))
        for col, values in zip(output_df, output):
            output_df[col] = list(values)

        name = Path(filename).name
        output_path = Path(apply_path) / f"prediction-{name}-{version}.json"
        output_df.index.rename("Slice", inplace=True)
        output_df.to_json(output_path, indent=1)
        output_df.to_csv(output_path.with_suffix(".csv"), sep=";")

        return output
