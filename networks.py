"""Models for the classification of the acquisition parameters
"""
from collections import OrderedDict
from typing import Callable, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

from SegClassRegBasis import config as cfg
from SegClassRegBasis.segbasisnet import SegBasisNet
from SegmentationArchitectures.utils import get_regularizer


def gaussian_kernel(
    size: int,
    std: float = 1.0,
) -> np.ndarray:
    """Makes 2D gaussian Kernel for convolution."""

    x = np.arange(-(size - 1) / 2, size / 2, dtype=np.float32)
    vals = np.exp(-(x**2) / (2 * std**2))
    gauss_kernel = np.einsum("i,j->ij", vals, vals)
    gauss_kernel_normalized = gauss_kernel / np.sum(gauss_kernel)

    return gauss_kernel_normalized


def conv_block(x, n_filters, regularizer, name):
    """Simple convolutional layer with Conv, Bn, ELU"""
    x = layers.Conv2D(
        filters=n_filters,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_regularizer=regularizer,
        name=f"{name}/conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}/bn")(x)
    x = layers.Activation("elu", name=f"{name}/act")(x)
    return x


# from https://www.tensorflow.org/guide/keras/custom_layers_and_models
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(
        self, trainable=True, name=None, dtype=None, dynamic=False, stddev=1, **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.stddev = float(stddev)

    def call(self, inputs, training=None, **_):
        """Call the layer"""
        z_mean, z_log_var = inputs
        zeros = tf.zeros_like(z_mean)
        epsilon = layers.GaussianNoise(stddev=self.stddev, name="epsilon")(
            zeros, training=training
        )
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


def auto_encoder(
    inputs: tf.keras.Input,
    depth=4,
    filter_base=16,
    regularize=(True, "L2", 0.001),
    skip_edges=False,
    smoothing_sigma=1,
    output_latent=False,
    output_min: Optional[float] = None,
    output_max: Optional[float] = None,
    variational=False,
    identity=False,
    keras_model: Model = Model,
    model_arguments=None,
    **kwargs,
) -> tf.keras.Model:
    """Build a simple model for classification

    Parameters
    ----------
    inputs : tf.keras.Input
        The input to the network
    depth : int, optional
        The number of convolutional layers, by default 4
    filter_base : int, optional
        The number of filters in the first layer, they will be doubled every block, by default 16
    regularize : tuple, optional
        The value for l2 regularization. By default: (True, "L2", 0.001).
    skip_edges : bool, optional
        Propagate the smoothed edge information to the last encoder
    smoothing_sigma : float, optional
        The sigma to use for smoothing before doing edge detection. By default 1
    output_latent : bool, optional
        If the latent data should be added to the output, by default False
    output_min : float, optional
        The minimum value, to which the output will be clipped, by default None
    output_max : float, optional
        The maximum value, to which the output will be clipped, by default None
    variational : bool, optional
        If a variational autoencoder should be used, the latent space is then sample from
        a distribution and the mean and log_var are added to the output, by default False
    identity : bool, optional
        Just output identity, useful for testing, by default False,
    keras_model : object, optional
        The model to use, by default tf.keras.Model
    model_arguments : dict, optional
        Arguments that should be passed to the keras model as keyword arguments

    Returns
    -------
    tf.keras.Model
        The resulting keras model
    """
    if model_arguments is None:
        model_arguments = {}

    input_channels = inputs.shape[-1]

    regularizer = get_regularizer(*regularize)

    if output_min is not None:
        if output_max is None:
            raise ValueError("Either provide output_min and output_max or neither")
        if output_max <= output_min:
            raise ValueError("output_min should be smaller than output_max")
    if output_max is None:
        if output_min is not None:
            raise ValueError("Either provide output_min and output_max or neither")

    # scale down multiple times
    if depth > 0:
        filters = [filter_base * (2**i) for i in range(depth)]
    else:
        filters = [filter_base]
    x = inputs
    for i, f in enumerate(filters[:-1]):
        x = conv_block(x, n_filters=f, regularizer=regularizer, name=f"auto_enc{i}")

    if variational:
        z_mean = conv_block(
            x, n_filters=filters[-1], regularizer=regularizer, name="auto_z_mean"
        )
        z_log_var = conv_block(
            x, n_filters=filters[-1], regularizer=regularizer, name="auto_z_log_var"
        )
        x = Sampling(name="auto_sampling")([z_mean, z_log_var])
    else:
        if depth > 0:
            x = conv_block(
                x, n_filters=filters[-1], regularizer=regularizer, name=f"auto_enc{depth-1}"
            )

    # we might use the latent dimension as output
    latent = x

    # scale up again
    for i, f in enumerate(filters[-2::-1]):
        x = layers.Conv2DTranspose(
            filters=f,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_regularizer=regularizer,
            name=f"auto_dec{i}/conv",
        )(x)
        x = layers.BatchNormalization(name=f"auto_dec{i}/bn")(x)
        x = layers.Activation("elu", name=f"auto_dec{i}/act")(x)

    if skip_edges:
        gauss_size = 9
        gauss_kernel = gaussian_kernel(gauss_size, std=smoothing_sigma)
        kernel = np.expand_dims(gauss_kernel, (-2, -1))

        if input_channels > 1:
            raise NotImplementedError(
                "Edge detection not implemented for more than 1 channel"
            )

        pad_s = gauss_size // 2
        inputs_smoothed = tf.pad(
            inputs,
            paddings=((0, 0), (pad_s, pad_s), (pad_s, pad_s), (0, 0)),
            mode="REFLECT",
        )
        # for some reason DepthwiseConv2D produces wrong results
        smoothing_layer = layers.Conv2D(
            filters=input_channels,
            kernel_size=gauss_size,
            strides=1,
            padding="valid",
            kernel_initializer=tf.keras.initializers.constant(kernel),
            trainable=False,
            name="auto_smoothing",
            use_bias=False,
        )
        smoothed = smoothing_layer(inputs_smoothed)

        sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_y = sobel_x.T
        sobel_kernel = np.array([sobel_x, sobel_y])[np.newaxis]
        sobel_layer = layers.Conv2D(
            filters=input_channels * 2,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.keras.initializers.constant(sobel_kernel),
            trainable=False,
            use_bias=False,
            name="auto_sobel_edges",
        )
        edges = sobel_layer(smoothed)

        x = layers.Conv2DTranspose(
            filters=filters[0],
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_regularizer=regularizer,
            name="auto_dec_final/conv_trans",
        )(x)
        x = layers.BatchNormalization(name="auto_dec_final/bn_trans")(x)
        x = layers.Activation("elu", name="auto_dec_final/act_trans")(x)

        x = tf.keras.layers.Concatenate(axis=-1, name="auto_concat_output_edges")(
            [x, edges]
        )
        x = layers.Conv2D(
            filters=filters[0],
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_regularizer=regularizer,
            name="auto_concat/conv",
        )(x)
        x = layers.BatchNormalization(name="auto_concat/bn_trans")(x)
        x = layers.Activation("elu", name="auto_concat/act_trans")(x)

    else:
        # Do final output without activation
        if depth > 0:
            x = layers.Conv2DTranspose(
                filters=filters[0],
                kernel_size=3,
                strides=2,
                padding="same",
                kernel_regularizer=regularizer,
                name="auto_dec_final/conv",
            )(x)
            x = layers.BatchNormalization(name="auto_dec_final/bn")(x)
            x = layers.Activation("elu", name="auto_dec_final/act")(x)

    x = layers.Conv2D(
        filters=filters[0],
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_regularizer=regularizer,
        name="auto_final_layer_0/conv",
    )(x)
    x = layers.BatchNormalization(name="auto_final_layer_0/bn")(x)
    x = layers.Activation("elu", name="auto_final_layer_0/act")(x)

    x = layers.Conv2D(
        filters=input_channels,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_regularizer=regularizer,
        name="auto_final_layer_1/conv",
    )(x)

    if output_min is not None and output_max is not None:
        x = tf.clip_by_value(x, output_min, output_max, name="auto_clip_output")

    model_output = [x]
    if output_latent:
        model_output.append(latent)
    if variational:
        model_output += [z_mean, z_log_var]
    if identity:
        model_output = tf.identity(inputs, name="auto_identity")

    return keras_model(inputs=inputs, outputs=model_output, **model_arguments)


class AutoEncoder(SegBasisNet):
    """Implements a simple autoencoder

    Parameters
    ----------
    loss_name : str
        Which loss is being used, this will be used to train the autoencoder
    tasks : OrderedDict[str, str], optional
        The tasks that should be performed, loss and metrics will be selected accordingly
    depth : int, optional
        The depth of the autoencoder, feature map will be reduced by 2**depth, by default 4
    filter_base : int, optional
        The number of features in the first feature map, they will double every layer, by default 16
    skip_edges : bool, optional
        Propagate the smoothed edge information to the last encoder
    output_min : float, optional
        The minimum value, to which the output will be clipped, by default None
    output_max : float, optional
        The maximum value, to which the output will be clipped, by default None
    variational : bool, optional
        If a variational autoencoder should be used, by default False
    smoothing_sigma : float, optional
        The sigma to use for smoothing before doing edge detection. By default 1
    is_training : bool, optional
        If in training, by default True
    do_finetune : bool, optional
        If finetuning is being done, by default False
    model_path : str, optional
        The path where the model is located for finetuning, by default ""
    regularize : tuple, optional
        Which regularizer should be used, by default (True, "L2", 0.00001)
    """

    def __init__(
        self,
        loss_name: str,
        tasks: OrderedDict,
        depth=4,
        filter_base=16,
        skip_edges=False,
        output_min=None,
        output_max=None,
        variational=False,
        smoothing_sigma=1,
        is_training=True,
        do_finetune=False,
        model_path=None,
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
            depth=depth,
            filter_base=filter_base,
            skip_edges=skip_edges,
            output_min=output_min,
            output_max=output_max,
            variational=variational,
            smoothing_sigma=smoothing_sigma,
            **kwargs,
        )

        self.divisible_by = 2**depth

    @staticmethod
    def get_name():
        return "AutoEncoder"

    def _build_model(self) -> Model:
        """Builds Model"""

        return auto_encoder(
            inputs=self.inputs["x"],
            depth=self.options["depth"],
            filter_base=self.options["filter_base"],
            skip_edges=self.options["skip_edges"],
            regularize=self.options["regularize"],
            output_min=self.options["output_min"],
            output_max=self.options["output_max"],
            variational=self.options["variational"],
            smoothing_sigma=self.options["smoothing_sigma"],
            identity=self.options.get("identity", False),
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
            "depth": self.options["depth"],
        }
        hyperparameters = {key: str(value) for key, value in hyp.items()}
        return hyperparameters

    def set_up_inputs(self):
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
                if y.shape == y_pred.shape:
                    y_up = y
                else:
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


class ResNet(SegBasisNet):
    """Implements a simple resnet

    Parameters
    ----------
    loss_name : Dict[str, str]
        Which loss is being used, this will be used to train the resnet
    tasks : OrderedDict[str, str], optional
        The tasks that should be performed, loss and metrics will be selected accordingly
    resnet_type : str, optional
        Which resnet should be used, by default ResNet50, 101 and 152 are also possible
    weights : str, optional
        Which weights should be used, can be None, "imagenet", a path or None, by default None
    is_training : bool, optional
        If in training, by default True
    do_finetune : bool, optional
        If finetuning is being done, by default False
    model_path : str, optional
        The path where the model is located for finetuning, by default ""
    regularize : tuple, optional
        Which regularizer should be used, by default (True, "L2", 0.00001)
    """

    def __init__(
        self,
        loss_name: Dict[str, str],
        tasks: OrderedDict,
        resnet_type="ResNet50",
        weights=None,
        is_training=True,
        do_finetune=False,
        model_path=None,
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
            resnet_type=resnet_type,
            weights=weights,
            **kwargs,
        )

        self.divisible_by = 16

    @staticmethod
    def get_name():
        return "ResNet"

    def set_up_inputs(self):
        ndim = len(cfg.train_input_shape) - 1
        if ndim == 2:
            super().set_up_inputs()
        input_shape = cfg.train_input_shape
        self.inputs["x"] = tf.keras.Input(
            shape=input_shape,
            batch_size=cfg.batch_size_train,
            dtype=cfg.dtype,
            name="input",
        )
        self.options["out_channels"] = cfg.num_classes_seg

    def _build_model(self) -> Model:
        """Builds Model"""

        inputs = self.inputs["x"]

        dim = len(inputs.get_shape()) - 2
        # remove batch
        if dim == 2:
            input_base = inputs[0].shape
        else:
            input_base = inputs[0, 0].shape

        resnet_params = dict(
            include_top=False,
            weights=self.options["weights"],
            input_shape=tuple(input_base),
            pooling="avg",
        )

        if self.options["resnet_type"] == "ResNet50":
            base_network = tf.keras.applications.resnet50.ResNet50(**resnet_params)
        elif self.options["resnet_type"] == "ResNet101":
            base_network = tf.keras.applications.resnet.ResNet101(**resnet_params)
        elif self.options["resnet_type"] == "ResNet152":
            base_network = tf.keras.applications.resnet.ResNet152(**resnet_params)

        if dim == 2:
            features = base_network(inputs)
        else:
            input_unstacked = tf.unstack(inputs, axis=1)
            output_unstacked = [base_network(t) for t in input_unstacked]
            output = tf.stack(output_unstacked, axis=1)
            output = tf.keras.layers.Conv1D(
                filters=output.shape[2],
                kernel_size=1,
                strides=1,
                activation="relu",
                name="stack_1d_conv_1",
                use_bias=True,
                kernel_regularizer=self.options["regularizer"],
            )(output)
            output = tf.keras.layers.Conv1D(
                filters=output.shape[2],
                kernel_size=1,
                strides=1,
                activation="relu",
                name="stack_1d_conv_2",
                use_bias=True,
                kernel_regularizer=self.options["regularizer"],
            )(output)
            output = tf.keras.layers.Conv1D(
                filters=output.shape[2] // output.shape[1],
                kernel_size=1,
                strides=1,
                activation="relu",
                name="stack_1d_conv_reduce",
                use_bias=True,
                kernel_regularizer=self.options["regularizer"],
            )(output)
            features = tf.reshape(
                output, (output.shape[0], output.shape[1] * output.shape[2])
            )

        tf.debugging.assert_all_finite(features, "NaN in features")

        # add final classification
        outputs = []
        for task_type, (name, n_label) in zip(
            self.tasks, self.options["label_shapes"].items()
        ):
            out = tf.keras.layers.Dense(
                units=n_label,
                activation="relu" if task_type == "classification" else None,
                kernel_regularizer=self.options["regularizer"],
                name=f"{name}-Dense" if task_type == "classification" else name,
            )(features)
            tf.debugging.assert_all_finite(out, f"NaN in {name} output 1")  # TODO: remove
            if task_type == "classification":
                out = tf.keras.layers.Softmax(name=f"{name}")(out)
            tf.debugging.assert_all_finite(out, f"NaN in {name} output 2")  # TODO: remove
            outputs.append(out)

        model = MaskedModel(inputs=inputs, outputs=outputs)
        return model


class MaskedModel(tf.keras.Model):
    """Do the training of the model with missing labels. Nans will be converted
    into zeros and assigned zero sample weight"""

    # pylint: disable=too-few-public-methods

    def _convert_data(self, data):
        if len(data) == 2:
            (x, y), sample_weights = data, None
        else:
            x, y, sample_weights = data
        if sample_weights is None:
            sample_weights = tuple(
                tf.cast(
                    tf.math.logical_not(
                        tf.math.reduce_any(
                            tf.math.is_nan(arr), axis=tuple(range(1, len(arr.shape)))
                        )
                    ),
                    tf.int32,
                )
                for arr in y
            )
        if isinstance(y, tuple):
            y = tuple(
                tf.where(tf.math.is_nan(arr), tf.cast(0, arr.dtype), arr) for arr in y
            )
        for num, y_t in enumerate(y):
            tf.debugging.assert_all_finite(y_t, f"NaNs found in y_true Nr. {num}")
        for num, x_t in enumerate(self(x)):
            tf.debugging.assert_all_finite(x_t, f"NaNs found in y_pred Nr. {num}")
        return x, y, sample_weights

    def train_step(self, data):
        """Do the training, see tf documentation"""
        x, y, sample_weights = self._convert_data(data)
        return super().train_step((x, y, sample_weights))

    def test_step(self, data):
        """Do the testing, see tf documentation"""
        x, y, sample_weights = self._convert_data(data)
        return super().test_step((x, y, sample_weights))

    def predict_step(self, data):
        """Do the prediction, see tf documentation"""
        x, y, sample_weights = self._convert_data(data)
        return super().predict_step((x, y, sample_weights))
