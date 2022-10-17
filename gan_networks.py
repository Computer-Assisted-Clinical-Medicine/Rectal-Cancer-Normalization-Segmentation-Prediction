"""
GAN model, which offers multiple choices of discriminators
"""
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Collection, Iterable, List, Optional, Tuple, Union

import numpy as np
import scipy
import SimpleITK as sitk
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers

from networks import AutoEncoder, auto_encoder
from SegClassRegBasis import config as cfg
from SegClassRegBasis import tf_utils, utils
from SegmentationArchitectures.utils import get_regularizer

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GANModel(Model):
    """
    Inherits from tf.keras.model, it implements an adversarial loss.

    clip_value can be used to clip the gradients to a maximum value
    """

    def __init__(
        self,
        *args,
        disc_real_fake: Optional[Model] = None,
        disc_real_fake_target_numbers=None,
        disc_real_fake_target_labels=None,
        disc_real_fake_tasks=None,
        disc_image: Optional[Model] = None,
        disc_image_target_numbers=None,
        disc_image_target_labels=None,
        disc_image_tasks=None,
        disc_latent: Optional[Model] = None,
        disc_latent_target_numbers=None,
        disc_latent_target_labels=None,
        disc_latent_tasks=None,
        clip_value: Optional[float] = None,
        variational=False,
        train_on_gen=False,
        latent_weight=1.0,
        image_weight=1.0,
        image_gen_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.disc_real_fake = disc_real_fake
        self.disc_real_fake_target_numbers = disc_real_fake_target_numbers
        if disc_real_fake_target_numbers is not None:
            if len(disc_real_fake_target_numbers) > 1:
                raise ValueError(
                    "there can only be one input for the real fake discriminator"
                )
        self.disc_real_fake_target_labels = disc_real_fake_target_labels
        self.disc_real_fake_tasks = disc_real_fake_tasks

        self.disc_image = disc_image
        self.disc_image_target_numbers = disc_image_target_numbers
        self.disc_image_target_labels = disc_image_target_labels
        self.disc_image_tasks = disc_image_tasks

        self.disc_latent = disc_latent
        self.disc_latent_target_numbers = disc_latent_target_numbers
        self.disc_latent_target_labels = disc_latent_target_labels
        self.disc_latent_tasks = disc_latent_tasks

        self.clip_value = clip_value

        self.variational = variational

        self.train_on_gen = train_on_gen

        self.latent_weight = float(latent_weight)
        self.image_weight = float(image_weight)
        self.image_gen_weight = float(image_gen_weight)

        # make sure the discriminators are compiled
        for disc in [self.disc_real_fake, self.disc_image, self.disc_latent]:
            if disc is None:
                continue
            if disc.compiled_loss is None:
                raise ValueError("Discriminators should already be compiled")

    def write_metrics(
        self,
        disc: Model,
        metrics: dict,
        predictions,
        labels,
        disc_name=None,
    ):
        """Take the labels and predictions and add them to the metrics dictionary"""
        if disc_name is None:
            disc_name = disc.name
        disc.compiled_metrics.reset_state()
        # calculate the metrics
        disc.compiled_metrics.update_state(labels, predictions)
        for met in disc.metrics:
            name = met.name
            if not name.startswith(disc_name):
                name = f"{disc_name}/{name}"
            if name.endswith("_loss"):
                name = name.replace("_loss", "/loss")
            metrics[name] = met.result()

    @tf.function
    def clip_gradients(self, gradients: list):
        """Clip the gradients to a maximum value

        Parameters
        ----------
        gradients : list
            The gradients

        Returns
        -------
        List
            The clipped gradients
        tf.Tensor
            The maximum gradient (before clipping)
        """
        # pylint:disable=invalid-unary-operand-type
        gradients = [g for g in gradients if g is not None]
        max_grad = tf.reduce_max([tf.reduce_max(tf.abs(g)) for g in gradients])
        if self.clip_value is not None:
            gradients = [
                tf.clip_by_value(g, -self.clip_value, self.clip_value) for g in gradients
            ]
        return gradients, max_grad

    @tf.function
    def train_step(self, data):
        """Perform the training step using an adversarial loss

        Adapted from https://keras.io/guides/customizing_what_happens_in_fit/

        Parameters
        ----------
        data : Tuple[tf.Tensor]
            A nested structure of `Tensor`s.

        Returns
        -------
        dict
            A `dict` containing values that will be passed to
            `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
            values of the `Model`'s metrics are returned. Example:
            `{'loss': 0.2, 'accuracy': 0.7}`.
        """

        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            raise NotImplementedError("Sample weights are not implemented")
        source_images, target_data = data

        if isinstance(target_data, tf.Tensor):
            target_data = (target_data,)

        losses = {}
        metrics = {}

        batch_size = tf.shape(source_images)[0]

        # Decode them to fake images
        generator_output = self(source_images, training=True)
        if len(self.outputs) == 1:
            generated_images = generator_output
        if self.disc_latent is not None:
            generated_images, latent_variables = generator_output[:2]

        # Train all discriminators
        if self.disc_real_fake is not None:
            real_image = target_data[self.disc_real_fake_target_numbers[0]]
            disc_input = tf.concat([generated_images, real_image], axis=0)
            # Assemble labels discriminating real from fake images
            fake_labels = tf.zeros((batch_size, 1))
            real_labels = tf.ones((batch_size, 1))
            labels_rf = tf.concat([fake_labels, real_labels], axis=0)

            # Add random noise to the labels - important trick!
            labels_rf += 0.05 * tf.random.uniform(tf.shape(labels_rf), dtype=tf.float32)

            with tf.GradientTape() as tape:
                predictions = self.disc_real_fake(disc_input, training=True)
                dis_loss = self.disc_real_fake.compiled_loss(labels_rf, predictions)
            grads = tape.gradient(dis_loss, self.disc_real_fake.trainable_weights)
            grads, max_grad = self.clip_gradients(grads)
            self.disc_real_fake.optimizer.apply_gradients(
                zip(grads, self.disc_real_fake.trainable_weights)
            )
            self.write_metrics(self.disc_real_fake, metrics, predictions, labels_rf)
            metrics["discriminator_real_fake/max_grad"] = max_grad
            losses["discriminator_real_fake/loss"] = dis_loss

        if self.disc_image is not None:
            labels_image = tuple(target_data[t] for t in self.disc_image_target_numbers)
            with tf.GradientTape() as tape:
                predictions = self.disc_image(source_images, training=True)
                dis_loss = self.disc_image.compiled_loss(labels_image, predictions)
                if self.train_on_gen:
                    predictions_gen = self.disc_image(generated_images, training=True)
                    dis_loss_gen = self.disc_image.compiled_loss(
                        labels_image, predictions_gen
                    )
                    dis_loss += dis_loss_gen * self.image_gen_weight
            grads = tape.gradient(dis_loss, self.disc_image.trainable_weights)
            grads, max_grad = self.clip_gradients(grads)
            self.disc_image.optimizer.apply_gradients(
                zip(grads, self.disc_image.trainable_weights)
            )
            self.write_metrics(self.disc_image, metrics, predictions, labels_image)
            metrics["disc_image/max_grad"] = max_grad
            losses["disc_image/loss"] = dis_loss
            if self.train_on_gen:
                self.write_metrics(
                    disc=self.disc_image,
                    metrics=metrics,
                    predictions=predictions_gen,
                    labels=labels_image,
                    disc_name="disc_image_gen",
                )
                losses["disc_image_gen/loss"] = dis_loss_gen

        if self.disc_latent is not None:
            labels_latent = tuple(target_data[t] for t in self.disc_latent_target_numbers)
            with tf.GradientTape() as tape:
                predictions = self.disc_latent(latent_variables, training=True)
                dis_loss = self.disc_latent.compiled_loss(labels_latent, predictions)
            grads = tape.gradient(dis_loss, self.disc_latent.trainable_weights)
            grads, max_grad = self.clip_gradients(grads)
            self.disc_latent.optimizer.apply_gradients(
                zip(grads, self.disc_latent.trainable_weights)
            )
            self.write_metrics(self.disc_latent, metrics, predictions, labels_latent)
            metrics["disc_latent/max_grad"] = max_grad
            losses["disc_latent/loss"] = dis_loss

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        gen_loss = tf.convert_to_tensor(0, dtype=tf.float32)
        with tf.GradientTape() as tape:
            generator_output = self(source_images, training=True)
            if len(self.outputs) == 1:
                pred_images = generator_output
            if self.disc_latent is not None:
                pred_images, latent_variables = generator_output[:2]
                generator_output = generator_output[2:]
            if self.variational:
                z_mean, z_log_var = generator_output
            if not isinstance(pred_images, tf.Tensor):
                pred_images = pred_images[0]
            # there might be an extra loss from the autoencoder
            if self.compiled_loss is not None:
                autoencoder_loss = self.compiled_loss(source_images, pred_images)
                losses["generator/autoencoder_loss"] = autoencoder_loss
                gen_loss += autoencoder_loss

            if self.disc_real_fake is not None:
                predictions = self.disc_real_fake(pred_images)
                disc_real_fake_loss = self.disc_real_fake.compiled_loss(
                    self.disc_real_fake_target_labels, predictions
                )
                gen_loss += disc_real_fake_loss
                losses["generator/disc_real_fake_loss"] = disc_real_fake_loss
                self.write_metrics(
                    self.disc_real_fake,
                    metrics,
                    predictions,
                    self.disc_real_fake_target_labels,
                    disc_name="generator-real-fake",
                )

            if self.disc_image is not None:
                predictions = self.disc_image(pred_images)
                disc_image_loss = (
                    self.disc_image.compiled_loss(
                        self.disc_image_target_labels, predictions
                    )
                    * self.image_weight
                )
                gen_loss += disc_image_loss
                losses["generator/disc_image_loss"] = disc_image_loss
                self.write_metrics(
                    self.disc_image,
                    metrics,
                    predictions,
                    self.disc_image_target_labels,
                    disc_name="generator-image",
                )

            if self.disc_latent is not None:
                predictions = self.disc_latent(latent_variables)
                disc_latent_loss = (
                    self.disc_latent.compiled_loss(
                        self.disc_latent_target_labels, predictions
                    )
                    * self.latent_weight
                )
                gen_loss += disc_latent_loss
                losses["generator/disc_latent_loss"] = disc_latent_loss
                self.write_metrics(
                    self.disc_latent,
                    metrics,
                    predictions,
                    self.disc_latent_target_labels,
                    disc_name="generator-latent",
                )

            if self.variational:
                # Add KL divergence regularization loss.
                kl_loss = -0.5 * tf.reduce_mean(
                    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
                )
                gen_loss += kl_loss
                losses["generator/kl_loss"] = kl_loss

        grads = tape.gradient(gen_loss, self.trainable_weights)
        grads, max_grad = self.clip_gradients(grads)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        metrics["generator/max_grad"] = max_grad

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(source_images, pred_images)
        for met in self.metrics:
            # remove last layer name from metric name
            metric_name = met.name.replace(self.output_names[0] + "_", "")
            metrics[f"generator/{metric_name}"] = met.result()

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        losses["generator/total_loss"] = gen_loss

        if self.compiled_loss is not None:
            losses["generator/autoencoder_loss"] = autoencoder_loss

        return metrics | losses

    @tf.function
    def test_step(self, data):
        """
        Perform the test step using an adversarial loss, does the same as the training
        step but does not update the weights
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            raise NotImplementedError("Sample weights are not implemented")
        source_images, target_data = data

        if isinstance(target_data, tf.Tensor):
            target_data = (target_data,)

        losses = {}
        metrics = {}

        batch_size = tf.shape(source_images)[0]

        # Decode them to fake images
        generator_output = self(source_images)
        if len(self.outputs) == 1:
            generated_images = generator_output
        if self.disc_latent is not None:
            generated_images, latent_variables = generator_output[:2]

        # Train all discriminators
        if self.disc_real_fake is not None:
            real_image = target_data[self.disc_real_fake_target_numbers[0]]
            disc_input = tf.concat([generated_images, real_image], axis=0)
            # Assemble labels discriminating real from fake images
            fake_labels = tf.zeros((batch_size, 1))
            real_labels = tf.ones((batch_size, 1))
            labels_rf = tf.concat([fake_labels, real_labels], axis=0)
            predictions = self.disc_real_fake(disc_input)
            self.write_metrics(self.disc_real_fake, metrics, predictions, labels_rf)

        if self.disc_image is not None:
            labels_image = tuple(target_data[t] for t in self.disc_image_target_numbers)
            predictions = self.disc_image(source_images)
            if self.train_on_gen:
                predictions_gen = self.disc_image(generated_images, training=True)
            self.write_metrics(self.disc_image, metrics, predictions, labels_image)
            if self.train_on_gen:
                self.write_metrics(
                    disc=self.disc_image,
                    metrics=metrics,
                    predictions=predictions_gen,
                    labels=labels_image,
                    disc_name="disc_image_gen",
                )

        if self.disc_latent is not None:
            labels_latent = tuple(target_data[t] for t in self.disc_latent_target_numbers)
            predictions = self.disc_latent(latent_variables)
            self.write_metrics(self.disc_latent, metrics, predictions, labels_latent)

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        gen_loss = tf.convert_to_tensor(0, dtype=tf.float32)
        generator_output = self(source_images, training=True)
        if len(self.outputs) == 1:
            pred_images = generator_output
        if self.disc_latent is not None:
            pred_images, latent_variables = generator_output[:2]
            generator_output = generator_output[2:]
        if self.variational:
            z_mean, z_log_var = generator_output
        if not isinstance(pred_images, tf.Tensor):
            pred_images = pred_images[0]
        # there might be an extra loss from the autoencoder
        if self.compiled_loss is not None:
            autoencoder_loss = self.compiled_loss(source_images, pred_images)
            gen_loss += autoencoder_loss

        if self.disc_real_fake is not None:
            predictions = self.disc_real_fake(pred_images)
            disc_real_fake_loss = self.disc_real_fake.compiled_loss(
                self.disc_real_fake_target_labels, predictions
            )
            gen_loss += disc_real_fake_loss
            losses["generator/disc_real_fake_loss"] = disc_real_fake_loss
            self.write_metrics(
                self.disc_real_fake,
                metrics,
                predictions,
                self.disc_real_fake_target_labels,
                disc_name="generator-real-fake",
            )

        if self.disc_image is not None:
            predictions = self.disc_image(pred_images)
            disc_image_loss = (
                self.disc_image.compiled_loss(self.disc_image_target_labels, predictions)
                * self.image_weight
            )
            gen_loss += disc_image_loss
            losses["generator/disc_image_loss"] = disc_image_loss
            self.write_metrics(
                self.disc_image,
                metrics,
                predictions,
                self.disc_image_target_labels,
                disc_name="generator-image",
            )

        if self.disc_latent is not None:
            predictions = self.disc_latent(latent_variables)
            disc_latent_loss = (
                self.disc_latent.compiled_loss(self.disc_latent_target_labels, predictions)
                * self.latent_weight
            )
            gen_loss += disc_latent_loss
            losses["generator/disc_latent_loss"] = disc_latent_loss
            self.write_metrics(
                self.disc_latent,
                metrics,
                predictions,
                self.disc_latent_target_labels,
                disc_name="generator-latent",
            )

        if self.variational:
            # Add KL divergence regularization loss.
            kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            )
            gen_loss += kl_loss
            losses["generator/kl_loss"] = kl_loss

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(source_images, pred_images)
        for met in self.metrics:
            # fix some metric names
            if met.name == "dec_final/conv_loss":
                continue
            metric_name = met.name.replace(self.output_names[0] + "_", "")
            metrics[f"generator/{metric_name}"] = met.result()

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        losses["generator/total_loss"] = gen_loss

        if self.compiled_loss is not None:
            losses["generator/autoencoder_loss"] = autoencoder_loss

        return metrics | losses


class AutoencoderGAN(AutoEncoder):
    """Implements a simple autoencoder with adversarial training.

    There are three types of discriminators, the real/fake discriminator, which takes the real
    and the generated image as argument. Then there are two additional discriminators, which
    can have different goals with the image or latent dimensions as target.

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
    discriminators : List[dict], optional
        The discriminator that should be used, will be passed on to self.get_discriminator
    is_training : bool, optional
        If in training, by default True
    do_finetune : bool, optional
        If finetuning is being done, by default False
    model_path : str, optional
        The path where the model is located for finetuning, by default ""
    regularize : tuple, optional
        Which regularizer should be used, by default (True, "L2", 0.00001)
    clip_value : Optional[float], optional
        If not none, gradients will be clipped to this value, by default None
    regression_min : float, optional
        The minimum value for regression output, by default 0
    regression_max : float, optional
        The maximum value for regression output, by default 1
    output_min : float, optional
        The minimum value, to which the output will be clipped, by default None
    output_max : float, optional
        The maximum value, to which the output will be clipped, by default None
    variational : bool, optional
        If a variational autoencoder should be used, by default False
    train_on_gen : bool, optional
        If the image discriminator should be trained on the generated images, by default False
    smoothing_sigma : float, optional
        The sigma to use for smoothing before doing edge detection. By default 1
    latent_weight : float, optional
        The weight for the latent discriminators, by default 1
    image_weight : float, optional
        The weight for the image discriminators, by default 1
    image_gen_weight : float, optional
        The weight of the image discriminator trained on the generated images,
        by default 1
    """

    def __init__(
        self,
        loss_name: str,
        tasks: OrderedDict,
        depth=4,
        filter_base=16,
        skip_edges=False,
        discriminators=None,
        is_training=True,
        do_finetune=False,
        model_path="",
        regularize=(True, "L2", 0.00001),
        clip_value=None,
        regression_min=0,
        regression_max=1,
        output_min=None,
        output_max=None,
        variational=False,
        train_on_gen=False,
        smoothing_sigma=1,
        latent_weight=1,
        image_weight=1,
        image_gen_weight=1,
        **kwargs,
    ):

        if discriminators is None:
            discriminators = []
        elif not isinstance(discriminators, list):
            raise ValueError("discriminators should be a list of dictionaries")

        self.discriminators = []
        self.discriminator_targets = []

        self.disc_real_fake = None
        self.disc_real_fake_tasks = None
        self.disc_real_fake_target_numbers = None
        self.disc_real_fake_target_labels = None
        self.disc_image = None
        self.disc_image_tasks = None
        self.disc_image_target_numbers = None
        self.disc_image_target_labels = None
        self.disc_latent = None
        self.disc_latent_tasks = None
        self.disc_latent_target_numbers = None
        self.disc_latent_target_labels = None

        self.regression_min = regression_min
        self.regression_max = regression_max

        super().__init__(
            loss_name=loss_name,
            tasks=tasks,
            depth=depth,
            filter_base=filter_base,
            skip_edges=skip_edges,
            output_min=output_min,
            output_max=output_max,
            variational=variational,
            train_on_gen=train_on_gen,
            smoothing_sigma=smoothing_sigma,
            latent_weight=latent_weight,
            image_weight=image_weight,
            image_gen_weight=image_gen_weight,
            is_training=is_training,
            do_finetune=do_finetune,
            model_path=model_path,
            regularize=regularize,
            discriminators=discriminators,
            clip_value=clip_value,
            **kwargs,
        )

        # remove all losses that are just a discriminator
        self.outputs["loss"] = [
            loss
            for loss, tsk in zip(self.outputs["loss"], self.tasks)
            if tsk == "autoencoder"
        ]
        assert len(self.outputs["loss"]) == 1
        # Use no loss for the latent output
        if self.disc_latent is not None:
            self.outputs["loss"].append(None)
        if self.options["variational"]:
            self.outputs["loss"].append(None)
            self.outputs["loss"].append(None)

    @staticmethod
    def get_name():
        return "AutoencoderGAN"

    def get_discriminator(
        self,
        discriminators: List[dict],
        input_type: str,
        regularize: Optional[tuple] = None,
        disc_type="SimpleConv",
        discriminator_n_conv=3,
        model_name=None,
        **kwargs,
    ) -> Tuple[Model, List[str], List[int], List[tf.Tensor]]:
        """Generate the discriminator needed for training

        Parameters
        ----------
        discriminators : List[dict]
            The individual discriminators to combine, they should have the following fields:
            -- name: the name of the discriminator (used as output name)
            -- target_labels: target label to predict, optional for classification
            -- goal: The goal of the discriminator, confuse or predict.
              confuse means that there should be not clear result
            -- loss: the loss to use
            -- loss_weight: which weight to assign the loss, by default 1
        input_type : str
            Which input type to use, it can be image or latent (for latent space)
        regularize : Optional[tuple], optional
            If regularization should be performed, by default None
        disc_type : str, optional
            The type of the discriminator, by default "SimpleConv"
        discriminator_n_conv : int, optional
            The number of the convolutional layers, by default 3
        model_name : str, optional
            The model name, if None, the input type, by default None

        Returns
        -------
        Model
            The discriminator
        List[str]
            The tasks of the individual outputs (classification or regression)
        List[int]
            The target number in the output
        List[tf.Tensor]
            The target labels for the desired output
        """

        batch_size = cfg.batch_size_train  # TODO: put into options
        if model_name is None:
            model_name = f"disc_{input_type}"

        # if the target is the image, use autoencoder as task, otherwise, use the name
        task_target_numbers = {name: n for n, name in enumerate(self.task_names)}
        task_target_numbers["discriminator_real_fake"] = task_target_numbers["autoencoder"]

        tasks = []
        output_shapes = []
        output_names = []
        target_numbers = []
        target_labels = []
        for disc in discriminators:
            name = disc["name"]
            if name == "discriminator_real_fake":
                tasks.append("discriminator-regression")
            else:
                tasks.append(dict(zip(self.task_names, self.tasks)).get(name, None))

            if name in self.options["label_shapes"]:
                out_shape = self.options["label_shapes"][name]
            elif name == "discriminator_real_fake":
                out_shape = 1
            else:
                raise ValueError(f"{name} unknown")
            output_shapes.append(out_shape)
            output_names.append(name)

            target_numbers.append(task_target_numbers[name])

            if disc.get("target_labels", None) is None:
                if disc["goal"] == "confuse":
                    target_lbl = [1]  # all real
                else:
                    if out_shape == 1:
                        raise ValueError("target_labels have to be provided for regression")
                    target_lbl = [1 / out_shape] * out_shape
            else:
                if out_shape == 1:
                    mapping_dict = self.options["mapping"]["regression"][name]
                    mapping = scipy.interpolate.interp1d(
                        list(mapping_dict.values()), list(mapping_dict.keys())
                    )
                    target_lbl = [int(mapping(disc["target_labels"]))]
                else:
                    # create one hot vector
                    target_lbl = np.zeros(out_shape)
                    target_lbl[
                        self.options["mapping"]["classification"][name][
                            disc["target_labels"]
                        ]
                    ] = 1

            # add batch dimension
            target_lbl_batch = tf.repeat(tf.expand_dims(target_lbl, 0), batch_size, 0)

            target_labels.append(target_lbl_batch)

        if regularize is None:
            regularizer = get_regularizer(*self.options["regularize"])
        else:
            regularizer = get_regularizer(*regularize)

        if input_type == "image":
            input_shape = self.inputs["x"].shape[1:]
        elif input_type == "latent":
            input_shape = (
                None,
                None,
                int(self.options["filter_base"] * (2 ** (self.options["depth"] - 1))),
            )
        else:
            raise ValueError(f"Input type {input_type} unknown")

        if disc_type == "SimpleConv":
            model_input = keras.Input(shape=input_shape, batch_size=cfg.batch_size_train)
            x = model_input
            for i in range(discriminator_n_conv):
                x = layers.Conv2D(
                    32 * 2**i,
                    (3, 3),
                    strides=(2, 2),
                    padding="same",
                    kernel_regularizer=regularizer,
                )(x)
                x = layers.SpatialDropout2D(0.2)(x)
                x = layers.LeakyReLU(alpha=0.2)(x)

            outputs = []
            for n_out, tsk, out_name in zip(output_shapes, tasks, output_names):
                out = layers.Conv2D(n_out, kernel_size=1, kernel_regularizer=regularizer)(x)

                if n_out > 1 and "classification" in tsk:
                    out = layers.Softmax()(out)
                elif "regression" in tsk:
                    out = tf.clip_by_value(
                        out,
                        clip_value_min=self.regression_min,
                        clip_value_max=self.regression_max,
                    )

                out = layers.GlobalAveragePooling2D(name=out_name)(out)

                outputs.append(out)

            discriminator = keras.Model(
                model_input,
                outputs,
                name=model_name,
            )

        return discriminator, tasks, target_numbers, target_labels

    def _build_model(self) -> Model:
        """Builds Model"""

        task_metrics_dict = {
            "discriminator-classification": (
                tf.keras.metrics.Precision,
                tf.keras.metrics.Recall,
                tf.keras.metrics.AUC,
            ),
            "discriminator-regression": (tf.keras.metrics.RootMeanSquaredError,),
        }

        self.real_fake_disc_list = [
            d
            for d in self.options["discriminators"]
            if d["name"] == "discriminator_real_fake"
        ]
        if len(self.real_fake_disc_list) > 0:
            (
                self.disc_real_fake,
                self.disc_real_fake_tasks,
                self.disc_real_fake_target_numbers,
                self.disc_real_fake_target_labels,
            ) = self.get_discriminator(
                self.real_fake_disc_list,
                input_type="image",
                discriminator_n_conv=self.options["disc_real_fake_n_conv"],
                model_name="discriminator_real_fake",
            )

        self.image_discs_list = [
            d
            for d in self.options["discriminators"]
            if d["input_type"] == "image" and d["name"] != "discriminator_real_fake"
        ]
        if len(self.image_discs_list) > 0:
            (
                self.disc_image,
                self.disc_image_tasks,
                self.disc_image_target_numbers,
                self.disc_image_target_labels,
            ) = self.get_discriminator(
                self.image_discs_list,
                input_type="image",
                discriminator_n_conv=self.options["disc_image_n_conv"],
            )

        self.latent_discs_list = [
            d for d in self.options["discriminators"] if d["input_type"] == "latent"
        ]
        if len(self.latent_discs_list) > 0:
            (
                self.disc_latent,
                self.disc_latent_tasks,
                self.disc_latent_target_numbers,
                self.disc_latent_target_labels,
            ) = self.get_discriminator(
                self.latent_discs_list,
                input_type="latent",
                discriminator_n_conv=self.options["disc_latent_n_conv"],
            )

        for name, disc_model, disc_list, disc_tasks in zip(
            ["real_fake", "image", "latent"],
            [self.disc_real_fake, self.disc_image, self.disc_latent],
            [self.real_fake_disc_list, self.image_discs_list, self.latent_discs_list],
            [self.disc_real_fake_tasks, self.disc_image_tasks, self.disc_latent_tasks],
        ):
            if disc_model is None:
                continue
            metric_objects = []
            for task in disc_tasks:
                metric_classes = task_metrics_dict[task]
                metric_objects.append(
                    tuple(m(name=f"{m.__name__}") for m in metric_classes)
                )
            l_r = self.options.get(f"disc_{name}_lr", 0.001)
            if isinstance(l_r, (list, tuple)):
                l_r = self.get_lr_scheduler(*l_r)
            disc_model.compile(
                optimizer=tf_utils.get_optimizer(
                    optimizer=self.options.get(f"disc_{name}_optimizer", "Adam"),
                    l_r=l_r,
                ),
                loss=[self.get_loss(disc["loss"]) for disc in disc_list],
                loss_weights=[disc.get("loss_weight", 1) for disc in disc_list],
                metrics=metric_objects,
            )

        return auto_encoder(
            inputs=self.inputs["x"],
            depth=self.options["depth"],
            filter_base=self.options["filter_base"],
            skip_edges=self.options["skip_edges"],
            output_latent=len(self.latent_discs_list) > 0,
            output_min=self.options["output_min"],
            output_max=self.options["output_max"],
            variational=self.options["variational"],
            smoothing_sigma=self.options["smoothing_sigma"],
            keras_model=GANModel,
            model_arguments={
                "disc_real_fake": self.disc_real_fake,
                "disc_real_fake_tasks": self.disc_real_fake_tasks,
                "disc_real_fake_target_numbers": self.disc_real_fake_target_numbers,
                "disc_real_fake_target_labels": self.disc_real_fake_target_labels,
                "disc_image": self.disc_image,
                "disc_image_tasks": self.disc_image_tasks,
                "disc_image_target_numbers": self.disc_image_target_numbers,
                "disc_image_target_labels": self.disc_image_target_labels,
                "disc_latent": self.disc_latent,
                "disc_latent_tasks": self.disc_latent_tasks,
                "disc_latent_target_numbers": self.disc_latent_target_numbers,
                "disc_latent_target_labels": self.disc_latent_target_labels,
                "clip_value": self.options["clip_value"],
                "variational": self.options["variational"],
                "train_on_gen": self.options["train_on_gen"],
                "latent_weight": self.options["latent_weight"],
                "image_weight": self.options["image_weight"],
                "image_gen_weight": self.options["image_gen_weight"],
            },
        )

    def _get_task_metrics(
        self, metrics: List[Union[str, Callable, Collection]], tasks: List[str]
    ):
        metric_objects = super()._get_task_metrics(metrics, tasks)
        metric_objects = tuple(m for m in metric_objects if len(m))
        if len(metric_objects) == 1:
            metric_objects = metric_objects[0]
        return metric_objects

    def _get_task_losses(self, loss_input: Union[str, dict, object, Iterable]):
        loss_objects = super()._get_task_losses(loss_input)
        return loss_objects

    def plot_model(self, save_dir: Path):
        for disc in [self.disc_real_fake, self.disc_image, self.disc_latent]:
            if disc is None:
                continue
            tf.keras.utils.plot_model(
                disc,
                to_file=save_dir / f"{disc.name}.png",
            )
            tf.keras.utils.plot_model(
                disc,
                to_file=save_dir / f"{disc.name}_with_shapes.png",
                show_shapes=True,
            )
        return super().plot_model(save_dir)

    def apply(self, version, application_dataset, filename, apply_path):
        """Apply the network to test data. If the network is 2D, it is applied
        slice by slice. If it is 3D, it is applied to the whole images. If that
        runs out of memory, it is applied in patches in z-direction with the same
        size as used in training.

        Parameters
        ----------
        version : int or str
            The epoch, can be int or identifier (final for example)
        application_dataset : ApplyBasisLoader
            The dataset
        filename : str
            The file that is being processed, used to generate the new file name
        apply_path : str
            Where the files are written
        """

        output = self.get_network_output(application_dataset, filename)
        # ignore the latent dimension
        apply_path = Path(apply_path)

        assert "autoencoder" in self.task_names

        name = Path(filename).name
        res_name = f"prediction-{name}-{version}"

        # export the image
        # ignore the latent dimension
        out = output[0]
        # clip to the right range
        out = np.clip(out, a_min=-1.0, a_max=1.0)
        if self.options["rank"] == 2:
            out = application_dataset.remove_padding(out)
        pred_img = utils.output_to_image(
            output=out,
            task="autoencoder",
            processed_image=application_dataset.get_processed_image(filename),
            original_image=application_dataset.get_original_image(filename),
        )
        new_image_path = apply_path / f"{res_name}_autoencoder{cfg.file_suffix}"
        sitk.WriteImage(pred_img, str(new_image_path.resolve()))
