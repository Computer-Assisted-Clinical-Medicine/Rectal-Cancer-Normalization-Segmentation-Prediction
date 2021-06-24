"""
Different functions to help interpet the models.
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import ndimage


def grad_cam(
    input_model: tf.keras.Model,
    images: np.ndarray,
    layer_name: str,
    cls=1,
    pixel_weights=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a GRAD-CAM activation map for a batch of input images. To get the
    maps, all probabilities aver averaged over the image dimensions.
    See also https://arxiv.org/pdf/2002.11434.pdf

    Parameters
    ----------
    input_model : tf.keras.Model
        The model to use
    images : np.ndarray
        An array of input images with shape (batch, width, height, channels)
    layer_name : str
        The name of the layer for which the output should be used
    cls : int, optional
        The class to use, by default 1
    pixel_weights : np.array, optional
        If weights should be apply to the pixels when averaging the label. The
        array should have the same spatial dimensions as the input, by default None

    Returns
    -------
    np.ndarray
        The output of the network (as comparison)
    np.ndarray
        The activation maps for all images with shape (batch, width, height)
        They are upsampled if necessary.
    """

    # make sure the function is not running in eager mode
    assert not tf.executing_eagerly(), "This does not work in eager mode."

    y: tf.Tensor = input_model.output

    conv_output = input_model.get_layer(layer_name).output

    if pixel_weights is not None:
        # norm the weights
        pixel_weights = pixel_weights / np.mean(pixel_weights)

    # the value for the class is the y^c from the paper
    # to get an average over all pixels, just take the average
    if len(y.shape) == 4:
        if pixel_weights is None:
            y_c = tf.reduce_mean(y[..., cls], axis=(1, 2))
        else:
            y_c = tf.reduce_mean(y[..., cls] * pixel_weights, axis=(1, 2))
    else:
        # otherwise, this is a classification task with two dimensions
        assert len(y.shape) == 2
        y_c = y[:, cls]

    # calculate the gradients with respect to the feature map of the layer
    # this is the partial derivativ from the paper (the list has dim. 1)
    grads = K.gradients(y_c, conv_output)[0]

    # make it a function to be able to get actual values
    model_function = K.function([input_model.input], [conv_output, grads, y])

    output, grads_fm, y = model_function([images])

    results = np.zeros(images.shape[:3])
    width, height = images.shape[1:3]

    for num, (out, grad) in enumerate(zip(output, grads_fm)):

        # perform the sum to get the weights (w_c^k or alpha_c^k in the paper)
        weights = np.mean(grad, axis=(0, 1))

        # apply the weights to the feature map
        cam = np.dot(out, weights)

        # do the ReLU
        cam = np.maximum(cam, 0)

        # resize to the output shape
        cam = resize_image(width, height, cam)

        results[num] = cam

    # norm results
    results = results / np.max(results)

    return y, results


def resize_image(width, height, cam):
    return ndimage.zoom(
        input=cam, zoom=(width / cam.shape[0], height / cam.shape[1]), order=2
    )


def grad_cam_plus_plus(
    input_model: tf.keras.Model,
    images: np.ndarray,
    layer_name: str,
    cls=1,
    pixel_weights=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a GRAD-CAM activation map for a batch of input images. To get the
    maps, all probabilities aver averaged over the image dimensions.
    See also https://arxiv.org/pdf/1710.11063.pdf

    Parameters
    ----------
    input_model : tf.keras.Model
        The model to use
    images : np.ndarray
        An array of input images with shape (batch, width, height, channels)
    layer_name : str
        The name of the layer for which the output should be used. This should
        be the rectified feature map. (At least accroding to the authors git repo)
    cls : int, optional
        The class to use, by default 1
    pixel_weights : np.array, optional
        If weights should be apply to the pixels when averaging the label. The
        array should have the same spatial dimensions as the input, by default None

    Returns
    -------
    np.ndarray
        The output of the network (as comparison)
    np.ndarray
        The activation maps for all images with shape (batch, width, height)
        They are upsampled if necessary.
    """

    # for numerical stability
    eps = 1e-9

    # make sure the function is not running in eager mode
    assert not tf.executing_eagerly(), "This does not work in eager mode."

    y = input_model.output

    conv_output = input_model.get_layer(layer_name).output

    if pixel_weights is not None:
        # norm the weights
        pixel_weights = pixel_weights / np.mean(pixel_weights)

    # the value for the class is the y^c from the paper
    # to get an average over all pixels, just take the average
    if len(y.shape) == 4:
        if pixel_weights is None:
            y_c = tf.reduce_mean(y[..., cls], axis=(1, 2))
        else:
            y_c = tf.reduce_mean(y[..., cls] * pixel_weights, axis=(1, 2))
    else:
        # otherwise, this is a classification task with two dimensions
        assert len(y.shape) == 2
        y_c = y[:, cls]

    # calculate the gradients with respect to the feature map of the layer
    # this is the partial derivativ from the paper
    grads = K.gradients(y_c, conv_output)[0]
    # norm grads
    grads = grads / tf.reduce_max(grads)
    # also get derivatives of the exp(Y_c) (with a lot more axes to multiply batchwise)
    first = tf.multiply(
        tf.expand_dims(
            tf.expand_dims(tf.expand_dims(K.exp(y_c), axis=-1), axis=-1), axis=-1
        ),
        grads,
    )
    # first = grads
    second = first * grads
    third = second * grads

    # make it a function to be able to get actual values
    model_function = K.function([input_model.input], [conv_output, grads, second, third, y])

    output, grads_val, second_grad, third_grad, y = model_function([images])

    if np.any(output < 0):
        raise Warning(
            "There are some values below 0 in the layer output. "
            + "You should use conv. feature maps after activation."
        )

    results = np.zeros(images.shape[:3])
    width, height = images.shape[1:3]

    for num, (out, grad, grad_exp_2, grad_exp_3) in enumerate(
        zip(output, grads_val, second_grad, third_grad)
    ):

        # out = np.maximum(out, 0)

        # calculate alpha
        # alpha = 1 / (2 + np.sum(out, axis=(0, 1)) * grad + eps)
        alpha_num = grad_exp_2
        alpha_denom = 2 * grad_exp_2 + grad_exp_3 * np.sum(out, axis=(0, 1)) + eps
        # alpha_denom[np.isclose(alpha_denom, 0)] = 1
        alpha = alpha_num / alpha_denom

        # perform the sum to get the weights
        gradient_weights = np.sum(alpha * np.maximum(grad, 0), axis=(0, 1))

        # apply the weights to the feature map
        cam = np.dot(out, gradient_weights)

        # do the ReLU
        cam = np.maximum(cam, 0)

        # resize to the output shape
        cam = resize_image(width, height, cam)

        results[num] = cam

    # norm results
    results = results / np.max(results)

    return y, results


def visualize_map(
    map_img: np.ndarray, img: np.ndarray, label: np.ndarray, pred: np.ndarray
):
    """Visualize the cam images with overlays with and without the input image
    and with outlines of the prediction and ground truth. All input arrays should
    have the same dimensions.

    Parameters
    ----------
    map_img : np.ndarray
        The map that is being visualized
    img : np.ndarray
        The original image
    label : np.ndarray
        The ground truth (as float with values 0 or 1)
    pred : np.ndarray
        The prediction (as float with values between 0 and 1)
    """
    label_contour = find_contour(label.astype(float))
    pred_contour = find_contour(pred)

    _, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(img[..., 0], cmap="gray")
    axes[0].imshow(np.ones(map_img.shape), cmap="bwr", vmin=0, vmax=1, alpha=map_img)
    axes[0].imshow(label_contour, cmap="Wistia", alpha=(label_contour > 0.1).astype(float))
    disable_ticks(axes[0])

    axes[1].imshow(np.ones(map_img.shape), cmap="bwr", vmin=0, vmax=1, alpha=map_img)
    axes[1].imshow(pred_contour, cmap="binary", alpha=(pred_contour > 0.1).astype(float))
    axes[1].imshow(label_contour, cmap="Wistia", alpha=(label_contour > 0.1).astype(float))
    disable_ticks(axes[1])


def find_contour(image: np.ndarray) -> np.ndarray:
    """Find the contour of an image

    Parameters
    ----------
    image : np.ndarray
        The image as 2D numpy array

    Returns
    -------
    np.ndarray
        The contour of the image (using and edge detection filter and normed to 1)
    """
    edge_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    contour_image = ndimage.convolve(image, edge_filter)
    # norm it
    contour_image = contour_image / contour_image.max()
    return contour_image


def disable_ticks(ax_to_disable):
    ax_to_disable.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
