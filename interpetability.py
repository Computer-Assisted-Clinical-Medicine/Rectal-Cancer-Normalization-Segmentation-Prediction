"""
Different functions to help interpret the models.
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import ndimage


def resize_image(width, height, cam):
    return ndimage.zoom(
        input=cam, zoom=(width / cam.shape[0], height / cam.shape[1]), order=2
    )


def grad_cam(
    model: tf.keras.Model,
    images: np.ndarray,
    layer_name: str,
    cls=1,
    pixel_weights=None,
    apply_relu=False,
    smooth=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a GRAD-CAM activation map for a batch of input images. To get the
    maps, all probabilities aver averaged over the image dimensions.
    See also https://arxiv.org/pdf/2002.11434.pdf
    If the result is very noisy, smoothing can also be applied, the gradient
    is then calculated 50 times with added noise to the input.

    Parameters
    ----------
    model : tf.keras.Model
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
    apply_relu : bool, optional
        If a ReLU should be applied to the output of the conv layer, by default False
    smooth : bool, optional
        Smooth the gradients by running it 50 times with gaussian noise with
        15% of the input magnitude, by default False

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

    y: tf.Tensor = model.output

    conv_output = model.get_layer(layer_name).output

    y_c = get_mean_prediction(pixel_weights, y, cls)

    # calculate the gradients with respect to the feature map of the layer
    # this is the partial derivative from the paper (the list has dim. 1)
    grads = K.gradients(y_c, conv_output)[0]

    # make it a function to be able to get actual values
    model_function = K.function([model.input], [conv_output, grads, y])

    output, grads_fm, y = model_function([images])

    if smooth:
        noise_scale = (images.max() - images.min()) * 0.15
        grad_function = K.function(model.input, grads)
        grads_fm = np.zeros_like(grads_fm)
        for _ in range(50):
            noise = np.random.normal(loc=0, scale=noise_scale, size=images.shape)
            grads_fm += grad_function(images + noise)

    if np.any(output < 0) and not apply_relu:
        raise Warning(
            "There are some values below 0 in the layer output. "
            + "You should use conv. feature maps after activation."
        )
    if apply_relu:
        output = np.maximum(0, output)

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

        # apply ReLu again (interpolation can lead to negative values)
        cam = np.maximum(cam, 0)

        results[num] = cam

    # norm results
    results = results / np.max(results)

    return y, results


def grad_cam_plus_plus(
    model: tf.keras.Model,
    images: np.ndarray,
    layer_name: str,
    cls=1,
    pixel_weights=None,
    apply_relu=False,
    smooth=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a GRAD-CAM activation map for a batch of input images. To get the
    maps, all probabilities aver averaged over the image dimensions.
    See also https://arxiv.org/pdf/1710.11063.pdf
    The gradients are normed to a maximum of 1 to prevent numerical instabilities.
    This is done for the whole batch and should only affect the scale of the
    output (which is arbitrary anyway.)
    If the result is very noisy, smoothing can also be applied, the gradients
    are then calculated 50 times with added noise to the input.

    Parameters
    ----------
    model : tf.keras.Model
        The model to use
    images : np.ndarray
        An array of input images with shape (batch, width, height, channels)
    layer_name : str
        The name of the layer for which the output should be used. This should
        be the rectified feature map. (At least according to the authors git repo)
    cls : int, optional
        The class to use, by default 1
    pixel_weights : np.array, optional
        If weights should be apply to the pixels when averaging the label. The
        array should have the same spatial dimensions as the input, by default None
    apply_relu : bool, optional
        If a ReLU should be applied to the output of the conv layer, by default False
    smooth : bool, optional
        Smooth the gradients by running it 50 times with gaussian noise with
        15% of the input magnitude, by default False

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

    y = model.output

    conv_output = model.get_layer(layer_name).output

    y_c = get_mean_prediction(pixel_weights, y, cls)

    # calculate the gradients with respect to the feature map of the layer
    # this is the partial derivative from the paper
    grads = K.gradients(y_c, conv_output)[0]
    # norm grads
    grads = grads / tf.reduce_max(grads)
    # also get derivatives of the exp(Y_c) (with a lot more axes to multiply batch wise)
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
    model_function = K.function([model.input], [conv_output, grads, second, third, y])

    output, grads_val, second_grad, third_grad, y = model_function([images])

    if smooth:
        noise_scale = (images.max() - images.min()) * 0.15
        grad_function = K.function(model.input, [grads, second, third])
        grads_val = np.zeros_like(grads_val)
        second_grad = np.zeros_like(second_grad)
        third_grad = np.zeros_like(third_grad)
        for _ in range(50):
            noise = np.random.normal(loc=0, scale=noise_scale, size=images.shape)
            grad_result = grad_function(images + noise)
            grads_val += grad_result[0]
            second_grad += grad_result[1]
            third_grad += grad_result[2]

    if np.any(output < 0) and not apply_relu:
        raise Warning(
            "There are some values below 0 in the layer output. "
            + "You should use conv. feature maps after activation."
        )
    if apply_relu:
        output = np.maximum(0, output)

    results = np.zeros(images.shape[:3])
    width, height = images.shape[1:3]

    for num, (out, grad, grad_exp_2, grad_exp_3) in enumerate(
        zip(output, grads_val, second_grad, third_grad)
    ):

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

        # apply ReLu again (interpolation can lead to negative values)
        cam = np.maximum(cam, 0)

        results[num] = cam

    # norm results
    results = results / np.max(results)

    return y, results


def gradients(
    model: tf.keras.Model, images: np.ndarray, cls=1, pixel_weights=None, smooth=False
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate input gradients for a batch of input images. To get the
    maps, all probabilities aver averaged over the image dimensions.
    Gradients can be noisy, so enabling smoothing is a good idea.
    See also https://arxiv.org/pdf/1706.03825.pdf for the smooth gradients

    Parameters
    ----------
    model : tf.keras.Model
        The model to use
    images : np.ndarray
        An array of input images with shape (batch, width, height, channels)
    cls : int, optional
        The class to use, by default 1
    pixel_weights : np.array, optional
        If weights should be apply to the pixels when averaging the label. The
        array should have the same spatial dimensions as the input, by default None
    smooth : bool, optional
        Smooth the gradients by running it 50 times with gaussian noise with
        15% of the input magnitude, by default False

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

    y: tf.Tensor = model.output

    y_c = get_mean_prediction(pixel_weights, y, cls)

    # calculate the gradients with respect to the input
    grads = K.gradients(y_c, model.input)[0]

    # make it a function to be able to get actual values
    model_function = K.function([model.input], [grads, y])

    input_grads, y = model_function([images])

    if smooth:
        noise_scale = (images.max() - images.min()) * 0.15
        grad_function = K.function(model.input, grads)
        input_grads = np.zeros_like(input_grads)
        for _ in range(50):
            noise = np.random.normal(loc=0, scale=noise_scale, size=images.shape)
            input_grads += grad_function(images + noise)

    # norm gradients
    input_grads = input_grads / np.max(input_grads)

    return y, input_grads


def get_mean_prediction(pixel_weights: np.ndarray, y: tf.Tensor, cls: int) -> tf.Tensor:
    """Get the mean prediction of the classification pixels if there is a 2D map
    of pixels. Otherwise, just the classification pixel is returned. An array
    of weights of all pixels can also be provided.

    Parameters
    ----------
    pixel_weights : np.ndarray
        The weights of the pixels (doesn't have to be normed)
    y : tf.Tensor
        The labels as a 4D or 2D Tensor (batch, width, height, classes)
    cls : int
        The class to use

    Returns
    -------
    tf.Tensor
        A 1D Tensor which represents the average class of all pixels.
    """
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
    return y_c


def visualize_map(
    map_img: np.ndarray, img: np.ndarray, label: np.ndarray, pred: np.ndarray
) -> Tuple[plt.Figure, np.ndarray]:
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(img[..., 0], cmap="gray")
    axes[0].imshow(np.ones(map_img.shape), cmap="bwr", vmin=0, vmax=1, alpha=map_img)
    axes[0].imshow(label_contour, cmap="Wistia", alpha=(label_contour > 0.1).astype(float))
    disable_ticks(axes[0])

    axes[1].imshow(np.ones(map_img.shape), cmap="bwr", vmin=0, vmax=1, alpha=map_img)
    axes[1].imshow(pred_contour, cmap="binary", alpha=(pred_contour > 0.1).astype(float))
    axes[1].imshow(label_contour, cmap="Wistia", alpha=(label_contour > 0.1).astype(float))
    disable_ticks(axes[1])

    return fig, axes


def visualize_gradients(
    grad_img: np.ndarray, input_img: np.ndarray, normalize=False
) -> Tuple[plt.Figure, np.ndarray]:
    """Visualize the gradients, there are two color images generated for the
    rectified and absolute gradients. Then, for each channel, the gradients and
    the gradients overlayed over the input image are plotted.

    Parameters
    ----------
    grad_img : np.ndarray
        The gradients as 3D array (width, height, channels)
    input_img : np.ndarray
        The input images as 3D array (width, height, channels)
    normalize : bool, optional
        If the input should be normalized per image, by default False
    """
    if normalize:
        # norm the gradients
        maximum = np.quantile(np.abs(grad_img), 0.95)
        grad_img = np.clip(grad_img, a_min=-maximum, a_max=maximum)  # type: ignore
        grad_img = grad_img / maximum

    grad_img_relu = np.maximum(0, grad_img)

    nrows = 1 + grad_img.shape[-1]
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(8 * 2, 8 * nrows))

    axes[0, 0].imshow(grad_img_relu / grad_img_relu.max())
    axes[0, 0].set_title("Rectified Gradients")

    grad_img_abs = np.abs(grad_img)
    axes[0, 1].imshow(grad_img_abs / grad_img_abs.max())
    axes[0, 1].set_title("Absolute Gradients")

    for i, ax in zip(range(grad_img.shape[-1]), axes[1:]):
        ax[0].imshow(input_img[..., i], vmin=0, vmax=1, cmap="gray")
        ax[0].imshow(
            grad_img[..., i], vmin=-1, vmax=1, cmap="bwr", alpha=np.abs(grad_img[..., i])
        )
        ax[0].set_title(f"Channel {i}, normalized gradient with input")

        cbar_im = ax[1].imshow(grad_img[..., i], vmin=-1, vmax=1, cmap="bwr")
        ax[1].set_title(f"Channel {i}, normalized gradient")
        fig.colorbar(cbar_im, ax=ax[1])

    for ax in axes.flat:
        disable_ticks(ax)

    return fig, axes


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
    if np.any(contour_image > 0):
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
