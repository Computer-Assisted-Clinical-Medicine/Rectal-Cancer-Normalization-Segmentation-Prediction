# pylint:disable=pointless-string-statement

# %% [markdown]
"""
# Analyze network
## imports and definitions
"""

import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import seaborn as sns
import SimpleITK as sitk
import tensorflow as tf
import tensorflow.keras.backend as K

from interpetability import grad_cam, grad_cam_plus_plus, visualize_map
from SegmentationNetworkBasis.NetworkBasis import loss
from SegmentationNetworkBasis.NetworkBasis.metric import Dice

tf.python.framework.ops.disable_eager_execution()

# %% [markdown]
"""
## load network
"""

EXP = 0
FOLD = 0
FOREGROUND = 1

LAYER = "pred-conv1/act"

data_dir = Path(os.environ["data_dir"])
experiment_dir = Path(os.environ["experiment_dir"])

hparam_file = experiment_dir / "hyperparameters.csv"
hparams = pd.read_csv(hparam_file, sep=";")
experiment_path = Path(hparams.loc[EXP, "path"])
model_path = experiment_path / f"fold-{FOLD}" / "models" / "model-best"

custom = {"Dice": Dice, "dice_loss": loss.dice_loss}
model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom)

# %% [markdown]
"""
## load image and generate example data from it
"""

preprocessed_path = experiment_dir / "data_preprocessed" / "pre_QUANTILE_resampled"
preprocessed_image = preprocessed_path / "sample-1003_1.mhd"
labels_path = preprocessed_path / "label-1003_1.mhd"

mri_image = sitk.GetArrayFromImage(sitk.ReadImage(str(preprocessed_image)))
labels = sitk.GetArrayFromImage(sitk.ReadImage(str(labels_path)))

input_shape = np.array(model.input.get_shape().as_list())

# pad if the image is too small
if np.any(input_shape[1:3] >= mri_image.shape[1:3]):
    pad = np.max(input_shape[1:3] - mri_image.shape[1:3]) // 2 + 2
    mri_image = np.pad(mri_image, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    labels = np.pad(labels, ((0, 0), (pad, pad), (pad, pad)))

assert mri_image.shape[-1] == input_shape[-1]

start = (np.array(mri_image.shape[1:3]) - input_shape[1:3]) // 2
stop = np.array(mri_image.shape[1:3]) - input_shape[1:3] - start
z_center = int(
    np.round(
        np.average(np.arange(mri_image.shape[0]), weights=np.sum(labels != 0, axis=(1, 2)))
    )
)
z_start = z_center - input_shape[0] // 2
z_stop = z_start + input_shape[0]

input_image_np = mri_image[z_start:z_stop, start[0] : -stop[0], start[1] : -stop[1]]
input_labels_np = labels[z_start:z_stop, start[0] : -stop[0], start[1] : -stop[1]]

nrows = input_shape[0] // 4
fig, axes = plt.subplots(nrows, 4, figsize=(10, 2.5 * nrows))

for img, lbl, ax in zip(input_image_np[..., 0], input_labels_np, axes.flat):

    img_u8 = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)

    pil_image = PIL.Image.fromarray(img_u8).convert("RGB")
    pil_image.putalpha(255)
    label_red = np.pad(np.expand_dims(lbl, axis=2), pad_width=((0, 0), (0, 0), (0, 2)))
    label_image = PIL.Image.fromarray(np.uint8(label_red * 255)).convert("RGB")
    label_image_alpha = PIL.Image.fromarray(np.uint8(lbl * 100)).convert("L")
    label_image.putalpha(label_image_alpha)
    # color it red
    pil_image_res = PIL.Image.alpha_composite(pil_image, label_image)

    ax.imshow(pil_image_res)
    ax.axis("off")

plt.suptitle("Images used as input")
plt.tight_layout()
plt.show()
plt.close()


# %% [markdown]
"""
## Do Grad-CAM
"""

pred, gradcam = grad_cam(model, input_image_np, LAYER)

for gradcam_img, input_img, label_img, pred_img in zip(
    gradcam, input_image_np, input_labels_np, pred[..., 1]
):

    visualize_map(gradcam_img, input_img, label_img, pred_img)
    plt.show()
    plt.close()


# %% [markdown]
"""
## Do Grad-CAM++
"""

pred, gradcam = grad_cam_plus_plus(model, input_image_np, LAYER)

for gradcam_img, input_img, label_img, pred_img in zip(
    gradcam, input_image_np, input_labels_np, pred[..., 1]
):

    visualize_map(gradcam_img, input_img, label_img, pred_img)
    plt.show()
    plt.close()


# %% [markdown]
"""
## Run the model with the images as input
"""

tf.config.run_functions_eagerly(True)

input_tf = tf.convert_to_tensor(input_image_np)
labels_one_hot = np.squeeze(np.eye(2)[input_labels_np.flat]).reshape(
    input_labels_np.shape + (-1,)
)
labels_tf = tf.convert_to_tensor(labels_one_hot)

model_loss = loss.dice_loss

probabilities_graph = model.layers[-1].output
loss_batch = model_loss(y_true=labels_tf, y_pred=probabilities_graph)

model_function = K.function(
    [model.input],
    [
        probabilities_graph,
        K.gradients(loss_batch, model.trainable_weights),
        K.gradients(loss_batch, probabilities_graph)[0],
        K.gradients(loss_batch, model.layers[-2].output)[0],
        model.layers[-2].output,
        model.trainable_weights,
    ],
)

(
    probabilities,
    gradients,
    loss_grad,
    loss_grad_softmax,
    last_layer_result,
    trainable_weights,
) = model_function([input_image_np])

if np.all([np.all(np.isclose(grad, 0)) for grad in gradients]):
    print("All values in the hard DICE loss gradient are 0.")


# %% [markdown]
"""
## Analyze the last layer
"""

hist = partial(sns.histplot, log_scale=(False, True))

hist(loss_grad.reshape(-1, 2))
plt.xlabel("gradient")
plt.title("Gradients of the predictions")
plt.show()
plt.close()

hist(probabilities.reshape(-1, 2))
plt.xlabel("probability of class")
plt.title("Output probabilities")
plt.show()
plt.close()

last_layer = model.layers[-2]
hist(last_layer_result.reshape(-1, 2))
plt.title("Output of last layer (before softmax)")
plt.show()
plt.close()
print(f"Minimum Background: {last_layer_result[...,0].min():.1f}")
print(f"Maximum Foreground: {last_layer_result[...,1].max():.1f}")

hist(trainable_weights[-1].reshape(-1))
plt.xlabel("Kernel")
plt.title("Kernel of last layer")
plt.show()
plt.close()

hist(gradients[-1].reshape(-1))
plt.xlabel("Gradient")
plt.title("Gradients of last layer")
plt.show()
plt.close()


# %% [markdown]
"""
## Analyze all layers
"""

weight_names = [weights.name for weights in model.trainable_weights]
for layer_grad, weights, name in zip(gradients, trainable_weights, weight_names):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    if "bias" in name:
        N_BINS = 20
    else:
        N_BINS = 50

    ax = axes[0]
    sns.histplot(layer_grad.reshape(-1), kde=True, bins=N_BINS, ax=ax)
    ax.set_title("Gradients")

    ax = axes[1]
    sns.histplot(weights.reshape(-1), kde=True, bins=N_BINS, ax=ax)
    ax.set_title("Kernel")

    plt.suptitle(name)
    plt.tight_layout()
    plt.show()
    plt.close()

    break
