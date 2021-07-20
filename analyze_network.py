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

import interpetability
from SegmentationNetworkBasis.NetworkBasis import loss
from SegmentationNetworkBasis.NetworkBasis.metric import Dice

tf.python.framework.ops.disable_eager_execution()

# %% [markdown]
"""
## load network
"""

EXP = 3
FOLD = 0
FOREGROUND = 1

data_dir = Path(os.environ["data_dir"])
experiment_dir = Path(os.environ["experiment_dir"])
experiment_dir = experiment_dir.parent / "Good_Models"  # TODO: remove

hparam_file = experiment_dir / "hyperparameters.csv"
hparams = pd.read_csv(hparam_file, sep=";")
experiment_path = Path(hparams.loc[EXP, "path"])
model_path = experiment_dir / experiment_path / f"fold-{FOLD}" / "models" / "model-best"

custom = {"Dice": Dice, "dice_loss": loss.dice_loss}
model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom)

if model.name == "DeepLabv3plus":
    LAYER = "pred-conv1/act"
    backbone = hparams.loc[EXP, "backbone"]
elif model.name == "UNet":
    LAYER = "UNet2D/last/Conn2D"

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
### Average over all pixels
"""

pred, gradcam = interpetability.grad_cam(model, input_image_np, LAYER, apply_relu=True)

for gradcam_img, input_img, label_img, pred_img in zip(
    gradcam, input_image_np, input_labels_np, pred[..., 1]
):

    interpetability.visualize_map(gradcam_img, input_img, label_img, pred_img)
    plt.show()
    plt.close()

# %% [markdown]
"""
### Do Grad-CAM over the block output layers
"""

layers = []
if model.name == "DeepLabv3plus":
    if backbone == "densenet121":
        layers += ["conv1/relu", "conv2_block1_concat"]
        layers += [f"pool{i}_conv" for i in range(2, 5)]
    elif backbone == "mobilenet_v2":
        layers += [
            "Conv1",
            "block_1_project_BN",
            "block_2_add",
            "block_4_add",
            "block_7_add",
            "block_11_add",
            "block_13_project_BN",
        ]
    elif backbone == "resnet50":
        layers += [
            "conv1_conv",
            "conv2_block1_out",
            "conv3_block1_preact_bn",
            "conv4_block1_preact_bn",
            "conv5_block1_out",
        ]
    else:
        raise ValueError(f"BAckbone {backbone} not recognized.")
    layers += [
        "low-level-reduction/bn",
        "high-feature-red/bn",
        "pred-conv0/bn",
        "pred-conv1/bn",
        "logits",
    ]

    use_images = slice(2, 3)
elif model.name == "UNet":
    layers += ["UNet2D-encode0/basic/conv0_act"]
    layers += [f"UNet2D-encode{i}/basic/add" for i in range(4)]
    layers += ["UNet2D-bottleneck/add"]
    layers += [f"UNet2D-decode{i}/basic/add" for i in range(5, 9)]
    layers += ["UNet2D/last/Conn2D"]

    use_images = slice(7, 8)

for layer in layers:
    print(layer)
    pred, gradcam = interpetability.grad_cam(model, input_image_np, layer, apply_relu=True)

    for gradcam_img, input_img, label_img, pred_img in zip(
        gradcam[use_images],
        input_image_np[use_images],
        input_labels_np[use_images],
        pred[use_images, ..., 1],
    ):
        gradcam_img = gradcam_img / gradcam_img.max()
        interpetability.visualize_map(gradcam_img, input_img, label_img, pred_img)
        plt.suptitle(layer)
        plt.tight_layout()
        # plt.savefig(layer.replace("/", "-") + ".png")
        plt.show()
        plt.close()


# %% [markdown]
"""
## Do smooth Grad-CAM
### Average over all pixels
"""

_, gradcam_smooth = interpetability.grad_cam(
    model, input_image_np, LAYER, apply_relu=True, smooth=True
)

for gradcam_img, input_img, label_img, pred_img in zip(
    gradcam_smooth, input_image_np, input_labels_np, pred[..., 1]
):

    interpetability.visualize_map(gradcam_img, input_img, label_img, pred_img)
    plt.show()
    plt.close()

# %% [markdown]
"""
### Average over wrongly classified images
"""

pixel_weights_fp = np.abs(input_labels_np - (pred[..., 1] > 0.5).astype(float))

_, gradcam_f = interpetability.grad_cam(
    model=model,
    images=input_image_np,
    layer_name=LAYER,
    pixel_weights=pixel_weights_fp,
    apply_relu=True,
)


for gradcam_img, gradcam_orig, input_img, label_img, pred_img in zip(
    gradcam_f, gradcam, input_image_np, input_labels_np, pred[..., 1]
):

    interpetability.visualize_map(gradcam_img, input_img, label_img, pred_img)
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(gradcam_img - gradcam_orig)
    plt.title("Difference to original Grad-CAM with all pixels.")
    plt.colorbar()
    plt.show()
    plt.close()


# %% [markdown]
"""
## Do Grad-CAM++
"""

pred_plus, gradcam_plus = interpetability.grad_cam_plus_plus(
    model, input_image_np, LAYER, apply_relu=True
)

for gradcam_img, input_img, label_img, pred_img in zip(
    gradcam_plus, input_image_np, input_labels_np, pred_plus[..., 1]
):

    interpetability.visualize_map(gradcam_img, input_img, label_img, pred_img)
    plt.show()
    plt.close()

# %% [markdown]
"""
### Do Grad-CAM++ over the block output layers
"""

for layer in layers:
    print(layer)
    pred, gradcam_plus = interpetability.grad_cam_plus_plus(
        model, input_image_np, layer, apply_relu=True
    )

    for gradcam_img, input_img, label_img, pred_img in zip(
        gradcam_plus[use_images],
        input_image_np[use_images],
        input_labels_np[use_images],
        pred[use_images, ..., 1],
    ):
        gradcam_img = gradcam_img / gradcam_img.max()
        interpetability.visualize_map(gradcam_img, input_img, label_img, pred_img)
        plt.suptitle(layer)
        plt.tight_layout()
        # plt.savefig(layer.replace("/", "-") + ".png")
        plt.show()
        plt.close()

# %% [markdown]
"""
### Compare GradCam and GradCam++
"""

label = input_labels_np[use_images][0]
label_contour = interpetability.find_contour(label.astype(float))
img = input_image_np[use_images][0]

for layer in layers:
    print(layer)
    pred, gradcam = interpetability.grad_cam(model, input_image_np, layer, apply_relu=True)
    _, gradcam_plus = interpetability.grad_cam_plus_plus(
        model, input_image_np, layer, apply_relu=True
    )

    # get the images
    pred_contour = interpetability.find_contour(pred[use_images, ..., 1][0])
    gradcam = gradcam[use_images][0]
    gradcampp = gradcam_plus[use_images][0]

    # norm them
    gradcam = gradcam / gradcam.max()
    gradcampp = gradcampp / gradcam.max()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, map_img in zip(axes, [gradcam, gradcampp]):

        ax.imshow(img[..., 0], cmap="gray")
        ax.imshow(np.ones(map_img.shape), cmap="bwr", vmin=0, vmax=1, alpha=map_img)
        ax.imshow(label_contour, cmap="Wistia", alpha=(label_contour > 0.1).astype(float))
        interpetability.disable_ticks(ax)

    axes[0].set_title("GradCam")
    axes[1].set_title("GradCam++")

    plt.suptitle(layer)
    plt.tight_layout()
    plt.savefig(layer.replace("/", "-") + ".png")
    plt.show()
    plt.close()

# %% [markdown]
"""
## Do Gradients
### Make Smooth Gradients
The gradients are normalized for each image when generating the visualization,
one image will dominate the whole batch otherwise.
"""

_, smooth_grads = interpetability.gradients(model, input_image_np, smooth=True)

for num, (grad_img, input_img) in enumerate(zip(smooth_grads, input_image_np)):

    print(f"\nSlice {num}:\n")

    print("Mean Absolute values:")
    mean_abs = np.mean(np.abs(grad_img), axis=(0, 1))
    mean_abs_norm = mean_abs / mean_abs.sum()
    for i in range(grad_img.shape[-1]):
        print(f"Channel {i}: {mean_abs[i]:.3f} ({mean_abs_norm[i]*100:4.1f} %)")
    print("Mean Rectified values:")
    mean_relu = np.mean(np.maximum(0, grad_img), axis=(0, 1))
    mean_relu_norm = mean_relu / mean_relu.sum()
    for i in range(grad_img.shape[-1]):
        print(f"Channel {i}: {mean_relu[i]:.3f} ({mean_relu_norm[i]*100:4.1f} %)")

    interpetability.visualize_gradients(grad_img, input_img, normalize=True)
    plt.show()
    plt.close()

print("\nAll Slices:\n")

print("Mean Absolute values:")
mean_abs = np.mean(np.abs(smooth_grads), axis=(0, 1, 2))
mean_abs_norm = mean_abs / mean_abs.sum()
for i in range(smooth_grads.shape[-1]):
    print(f"Channel {i}: {mean_abs[i]:.3f} ({mean_abs_norm[i]*100:4.1f} %)")
print("Mean Rectified values:")
mean_relu = np.mean(np.maximum(0, smooth_grads), axis=(0, 1, 2))
mean_relu_norm = mean_relu / mean_relu.sum()
for i in range(smooth_grads.shape[-1]):
    print(f"Channel {i}: {mean_relu[i]:.3f} ({mean_relu_norm[i]*100:4.1f} %)")

# %% [markdown]
"""
### Gradients for false positives
"""

_, smooth_grads_fp = interpetability.gradients(
    model=model, images=input_image_np, smooth=True, pixel_weights=pixel_weights_fp
)

for num, (grad_img_fp, grad_img, input_img, pix_weights) in enumerate(
    zip(smooth_grads_fp, smooth_grads, input_image_np, pixel_weights_fp)
):

    # skip slices without false positives
    if np.all(np.isclose(pix_weights, 0)):
        print(f"\nSlice {num} does not contains FPs\n")
        continue

    print(f"\nSlice {num}:\n")

    # normalize using the original gradients
    maximum = np.quantile(np.abs(grad_img), 0.95)
    grad_img_fp = np.clip(grad_img_fp, a_min=-maximum, a_max=maximum)
    grad_img_fp = grad_img_fp / maximum

    fig, axes = interpetability.visualize_gradients(grad_img_fp, input_img)
    for ax in axes.flat:
        ax.imshow(pix_weights, alpha=pix_weights * 0.7, cmap="gray")
    fig.suptitle("Gradients of False Positives")
    plt.show()
    plt.close()

    print()
    print()

print("Mean Absolute values:")
mean_abs = np.mean(np.abs(smooth_grads_fp), axis=(0, 1, 2))
mean_abs_norm = mean_abs / mean_abs.sum()
for i in range(smooth_grads_fp.shape[-1]):
    print(f"Channel {i}: {mean_abs[i]:.3f} ({mean_abs_norm[i]*100:4.1f} %)")
print("Mean Rectified values:")
mean_relu = np.mean(np.maximum(0, smooth_grads_fp), axis=(0, 1, 2))
mean_relu_norm = mean_relu / mean_relu.sum()
for i in range(smooth_grads_fp.shape[-1]):
    print(f"Channel {i}: {mean_relu[i]:.3f} ({mean_relu_norm[i]*100:4.1f} %)")

# %% [markdown]
"""
### Combination of smooth gradients with grad-CAM
Gradients are combined with grad-CAM by using elementwise multiplication.
"""

for grad_img, gradcam_img, input_img in zip(smooth_grads, gradcam_plus, input_image_np):

    interpetability.visualize_gradients(
        (grad_img.T * gradcam_img.T).T, input_img, normalize=True
    )
    plt.tight_layout()
    plt.show()
    plt.close()

# %% [markdown]
"""
## Run the model with the images as input
"""

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
