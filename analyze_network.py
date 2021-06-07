# pylint:disable=pointless-string-statement

# %% [markdown]
'''
# Analyze network
## imports and definitions
'''

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import seaborn as sns
import SimpleITK as sitk
import tensorflow as tf

from SegmentationNetworkBasis.NetworkBasis import loss
from SegmentationNetworkBasis.NetworkBasis.metric import Dice

# %% [markdown]
'''
## load network
'''

EXP = 0
FOLD = 4

data_dir = Path(os.environ['data_dir'])
experiment_dir = Path(os.environ['experiment_dir'])

hparam_file = experiment_dir / 'hyperparameters.csv'
hparams = pd.read_csv(hparam_file, sep=';')
experiment_path = Path(hparams.loc[EXP, 'path'])
model_path = experiment_path / f'fold-{FOLD}' / 'models' / 'model-best'

model = tf.keras.models.load_model(
    model_path,
    compile=False,
    custom_objects={'Dice' : Dice}
)

# %% [markdown]
'''
## load image and generate example data from it
'''
preprocessed_path = experiment_dir / 'data_preprocessed' / 'pre_QUANTILE_resampled'
preprocessed_image = preprocessed_path / 'sample-1001_1.mhd'
labels_path = preprocessed_path / 'label-1001_1.mhd'

image = sitk.GetArrayFromImage(sitk.ReadImage(str(preprocessed_image)))
labels = sitk.GetArrayFromImage(sitk.ReadImage(str(labels_path)))

input_shape = model.input.get_shape().as_list()

assert image.shape[-1] == input_shape[-1]

start = (np.array(image.shape[1:3]) - input_shape[1:3]) // 2
stop = np.array(image.shape[1:3]) - input_shape[1:3] - start
Z_START = 22
z_stop = Z_START + input_shape[0]

input_image_np = image[Z_START:z_stop,start[0]:-stop[0],start[1]:-stop[1]]
input_labels_np = labels[Z_START:z_stop,start[0]:-stop[0],start[1]:-stop[1]]

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for img, lbl, ax in zip(input_image_np[...,0], input_labels_np, axes.flat):

    img_u8 = np.uint8((img-img.min())/(img.max()-img.min())*255)

    pil_image = PIL.Image.fromarray(img_u8).convert('RGB')
    pil_image.putalpha(255)
    label_red = np.pad(np.expand_dims(lbl, axis=2), pad_width=((0,0),(0,0),(0,2)))
    label_image = PIL.Image.fromarray(np.uint8(label_red*255)).convert('RGB')
    label_image_alpha = PIL.Image.fromarray(np.uint8(lbl*100)).convert('L')
    label_image.putalpha(label_image_alpha)
    # color it red
    pil_image_res = PIL.Image.alpha_composite(pil_image, label_image)

    ax.imshow(pil_image_res)
    ax.axis('off')

plt.suptitle('Images used as input')
plt.tight_layout()
plt.show()
plt.close()

# %% [markdown]
'''
## Run it with random input

This creates random tensors as input and output and calculates the gradients.
'''

input_tf = tf.convert_to_tensor(input_image_np)
labels_one_hot = np.squeeze(np.eye(2)[input_labels_np.flat]).reshape(input_labels_np.shape + (-1,))
labels_tf = tf.convert_to_tensor(labels_one_hot)

model_loss = loss.dice_loss

def watch_layer(layer, tape):
    """
    Make an intermediate hidden `layer` watchable by the `tape`.
    After calling this function, you can obtain the gradient with
    respect to the output of the `layer` by calling:

        grads = tape.gradient(..., layer.result)

    from https://stackoverflow.com/a/56567364
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Store the result of `layer.call` internally.
            layer.result = func(*args, **kwargs)
            # From this point onwards, watch this tensor.
            tape.watch(layer.result)
            # Return the result to continue with the forward pass.
            return layer.result
        return wrapper
    layer.call = decorator(layer.call)
    return layer

grad_tape = tf.GradientTape(persistent=True)
with grad_tape:
    watch_layer(model.layers[-2], grad_tape)  
    probabilities = model(input_tf)
    loss_batch = model_loss(y_true=labels_tf, y_pred=probabilities)

# do backpropagation
gradients = grad_tape.gradient(loss_batch, model.trainable_weights)
# get loss gradient
loss_grad = grad_tape.gradient(loss_batch, probabilities)
# pre softmax gradient
loss_grad_softmax = grad_tape.gradient(loss_batch, model.layers[-2].result)  

if np.all([np.all(np.isclose(grad.numpy(), 0)) for grad in gradients]):
    print('All values in the hard DICE loss gradient are 0.')

# %% [markdown]
'''
## Analyze the last layer
'''

sns.histplot(loss_grad.numpy().reshape(-1, 2))
plt.xlabel('gradient')
plt.title('Gradients of the predictions')
plt.show()
plt.close()

sns.histplot(probabilities.numpy().reshape(-1, 2))
plt.xlabel('probability of class')
plt.title('Output probabilities')
plt.show()
plt.close()

last_layer = model.layers[-2]
sns.histplot(last_layer.result.numpy().reshape(-1, 2))
plt.title('Output of last layer (before softmax)')
plt.show()
plt.close()
print(f'Minimum Background: {last_layer.result.numpy()[...,0].min():.1f}')
print(f'Maximum Foreground: {last_layer.result.numpy()[...,1].max():.1f}')

sns.histplot(last_layer.kernel.numpy().reshape(-1))
plt.xlabel('Kernel')
plt.title('Kernel of last layer')
plt.show()
plt.close()

sns.histplot(gradients[-1].numpy().reshape(-1))
plt.xlabel('Gradient')
plt.title('Gradients of last layer')
plt.show()
plt.close()


# %% [markdown]
'''
## Analyze all layers
'''

for layer_grad, weights in zip(gradients, model.trainable_weights):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    if 'bias' in weights.name:
        N_BINS = 20
    else:
        N_BINS = 50

    ax = axes[0]
    sns.histplot(layer_grad.numpy().reshape(-1), kde=True, bins=N_BINS, ax=ax)
    ax.set_title('Gradients')

    ax = axes[1]
    sns.histplot(weights.numpy().reshape(-1), kde=True, bins=N_BINS, ax=ax)
    ax.set_title('Kernel')

    plt.suptitle(weights.name)
    plt.tight_layout()
    plt.show()
    plt.close()
