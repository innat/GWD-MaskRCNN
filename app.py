# ------------ tackle some noisy warning
import os
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random

import gdown
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

import mrcnn.model as modellib
from config import WheatDetectorConfig
from config import WheatInferenceConfig
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import log
from utils import get_ax


# for reproducibility
def seed_all(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)


ORIG_SIZE = 1024
seed_all(42)

config = WheatDetectorConfig()
inference_config = WheatInferenceConfig()


def get_model_weight(model_id):
    """Get the trained weights."""
    if not os.path.exists("model.h5"):
        model_weight = gdown.download(id=model_id, quiet=False)
    else:
        model_weight = "model.h5"
    return model_weight


def get_model():
    """Get the model."""
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir="./")
    return model


def load_model(model_id):
    """Load trained model."""
    weight = get_model_weight(model_id)
    model = get_model()
    model.load_weights(weight, by_name=True)
    return model


def prepare_image(image):
    """Prepare incoming sample."""
    image = image[:, :, ::-1]
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]

    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)

    resized_image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE,
    )

    return resized_image


def predict_fn(image):

    image = prepare_image(image)

    model = load_model(model_id="1k4_WGBAUJCPbkkHkvtscX2jufTqETNYd")
    results = model.detect([image])
    r = results[0]
    class_names = ["Wheat"] * len(r["rois"])

    image = visualize.display_instances(
        image,
        r["rois"],
        r["masks"],
        r["class_ids"],
        class_names,
        r["scores"],
        ax=get_ax(),
        title="Predictions",
    )

    return image[:, :, ::-1]

title="Global Wheat Detection with Mask-RCNN Model"
description="<strong>Model</strong>: Mask-RCNN. <strong>Backbone</strong>: ResNet-101. Trained on: <a href='https://www.kaggle.com/competitions/global-wheat-detection/overview'>Global Wheat Detection Dataset (Kaggle)</a>. </br>The code is written in <code>Keras (TensorFlow 1.14)</code>. One can run the full code on Kaggle: <a href='https://www.kaggle.com/code/ipythonx/keras-global-wheat-detection-with-mask-rcnn'>[Keras]:Global Wheat Detection with Mask-RCNN</a>"
article = "<p>The model received <strong>0.6449</strong> and <strong>0.5675</strong> mAP (0.5:0.75:0.05) on the public and private test dataset respectively. The above examples are from test dataset without ground truth bounding box. Details: <a href='https://www.kaggle.com/competitions/global-wheat-detection/data'>Global Wheat Dataset</a></p>"

iface = gr.Interface(
    fn=predict_fn,
    inputs=gr.inputs.Image(label="Input Image"),
    outputs=gr.outputs.Image(label="Prediction"),
    title=title,
    description=description,
    article=article,
    examples=[
        ["examples/2fd875eaa.jpg"],
        ["examples/51b3e36ab.jpg"],
        ["examples/51f1be19e.jpg"],
        ["examples/53f253011.jpg"],
        ["examples/348a992bb.jpg"],
        ["examples/796707dd7.jpg"],
        ["examples/aac893a91.jpg"],
        ["examples/cb8d261a3.jpg"],
        ["examples/cc3532ff6.jpg"],
        ["examples/f5a1f0358.jpg"],
    ],
)
iface.launch()
