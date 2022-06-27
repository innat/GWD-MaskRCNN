from mrcnn.config import Config


class WheatDetectorConfig(Config):
    # Give the configuration a recognizable name
    NAME = "wheat"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    BACKBONE = "resnet101"
    NUM_CLASSES = 2
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    STEPS_PER_EPOCH = 120
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 0.0005
    TRAIN_ROIS_PER_IMAGE = 350
    DETECTION_MIN_CONFIDENCE = 0.60
    VALIDATION_STEPS = 60
    MAX_GT_INSTANCES = 500
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.0,
        "rpn_bbox_loss": 1.0,
        "mrcnn_class_loss": 1.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.0,
    }


class WheatInferenceConfig(WheatDetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
