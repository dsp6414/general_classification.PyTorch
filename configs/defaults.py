# coding : utf-8

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()

_C.INPUT.IN_CHANNELS = 3
# Size of the image during training and testing
_C.INPUT.IMG_SIZE = 224
# Scale range for the image during training
_C.INPUT.SCALE_TRAIN = (0.5, 1.0)
# Center crop probability during test
_C.INPUT.CROP_PCT_TEST = 1.0

# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = (0.485, 0.456, 0.406)
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = (0.229, 0.224, 0.225)
# Convert image to BGR format, in range 0-255
_C.INPUT.TO_BGR255 = False

# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0

# Flips
_C.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.5
_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0

# Random erase during training
_C.INPUT.RANDOM_ERASE_PROB = 0.0
_C.INPUT.RANDOM_ERASE_MODE = 'const'

# Auto augment
_C.INPUT.AUTO_AUGMENT = 'auto'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ""
_C.DATASETS.TEST = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = 'resnet50'
# Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")
_C.MODEL.GLOBAL_POOL = 'avg'
# Dropout rate (default: 0.)
_C.MODEL.DROP_RATE = 0.0
# Drop connect rate (default: 0.)
_C.MODEL.DROP_CONNECT = 0.0
_C.MODEL.NUM_CLASSES = 10

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.BATCH_SIZE = 100

_C.SOLVER.OPTIMIZER = "sgd"

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MIN_LR = 0.00001
# LR scheduler (default: "step")
_C.SOLVER.SCHEDULER = 'step'
# Epoch interval to decay LR for StepLR
_C.SOLVER.DECAY_EPOCHS = 10
# LR decay rate (default: 0.1)
_C.SOLVER.DECAY_RATE = 0.1
# Epochs to cooldown LR at min_lr, after cyclic schedule ends
_C.SOLVER.COOLDOWN_EPOCHS = 10
# Warmup learning rate (default: 0.0001)
_C.SOLVER.WARMUP_LR = 0.0001
# Epochs to warmup LR, if scheduler supports
_C.SOLVER.WARMUP_EPOCHS = 3

_C.SOLVER.LABEL_SMOOTHING = 0.0

# Enable tracking moving average of model weights
_C.SOLVER.EMA = False
# Decay factor for model weights moving average (default: 0.9998)
_C.SOLVER.EMA_DECAY = 0.9998
# Best metric (default: "prec1")
_C.SOLVER.EVAL_METRIC = 'prec1'

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""