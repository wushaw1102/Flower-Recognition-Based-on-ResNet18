"""Fine-tuning ResNet-18 on 102 flower classes.

Modules
-------
config      paths, constants, checkpoint resolution
data        transforms, ImageFolder wiring, demo index
model       ResNet-18 construction and checkpoint loading
engine      training / evaluation loops, two-stage schedule
metrics     accuracy, top-k, per-class report, confusion matrix
train       CLI: two-stage fine-tuning
evaluate    CLI: metrics bundle on a labelled folder
predict     CLI: single image or folder inference
visualize   figures used by the README
"""

__version__ = "1.1.0"
