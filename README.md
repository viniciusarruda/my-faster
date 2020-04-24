
---
# ------------------------------------------------------------------------
# Reminders
# ------------------------------------------------------------------------
- For each modification, test and do not move until obtain the same anterior result (or better).

---
# ------------------------------------------------------------------------
# Notes about the code
# ------------------------------------------------------------------------

## About the image standard
- Always the image and features are with the last two axis with the horizontal axis being the width and
    the vertical being the heigh. This means that if it is a representation:
    - Torch tensor or numpy array: [batch_size, channels, height, width] (you can think as [batch_size, channels, n_rows, n_cols])
- The (0, 0) point is at the top left corner of the image

## Other standard
- The format of bounding box is in [x_0, y_0, x_1, y_1]. X and Y lays on the horizontal and vertical axis, respectively.
- The format of the offset_box is in [x_0, y_0, width, height] (you can think as [x_0, y_0, n_cols, n_rows]).
- (x_0, y_0) is the top-left corner and (x_1, y_1) is the bottom-right corner.

---
# ------------------------------------------------------------------------
# TODO
# ------------------------------------------------------------------------
- [x] Assert the forward pass
- [x] Implement the loss and assert its corectness
- [x] Implement the backward and assert its correctness
- [x] Make some tests with the existing dataset to understand where I am
- [x] Use the newest PyTorch version (1.4 stable)
- [x] Use the built-in RoIAlign (do not delete the old, just comment or something similar)
- [x] Modify NMS to output the same as the ops.nms() (make a hard unit test)
        -> In fact, it behaves slightly different when handling degenerate boxes (not an error, see my github issues I've opened), but nothing that gets in the way of performance.
- [x] Use the built-in NMS (do not delete the old, just comment or something similar)
- [x] Check these (from https://github.com/aleju/papers/blob/master/neural-nets/Faster_R-CNN.md):
    - [x] Positive examples are anchor boxes that have an IoU with a ground truth bounding box of 0.7 or more. If no anchor  point has such an IoU with a specific box, the one with the highest IoU is used instead.
    - [x] They use 128 positive examples and 128 negative ones. If they can't come up with 128 positive examples, they add more negative ones.
    - [x] Look the test topic and also check if I am doing right
    - [x] Balance the training also for the regressor.
- [x] Remove the batch loop, since it is only one image (to add more images in the batch later, but I think I wont)
    - [x] remove batch channel of bbox_utils.py 
    - [x] remove batch channel of _anchors2proposals() function
    - [x] check if output is the same before and after the modification
- [ ] Fix valid anchors
    - [x] Implement visualization of the center of each anchor and valid anchor.
    - [x] Check if it is correct, the center of each valid anchor is not centralized.. why?
        - [x] Understand the _get_anchor_parameters() code
        - [x] Read that issue aboud computing anchor parameters
        - [x] Show the center of anchors when plotting positive anchors. Also interesting drawing the center of all anchors in a "background" color to see how it would be like. (this last was not done but satisfied)
        - [x] Understand why the result is not centralized
    - [future] Consider its center not the whole region (actually compare.. because the original code considers what is current implemented)
        - [future] Plot the all anchors cliping to the image size
                   These features are attempts to improve the faster. Your goal is to replicate the results.
                   At a later time this will be investigated, but not now.
    - [x] Do I need to keep all anchors or only the valid ones through the code? No, I removed.
        -[future] if removed.. the feature map can be reduced thus reducing computational processing?
- [x] Enable images with aspect ration different than 1 (and fix the comments accordingly)
    - [x] Check if the current version is giving the same output of the last version for a squared image
        - Actually, I discovered a bug in the previous version. Now it is fixed!
- [x] Clean/comment/review loss.py file (check to remove duplicata as calculating the iou)
    - [future] Comments and a deeper clean have to be made when everything is done
- [x] Improve the visualization including verbose information for debugging (Tensorboard)
    - [x] Training loss
    - [x] compare the visualizations before the editing and after
    - [x] add final loss
          https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/utils/visualization.py
    - [x] try to draw class name and prob on bbox
    - [x] Keep track of some images (e.g. 10) to plot their bboxes through the training iters/epochs
    - [x] Show all see_visualization stuff to the tensorboard.
        - [x] If good, delete the old. If so-so, check if I can download the content of the tensorboard, if so, delete the old. If bad, keep the old fashion way. -> Both kept
    - [near-future] Include a heatmap visualization of the features right before the RPN (maybe an insight to use attention here)
    - [near-future] Training mAP and AP for each class throught the training iters/epochs
    - [near-future] Show anchors and positive anchors along with its gt (just once)
    - [near-future] Training precision - recall ? (To get the feeling of it)
- [x] Single image with two (and then more) objects
    - [x] Nao vai dar pra comparar como antes devido a um bug.. a n ser que eu deixe com o bug e as classes
          tentar isso, deixar o bug e com classes para ver se da o mesmo resultado.. deveria!
          R: Deu o mesmo resultado deixando o bug..
- [x] Show results of epoch and iter 0, i.e., just with random weights
- [x] Images with different sizes and aspect ratios
    - [impracticable] Show anchors (just once) for each image in tensorboard
    - [x] Dataset with a single size but non-square (e.g., Cityscapes)
    - [near-future] Dataset which images are different in size (e.g., ?) (I should rely on the DataLoader)
- [x] RCNN_top
    - [x] make a function for the top+cls+reg+view/mean in the backbone (specific ones)
- [x] Check this class agnostic stuff
    - What this changes: Each class has its own bbox regression, but only the true labeled ones
      are considered in the loss, background also is ignored but considered for the classification
- [x] Check to get laveraged from ignore_index on cross entropy loss
------ hoje vir ate aqui
- [ ] GPU compatibility (To start to train in a large dataset and debug) 
- [ ] Gerar novo baseline para comparar.
    - [ ] Uma classe
    - [ ] Duas classes
    - [ ] Tres classes
    - [ ] Cinco classes
    - Todas devem ser checadas apos cada modificacao da rede para ver a consistencia.
- [ ] Check what should happen to the loss when there is no bbox to the second stage.
    There is an exit if enters in this condition. So a hard test case will help to find a case where enters in this exit and help to debug.
- [ ] Overfit some data.. check instability.. if persist, compare the overfiting against another Faster R-CNN implementation
- [ ] Check if the DataLoader is really efficient, if so, put the data pre-processing inside the getitem() - it will make the code simpler and to adapt to any image size easier. 
- [ ] Check about the pattern of bbox. I really need to consider bbox as pixels position or continuous ? 
      If continuous, makes sense to calculade the area as w1 - w0 + 1 (so, re-check the NMS) ?
---------- Well done! Keep going! --------------
- [ ] Add ResNet101 (currently is ResNet18)
- [ ] Document as in https://realpython.com/documenting-python-code/
---------- Well done! Keep going! --------------
- [ ] invalid anchors can be setted to -1 on labels ? (as dont care..)
- [ ] When processing the ground truths, show warnings when a certain GT box have no associated positive anchor
- [ ] what is the behavior when is there a not assigned gt ?
>    checar este behavior..
     eliminar esse gt caso for isso mesmo que eu estiver pensando, pois n vai servir de nada
     partir para o treino com batch balanceado!
- [ ] during training, use only the valid anchors, but in test, all anchors.. (vi em algum lugar isso)

---
# ------------------------------------------------------------------------
# Further TODO
# ------------------------------------------------------------------------
- [ ] Implement my own RoIAlign (can be in python instead of C), should give similar results to the built-in (but keep the faster as default) (https://stackoverflow.com/questions/60060016/why-does-roi-align-not-seem-to-work-in-pytorch)
- [x] Implement my own NMS, should give similar results to the built-in (but keep the faster as default)
- [ ] Check all the items marked as [future].

---
# ------------------------------------------------------------------------
# Lessons Learned 
# ------------------------------------------------------------------------
- The choice of anchor scale and ratios has a huge significance and impact
- Scaling the loss components (alpha, beta, ...) plays an important role in the optimization process
