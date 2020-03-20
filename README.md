
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
- The images are in the format I(n_rows, n_cols), and indexed always as (r, c)
- The height and the width of an image is handled in the code as n_rows and n_cols, respectively.
- The (0, 0) point is at the top left corner of the image

## Other standard
- The format of bounding box is in x,y,n_rows,n_cols unless the variable name contains a bbox word.

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
        -> In fact, it behaves slightly different when handling degenerate boxes,
           but nothing that gets in the way of performance.
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
    - [ ] Do I need to keep all anchors or only the valid ones through the code?
- [ ] Enable images with aspect ration different than 1 (and fix the comments accordingly)
- [ ] Clean/comment/review loss.py file (check to remove duplicata as calculating the iou)
- [ ] Document as in https://realpython.com/documenting-python-code/
- [ ] Improve the visualization including verbose information for debugging (Tensorboard)
    - [ ] Include a heatmap visualization of the features right before the RPN (maybe an insight to use attention here)
- [ ] Check about the pattern of bbox. I really need to consider bbox as pixels position or continuous ? 
      If continuous, makes sense to calculade the area as w1 - w0 + 1 (so, re-check the NMS) ?
---------- Well done! Keep going! --------------
- [ ] GPU compatibility (To start to train in a large dataset and debug) 
- [ ] Increase the size of the input image
- [ ] Increase the number of images in the training set
- [ ] Implement the resnet as feature extractor
- [ ] Fix the parameters for the new feature extractor
- [ ] Implement the training strategy correctly
- [ ] during training, use only the valid anchors, but in test, all anchors..
- [ ] use image with different size (no multiple yet, but different sizes)
- [ ] more anchor ratios and scales
- [ ] single image with two (and then more) objects
- [ ] multiple images
- [ ] RCNN_top
- [ ] invalid anchors can be setted to -1 on labels ? (as dont care..)
- [ ] When processing the ground truths, show warnings when a certain GT box have no associated positive anchor
- [ ] what is the behavior when is there a not assigned gt ?
>    checar este behavior..
     eliminar esse gt caso for isso mesmo que eu estiver pensando, pois n vai servir de nada
     partir para o treino com batch balanceado!
- [ ] Add ResNet101 (currently is ResNet18)

---
# ------------------------------------------------------------------------
# Further TODO
# ------------------------------------------------------------------------
- [ ] Implement my own RoIAlign (can be in python instead of C), should give similar results to the built-in (but keep the faster as default) (https://stackoverflow.com/questions/60060016/why-does-roi-align-not-seem-to-work-in-pytorch)
- [x] Implement my own NMS, should give similar results to the built-in (but keep the faster as default)

---
# ------------------------------------------------------------------------
# Lessons Learned 
# ------------------------------------------------------------------------
- The choice of anchor scale and ratios has a huge significance and impact
- Scaling the loss components (alpha, beta, ...) plays an important role in the optimization process
