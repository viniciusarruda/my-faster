
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
- [x] Fix valid anchors
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
- [x] GPU compatibility (To start to train in a large dataset and debug) 
- [x] Fix missing number of proposals
    - Solution was add gtboxes to the set of proposals from the RPN as made in the original implementation.
- [x] Fix DataLoader and refine code
    - [x] Clip the annotations to the image limits. If there is an image which has been clipped, warns the user!
    - [x] The limits of the annotations should be in [0, img_width/img_height) ? (I answered yes!) is so, np.clip(min=0, max=img_width-eps)
    - [x] Fix getitem() -- finish this de uma vez por todas! clean and comment code, leave just the relevant stuff, solve all issues related to it
    - [x] fix get item and balancing stuff in the two stages.
        - [x] CHECK CALMLY AGAIN!
        - [x] COMPARAR TBM A VISUALIZAÇÃO POIS ESTA ESQUISITA!!! 
    - [x] Check the annotations outputs drawn in the images.
- [x] Fix if-cases when balancing samples (there is four if-cases to fix)
- [x] Review the top N pre and pos when training and testing
- [x] Visualize the bboxes higher confidence on top. (revert the plot list since it is sorted after the nms)
- [x] First, retrain using ResNet18 - implement to base on config.py
- [x] Check the problem on not calling the inhirited functions on the backbone module
- [x] Check if I really need to call cpu() before numpy on visualizer.py (R: yep!)
    - [x] Check if already makes a copy (R: If not on cpu it copies, else does not)
    - [x] Check what is the behavior if is already on cpu()
    - [x] Check if there is an elegant alternative. (R: I didnt find)
- [x] Compare with the any implementation if it has bboxes out of the image.
- [x] Imprimir total loss tbm
PORQUE DEU O MESMO MIN E MAX WIDTH E HEIGHT COM TRAINVAL E TEST LA NA CAR206?
- [x] SOLVE THE NaN PROBLEM: 
    - esta explodindo os valores tanto pra muito negativo quando para muito positivo na reg_bbox da rpn..
    - try the initilization and the learning parameters of the network.
    - assert fg/bg com fg >= 1!
- [ ] Check the detach() issue
    - Maybe, I have to detach something to get a better result
- [ ] Regress the background bbox class to zero to see if get a better result
- [ ] Check mAP on train and test set. Check for high bias and high variance as learned on deep learning specialization.
      The goal is to have highest mAP on both train/test set with the lower difference between them.
      If there is higher difference, then one decision should be made to better the result. (regularization, etc)
      If there is lower difference but lower mAP, the decision is different from above. (bigger models, etc)
- [ ] Standardize variables:
    - bbox format:   always x0, y0, x1, y1   with name: var_name_bf
    - offset format: always cw, ch, w, h     with name: var_name_of
- [ ] Sera que se minimar a bbox background para tudo zero pode melhorar?
      No momento estou apenas ignorando a existencia dela.. ela nem existe.. apenas o nao-objeto da RPN
- [ ] Ainda tem como adicionar a contagem das bboxes que n foram associadas por ancoras.
    Pois tem as diretamente e indiretamente e também as nao associadas.. entao mostrar tbm essas estatisticas.
- [x] Check the necessity to assign 0 score to bboxes removed after NMS. acho que eh so pra sair as inferencias com bbox prob 0.. nas hora de calcular o map e tal
    - I didnt really understand why.. it is a padding.. but why?
- [x] Implement ResNet50
- [x] Implement ResNet101
- [ ] Overfit some data.. check instability.. if persist, compare the overfiting against another Faster R-CNN implementation
 - USE THIS AS A BASELINE WHILE DEBUGGING YOUR IMPLEMENTATION!
- [ ] Check if the DataLoader is really efficient, if so, put the data pre-processing inside the getitem() - it will make the code simpler and to adapt to any image size easier. 
---------- Well done! Keep going! --------------
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
# Code organization and optimization TODO
# ------------------------------------------------------------------------
- [ ] Reduce repeated code into functions
    - [ ] anchor_labels, get_target_mask and dataloader stuff
- [ ] Maybe: Change format bbox to [0, 1, 2, 3, class_idx] instead of keeping it separated
- [ ] Check about the pattern of bbox. I really need to consider bbox as pixels position or continuous ? 
      If continuous, makes sense to calculade the area as w1 - w0 + 1 (so, re-check the NMS) ?

---
# ------------------------------------------------------------------------
# Further TODO
# ------------------------------------------------------------------------
- [ ] Implement my own RoIAlign (can be in python instead of C), should give similar results to the built-in (but keep the faster as default) (https://stackoverflow.com/questions/60060016/why-does-roi-align-not-seem-to-work-in-pytorch)
- [x] Implement my own NMS, should give similar results to the built-in (but keep the faster as default)
- [ ] Check all the items marked as [future].
- [ ] take a look at initialization.. He initialization?
    it will be nice to implement several of them and compare

---
# ------------------------------------------------------------------------
# Lessons Learned 
# ------------------------------------------------------------------------
- The choice of anchor scale and ratios has a huge significance and impact
- Scaling the loss components (alpha, beta, ...) plays an important role in the optimization process
- The high amount of anchors can lean to a big discrepancy between positive and negative anchors, being the latter much higher. The optimization finds a short-cut by kicking every class to background, being hard to the positive proposals to acquire a high probability and be passed to thw second stage.
  To overcome this issue, a solution can be to reduce the number of proposals to be kept (top_N). This will make the batch more balanced. Why? Because when setting only the batch size for the RPN, there is a huge number of available proposals, and is highly probable that the most is negative examples, because their confident level is too high.
- The number of top_N in the RPN is for anchors! I am always confusing with objects!!!!! This justifies the chosen number top_N to be high.
- The tiny oscilations of the bboxes at the end of training (flickering bboxes effect) is due to the constant change of the winner anchor, i.e., the one with the highest prob. The NMS suppressed the others. So in one step, anchor A with prob 99% wins anchor B which has prob 98%, and since they overlaps with high threshold, B is suppressed by A. And in other step A gets prob, say 95%, and B gets 97%, thus A is suppressed by B.
  How one can solve this? I propose the Mean Non-Maximum Suppression: get the mean of the bboxes coordinates along the ones which have the iou above threshold. Also, the Logarithm Weighted Mean Non-Maximum Suppression: which weights more the bboxes with higher probs than the lower ones.
  Advantage of this: Does not need to sort? Stop flickering bboxes? Helps to improve the harder cases?