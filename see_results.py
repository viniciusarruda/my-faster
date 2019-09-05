from PIL import Image, ImageDraw
import os
import numpy as np
import json

def see_results(img_np, clss_out_np, bbox_out_np):

    img_np *= 255
    real = Image.fromarray(img_np.astype(np.uint8))

    real_draw = ImageDraw.Draw(real)

    for i in range(clss_out_np.shape[0]):

        print(clss_out_np[i])
        real_draw.rectangle([bbox_out_np[i, 0], bbox_out_np[i, 1], bbox_out_np[i, 2], bbox_out_np[i, 3]], outline='red')

    real.show()

    # input()


def see_rpn_results(img_np, labels_np, proposals_np, probs_object_np, annotation_np, anchors_np, valid_anchors_np, e):

    bboxes_np = _offset2bbox(proposals_np)
    annotation_np = _offset2bboxann(annotation_np)
    anchors_np = anchors_np[valid_anchors_np == 1, :]

    img_np *= 255
    real = Image.fromarray(img_np.astype(np.uint8))

    real_draw = ImageDraw.Draw(real)

    for b in range(labels_np.shape[0]):

        real_draw.rectangle([annotation_np[b, 0], annotation_np[b, 1], annotation_np[b, 2], annotation_np[b, 3]], outline='green')     
        print('Annotation as bbox: ', annotation_np[b, 0], annotation_np[b, 1], annotation_np[b, 2], annotation_np[b, 3])

        for a in range(labels_np.shape[1]):

            if labels_np[b, a] == 1.0:

                acw = anchors_np[a, 0]
                ach = anchors_np[a, 1]
                aw = anchors_np[a, 2]
                ah = anchors_np[a, 3]

                a0 = acw - 0.5 * (aw - 1)
                a1 = ach - 0.5 * (ah - 1)
                a2 = aw + a0 - 1
                a3 = ah + a1 - 1

                real_draw.rectangle([a0, a1, a2, a3], outline='magenta')                    
                real_draw.rectangle([bboxes_np[b, a, 0], bboxes_np[b, a, 1], bboxes_np[b, a, 2], bboxes_np[b, a, 3]], outline='red')                    

                print('Anchors as bbox: ', a0, a1, a2, a3)               
                print('Proposals as bbox: ', bboxes_np[b, a, 0], bboxes_np[b, a, 1], bboxes_np[b, a, 2], bboxes_np[b, a, 3])
                print('Proposals: ', proposals_np[b, a, 0], proposals_np[b, a, 1], proposals_np[b, a, 2], proposals_np[b, a, 3])
                print('Prob (obj, not obj): ', probs_object_np[b, a, 0], probs_object_np[b, a, 1])
                print()

    # real.show()
    real.save('output/rpn/{}.jpg'.format(e))


def see_rpn_final_results(img_np, proposals_np, probs_object_np, annotation_np, e):

    bboxes_np = _offset2bbox(proposals_np)
    annotation_np = _offset2bboxann(annotation_np)

    img_np *= 255
    real = Image.fromarray(img_np.astype(np.uint8))

    real_draw = ImageDraw.Draw(real)

    for b in range(probs_object_np.shape[0]):
        
        real_draw.rectangle([annotation_np[b, 0], annotation_np[b, 1], annotation_np[b, 2], annotation_np[b, 3]], outline='green')     
        print('Annotation as bbox: ', annotation_np[b, 0], annotation_np[b, 1], annotation_np[b, 2], annotation_np[b, 3])

        for a in range(probs_object_np.shape[1]):
                
            real_draw.rectangle([bboxes_np[b, a, 0], bboxes_np[b, a, 1], bboxes_np[b, a, 2], bboxes_np[b, a, 3]], outline='red')                    

            print('Proposals as bbox: ', bboxes_np[b, a, 0], bboxes_np[b, a, 1], bboxes_np[b, a, 2], bboxes_np[b, a, 3])
            print('Proposals: ', proposals_np[b, a, 0], proposals_np[b, a, 1], proposals_np[b, a, 2], proposals_np[b, a, 3])
            print('Prob: ', probs_object_np[b, a])            
            print()

    # real.show()
    real.save('output/final_rpn/{}.jpg'.format(e))


def see_final_results(img_np, clss_score_np, refined_proposals_np, annotation_np, e):

    annotation_np = _offset2bboxann(annotation_np)
    bboxes_np = _offset2bbox(refined_proposals_np)

    img_np *= 255
    real = Image.fromarray(img_np.astype(np.uint8))

    real_draw = ImageDraw.Draw(real)

    for b in range(annotation_np.shape[0]):

        real_draw.rectangle([annotation_np[b, 0], annotation_np[b, 1], annotation_np[b, 2], annotation_np[b, 3]], outline='magenta')                    
        print('Annotation as bbox: ', annotation_np[b, 0], annotation_np[b, 1], annotation_np[b, 2], annotation_np[b, 3])               

        for a in range(clss_score_np.shape[1]):
        
            real_draw.rectangle([bboxes_np[b, a, 0], bboxes_np[b, a, 1], bboxes_np[b, a, 2], bboxes_np[b, a, 3]], outline='red')                    
            print('Bboxes: ', bboxes_np[b, a, 0], bboxes_np[b, a, 1], bboxes_np[b, a, 2], bboxes_np[b, a, 3])
            print('Clss score: ', clss_score_np[b, a])
            print()

    # real.show()
    real.save('output/final/{}.jpg'.format(e))


def show_training_sample(img_np, annotation_np):

    img_np *= 255
    real = Image.fromarray(img_np.astype(np.uint8))

    print('Annotation np: ', annotation_np)

    real_draw = ImageDraw.Draw(real)
    real_draw.rectangle([annotation_np[0, 0], annotation_np[0, 1], annotation_np[0, 2], annotation_np[0, 3]], outline='red')
    real.show()


def show_anchors(anchors_np, valid_anchors_np):

    offset = 128
    img_np = np.zeros((128 + 2*offset, 128 + 2*offset, 3))
    real = Image.fromarray(img_np.astype(np.uint8))
    real_draw = ImageDraw.Draw(real)
    real_draw.rectangle([offset - 1, offset - 1, offset + 127 + 1, offset + 127 + 1], outline='yellow') # atencao para o -1 e +1 e seu significado !

    c = 0

    for i in range(valid_anchors_np.shape[0]):

        acw = anchors_np[i, 0]
        ach = anchors_np[i, 1]
        aw = anchors_np[i, 2]
        ah = anchors_np[i, 3]

        a0 = acw - 0.5 * (aw - 1)
        a1 = ach - 0.5 * (ah - 1)
        a2 = aw + a0 - 1
        a3 = ah + a1 - 1

        if valid_anchors_np[i] == 1:
            c += 1
            real_draw.rectangle([offset + a0, offset + a1, offset + a2, offset + a3], outline='magenta')
        # else:
            # real_draw.rectangle([offset + a0, offset + a1, offset + a2, offset + a3], outline='white')
            # real_draw.point([offset + anchors_np[i, 0], offset + anchors_np[i, 1]], fill='white')
    
    real.show()
    print(c)
    exit()

# def show_anchors(anchors_np):

#     offset = 128
#     img_np = np.zeros((128 + 2*offset, 128 + 2*offset, 3))
#     real = Image.fromarray(img_np.astype(np.uint8))
#     real_draw = ImageDraw.Draw(real)
#     real_draw.rectangle([offset - 1, offset - 1, offset + 127 + 1, offset + 127 + 1], outline='yellow') # atencao para o -1 e +1 e seu significado !

#     c = 0

#     for i in range(anchors_np.shape[0]):

#         acw = anchors_np[i, 0]
#         ach = anchors_np[i, 1]
#         aw = anchors_np[i, 2]
#         ah = anchors_np[i, 3]

#         a0 = acw - 0.5 * (aw - 1)
#         a1 = ach - 0.5 * (ah - 1)
#         a2 = aw + a0 - 1
#         a3 = ah + a1 - 1

#         if a0 >= 0 and a1 >= 0 and a2 <= 127 and a3 <= 127:
#             c += 1
#             real_draw.rectangle([offset + a0, offset + a1, offset + a2, offset + a3], outline='magenta')
#             real_draw.point([offset + anchors_np[i, 0], offset + anchors_np[i, 1]], fill='white')
    
#     real.show()
#     # print(c)
#     # exit()


def _offset2bbox(proposals):
    """
    proposals: batch_size, -1, 4
    bboxes: batch_size, -1, 4

    """

    bx0 = proposals[:, :, 0]
    by0 = proposals[:, :, 1]
    bx1 = bx0 + proposals[:, :, 2] - 1
    by1 = by0 + proposals[:, :, 3] - 1

    bboxes = np.stack((bx0, by0, bx1, by1), axis=2)

    return bboxes

def _offset2bboxann(proposals):
    """
    proposals: batch_size, -1, 4
    bboxes: batch_size, -1, 4

    """

    bx0 = proposals[:, 0]
    by0 = proposals[:, 1]
    bx1 = bx0 + proposals[:, 2] - 1
    by1 = by0 + proposals[:, 3] - 1

    bboxes = np.stack((bx0, by0, bx1, by1), axis=1)

    return bboxes