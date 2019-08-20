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


def see_rpn_results(img_np, labels_np, proposals_np, probs_object_np, annotation_np, anchors_np, e):

    img_np *= 255
    real = Image.fromarray(img_np.astype(np.uint8))

    real_draw = ImageDraw.Draw(real)

    for b in range(labels_np.shape[0]):
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
                real_draw.rectangle([proposals_np[b, a, 0], proposals_np[b, a, 1], proposals_np[b, a, 2], proposals_np[b, a, 3]], outline='red')                    
                real_draw.rectangle([annotation_np[b, 0], annotation_np[b, 1], annotation_np[b, 2], annotation_np[b, 3]], outline='green')     

                print('Anchors: ', a0, a1, a2, a3)               
                # print('Anchors orig: ', anchors_np[k*4+0, i, j], anchors_np[k*4+1, i, j], anchors_np[k*4+2, i, j], anchors_np[k*4+3, i, j])
                print('Proposals: ', proposals_np[b, a, 0], proposals_np[b, a, 1], proposals_np[b, a, 2], proposals_np[b, a, 3])
                print('Prob (obj, not obj): ', probs_object_np[b, a, 0], probs_object_np[b, a, 1])
                print('Annotation: ', annotation_np[b, 0], annotation_np[b, 1], annotation_np[b, 2], annotation_np[b, 3])
                print()
                # print('Reg_out: ', reg_out_np[b, k*4+0, i, j], reg_out_np[b, k*4+1, i, j], reg_out_np[b, k*4+2, i, j], reg_out_np[b, k*4+3, i, j])

    # real.show()
    real.save('output/{}.jpg'.format(e))


def show_training_sample(img_np, annotation_np):

    img_np *= 255
    real = Image.fromarray(img_np.astype(np.uint8))

    print('Annotation np: ', annotation_np)

    real_draw = ImageDraw.Draw(real)
    real_draw.rectangle([annotation_np[0, 0], annotation_np[0, 1], annotation_np[0, 2], annotation_np[0, 3]], outline='red')
    real.show()
