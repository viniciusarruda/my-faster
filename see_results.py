from PIL import Image, ImageDraw
import os
import numpy as np
import json
import matplotlib.pyplot as plt


def see_rpn_results(img_np, table_gts_positive_anchors, proposals_np, probs_object_np, annotation_np, anchors_np, valid_anchors_np, e):

    bboxes_np = _offset2bbox(proposals_np)
    annotation_np = _offset2bbox(annotation_np)
    anchors_np = anchors_np[valid_anchors_np == 1, :]

    img_np *= 255
    real = Image.fromarray(img_np.astype(np.uint8))

    real_draw = ImageDraw.Draw(real)

    for b in range(annotation_np.shape[0]):

        real_draw.rectangle([annotation_np[b, 0], annotation_np[b, 1], annotation_np[b, 2], annotation_np[b, 3]], outline='green')
        print('Annotation as bbox: ', annotation_np[b, 0], annotation_np[b, 1], annotation_np[b, 2], annotation_np[b, 3])

        selected_table_gts_positive_anchors = table_gts_positive_anchors[table_gts_positive_anchors[:, 0] == b, :]

        for a in range(selected_table_gts_positive_anchors.shape[0]):

            anchor_idx = selected_table_gts_positive_anchors[a, 1]

            acw = anchors_np[anchor_idx, 0]
            ach = anchors_np[anchor_idx, 1]
            aw = anchors_np[anchor_idx, 2]
            ah = anchors_np[anchor_idx, 3]

            a0 = acw - 0.5 * (aw - 1)
            a1 = ach - 0.5 * (ah - 1)
            a2 = aw + a0 - 1
            a3 = ah + a1 - 1

            real_draw.rectangle([a0, a1, a2, a3], outline='magenta')                    
            real_draw.rectangle([bboxes_np[anchor_idx, 0], bboxes_np[anchor_idx, 1], bboxes_np[anchor_idx, 2], bboxes_np[anchor_idx, 3]], outline='red')                    

            print('Anchors as bbox: ', a0, a1, a2, a3)               
            print('Proposals as bbox: ', bboxes_np[anchor_idx, 0], bboxes_np[anchor_idx, 1], bboxes_np[anchor_idx, 2], bboxes_np[anchor_idx, 3])
            print('Proposals: ', proposals_np[anchor_idx, 0], proposals_np[anchor_idx, 1], proposals_np[anchor_idx, 2], proposals_np[anchor_idx, 3])
            print('Prob (not obj, obj): ', probs_object_np[anchor_idx, 0], probs_object_np[anchor_idx, 1])
            print()
        

    # real.show()
    real.save('output/rpn/{}.jpg'.format(e))


def see_rpn_final_results(img_np, proposals_np, probs_object_np, annotation_np, e):

    bboxes_np = _offset2bbox(proposals_np)
    annotation_np = _offset2bbox(annotation_np)

    img_np *= 255
    real = Image.fromarray(img_np.astype(np.uint8))

    real_draw = ImageDraw.Draw(real)

    for b in range(annotation_np.shape[0]):
        
        real_draw.rectangle([annotation_np[b, 0], annotation_np[b, 1], annotation_np[b, 2], annotation_np[b, 3]], outline='green')
        print('Annotation as bbox: ', annotation_np[b, 0], annotation_np[b, 1], annotation_np[b, 2], annotation_np[b, 3])

        for a in range(probs_object_np.shape[0]):
                
            real_draw.rectangle([bboxes_np[a, 0], bboxes_np[a, 1], bboxes_np[a, 2], bboxes_np[a, 3]], outline='red')                    

            print('Proposals as bbox: ', bboxes_np[a, 0], bboxes_np[a, 1], bboxes_np[a, 2], bboxes_np[a, 3])
            print('Proposals: ', proposals_np[a, 0], proposals_np[a, 1], proposals_np[a, 2], proposals_np[a, 3])
            print('Prob: ', probs_object_np[a])            
            print()

    # real.show()
    real.save('output/final_rpn/{}.jpg'.format(e))


def see_final_results(img_np, clss_score_np, refined_proposals_np, annotation_np, e):

    bboxes_np = _offset2bbox(refined_proposals_np)
    annotation_np = _offset2bbox(annotation_np)

    img_np *= 255
    real = Image.fromarray(img_np.astype(np.uint8))

    real_draw = ImageDraw.Draw(real)

    for b in range(annotation_np.shape[0]):

        real_draw.rectangle([annotation_np[b, 0], annotation_np[b, 1], annotation_np[b, 2], annotation_np[b, 3]], outline='green')                    
        print('Annotation as bbox: ', annotation_np[b, 0], annotation_np[b, 1], annotation_np[b, 2], annotation_np[b, 3])               

        for a in range(clss_score_np.shape[0]):
        
            real_draw.rectangle([bboxes_np[a, 0], bboxes_np[a, 1], bboxes_np[a, 2], bboxes_np[a, 3]], outline='red')                    
            print('Bboxes: ', bboxes_np[a, 0], bboxes_np[a, 1], bboxes_np[a, 2], bboxes_np[a, 3])
            print('Clss score: ', clss_score_np[a])
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


def show_anchors(anchors_np, valid_anchors_np, image_size):

    def _show_anchors(anchors_np, image_size, filename, just_center=False):
        offset = 128
        img_np = np.zeros((image_size[0] + 2*offset, image_size[1] + 2*offset, 3))
        real = Image.fromarray(img_np.astype(np.uint8))
        real_draw = ImageDraw.Draw(real)
        real_draw.rectangle([offset - 1, offset - 1, offset + image_size[0] - 1 + 1, offset + image_size[1] - 1 + 1], outline='yellow') # atencao para o -1 e +1 e seu significado !

        for i in range(anchors_np.shape[0]):
            
            acw = anchors_np[i, 0]
            ach = anchors_np[i, 1]
            aw = anchors_np[i, 2]
            ah = anchors_np[i, 3]

            if just_center:
                a0 = acw - 1.0
                a1 = ach - 1.0
                a2 = acw + 1.0
                a3 = ach + 1.0
            else:
                a0 = acw - 0.5 * (aw - 1)
                a1 = ach - 0.5 * (ah - 1)
                a2 = aw + a0 - 1
                a3 = ah + a1 - 1

            real_draw.rectangle([offset + a0, offset + a1, offset + a2, offset + a3], outline='magenta')

        # real.show()
        real.save('output/anchors/{}.jpg'.format(filename))


    _show_anchors(anchors_np, image_size, 'all_anchors')
    _show_anchors(anchors_np, image_size, 'all_centers', just_center=True)
    print('A total of {} anchors.'.format(anchors_np.shape[0]))
    anchors_np = anchors_np[valid_anchors_np == 1, :]
    _show_anchors(anchors_np, image_size, 'valid_anchors')
    _show_anchors(anchors_np, image_size, 'valid_centers', just_center=True)
    print('A total of {} valid anchors.'.format(anchors_np.shape[0]))


def show_associated_positive_anchors(img_number, anchors_np, valid_anchors_np, table_gts_positive_anchors_np, annotation_np, image_size):

    vanchors_np = anchors_np[valid_anchors_np == 1, :]

    for gti in range(annotation_np.shape[0]):

        offset = 128
        img_np = np.zeros((image_size[0] + 2*offset, image_size[1] + 2*offset, 3))
        real = Image.fromarray(img_np.astype(np.uint8))
        real_draw = ImageDraw.Draw(real)
        real_draw.rectangle([offset - 1, offset - 1, offset + image_size[0] - 1 + 1, offset + image_size[1] - 1 + 1], outline='yellow') # atencao para o -1 e +1 e seu significado !

        x0 = annotation_np[gti, 0]
        y0 = annotation_np[gti, 1]
        x1 = annotation_np[gti, 0] + annotation_np[gti, 2] - 1
        y1 = annotation_np[gti, 1] + annotation_np[gti, 3] - 1

        real_draw.rectangle([offset + x0, offset + y0, offset + x1, offset + y1], outline='green')

        for pai in np.argwhere(table_gts_positive_anchors_np[:, 0] == gti): 

            positive_anchor_idx = table_gts_positive_anchors_np[pai[0], 1]

            acw = vanchors_np[positive_anchor_idx, 0]
            ach = vanchors_np[positive_anchor_idx, 1]
            aw = vanchors_np[positive_anchor_idx, 2]
            ah = vanchors_np[positive_anchor_idx, 3]

            a0 = acw - 0.5 * (aw - 1)
            a1 = ach - 0.5 * (ah - 1)
            a2 = aw + a0 - 1
            a3 = ah + a1 - 1

            real_draw.rectangle([offset + a0, offset + a1, offset + a2, offset + a3], outline='magenta')

        real.save('output/anchors/associated_positive_anchors/{}_{}.jpg'.format(img_number, gti))


def show_masked_anchors(e, anchors_np, valid_anchors_np, mask_np, table_gts_positive_anchors_np, annotation_np, image_size):

    show_associated_positive_anchors(e, anchors_np, valid_anchors_np, table_gts_positive_anchors_np, annotation_np, image_size)

    anchors_np = anchors_np[valid_anchors_np == 1, :]
    # mask_np = mask_np[0, :] # batch 1
    
    for mask, mask_name in zip([-1.0, 0.0, 1.0], ['middle', 'negative', 'positive']):

        masked_anchors_np = anchors_np[mask_np == mask, :]

        for i in range(masked_anchors_np.shape[0]):

            offset = 128
            img_np = np.zeros((image_size[0] + 2*offset, image_size[1] + 2*offset, 3))
            real = Image.fromarray(img_np.astype(np.uint8))
            real_draw = ImageDraw.Draw(real)
            real_draw.rectangle([offset - 1, offset - 1, offset + image_size[0] - 1 + 1, offset + image_size[1] - 1 + 1], outline='yellow') # atencao para o -1 e +1 e seu significado !

            for bi in range(annotation_np.shape[0]):
                
                x0 = annotation_np[bi, 0]
                y0 = annotation_np[bi, 1]
                x1 = annotation_np[bi, 0] + annotation_np[bi, 2] - 1
                y1 = annotation_np[bi, 1] + annotation_np[bi, 3] - 1

                real_draw.rectangle([offset + x0, offset + y0, offset + x1, offset + y1], outline='green')

            acw = masked_anchors_np[i, 0]
            ach = masked_anchors_np[i, 1]
            aw = masked_anchors_np[i, 2]
            ah = masked_anchors_np[i, 3]

            a0 = acw - 0.5 * (aw - 1)
            a1 = ach - 0.5 * (ah - 1)
            a2 = aw + a0 - 1
            a3 = ah + a1 - 1

            # if valid_anchors_np[i] == 1 and mask_np[c] == -1:
            real_draw.rectangle([offset + a0, offset + a1, offset + a2, offset + a3], outline='magenta')
            # else:
                # real_draw.rectangle([offset + a0, offset + a1, offset + a2, offset + a3], outline='white')
                # real_draw.point([offset + anchors_np[i, 0], offset + anchors_np[i, 1]], fill='white')

            a0 = acw - 1.0
            a1 = ach - 1.0
            a2 = acw + 1.0
            a3 = ach + 1.0
            real_draw.rectangle([offset + a0, offset + a1, offset + a2, offset + a3], outline='yellow')

            real.save('output/anchors/{}/{}_{}.jpg'.format(mask_name, e, i))

    # exit()


def _offset2bbox(proposals):
    """
    proposals: #proposals, 4
    bboxes: #bboxes, 4

    """
    # TODO remove this assertion
    assert len(proposals.shape) == 2

    bx0 = proposals[:, 0]
    by0 = proposals[:, 1]
    bx1 = bx0 + proposals[:, 2] - 1
    by1 = by0 + proposals[:, 3] - 1

    bboxes = np.stack((bx0, by0, bx1, by1), axis=1)

    return bboxes

class LossViz:
    
    def __init__(self):

        self.epoch = []
        
        self.rpn_prob_loss = []
        self.rpn_bbox_loss = []
        self.rpn_loss = []

        self.clss_reg_prob_loss = []
        self.clss_reg_bbox_loss = []
        self.clss_reg_loss = []


    def record(self, epoch, rpn_prob_loss, rpn_bbox_loss, rpn_loss, clss_reg_prob_loss, clss_reg_bbox_loss, clss_reg_loss):

        def safe_log(x):
            return np.log(x + 1.0)

        self.epoch.append(epoch)

        self.rpn_prob_loss.append(safe_log(rpn_prob_loss))
        self.rpn_bbox_loss.append(safe_log(rpn_bbox_loss))
        self.rpn_loss.append(safe_log(rpn_loss))

        self.clss_reg_prob_loss.append(safe_log(clss_reg_prob_loss))
        self.clss_reg_bbox_loss.append(safe_log(clss_reg_bbox_loss))
        self.clss_reg_loss.append(safe_log(clss_reg_loss))




    def save(self, filepath='output/loss.jpg'):

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

        axes[0, 0].plot(self.epoch, self.rpn_prob_loss)
        axes[0, 0].set_title("rpn_prob_loss")
        axes[0, 1].plot(self.epoch, self.rpn_bbox_loss)
        axes[0, 1].set_title("rpn_bbox_loss")

        axes[1, 0].plot(self.epoch, self.clss_reg_prob_loss)
        axes[1, 0].set_title("clss_reg_prob_loss")
        axes[1, 1].plot(self.epoch, self.clss_reg_bbox_loss)
        axes[1, 1].set_title("clss_reg_bbox_loss")

        fig.tight_layout()
        plt.savefig(filepath)


        # print('Epoch {}: rpn_prob_loss: {} + rpn_bbox_loss: {} = {}'.format(e, rpn_prob_loss_epoch / l, rpn_bbox_loss_epoch / l, rpn_loss_epoch / l))
        # print('       : clss_reg_prob_loss: {} + clss_reg_bbox_loss: {} = {}'.format(clss_reg_prob_loss_epoch / l, clss_reg_bbox_loss_epoch / l, clss_reg_loss_epoch / l))
        # print()