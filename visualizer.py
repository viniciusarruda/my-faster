import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import config

# def show_training_sample(img_np, annotation_np):

#     img_np *= 255
#     real = Image.fromarray(img_np.astype(np.uint8))
#     annotation_np = _offset2bbox(annotation_np)

#     print('Annotation np: ', annotation_np)

#     real_draw = ImageDraw.Draw(real)
#     for i in range(annotation_np.shape[0]):
#         real_draw.rectangle([annotation_np[i, 0], annotation_np[i, 1], annotation_np[i, 2], annotation_np[i, 3]], outline='red')
#     real.show()

class Viz:
    
    def __init__(self, tensorboard=False, files=False, screen=True):

        self.writer = SummaryWriter()

        self.tensorboard = tensorboard
        self.files = files
        self.screen = screen

        self.epochs = []
        
        self.rpn_prob_loss = []
        self.rpn_bbox_loss = []
        self.rpn_loss = []

        self.clss_reg_prob_loss = []
        self.clss_reg_bbox_loss = []
        self.clss_reg_loss = []

        # self.font = ImageFont.truetype("UbuntuMono-R.ttf", size=10)
        # self.font = ImageFont.truetype("Quicksand-Regular.ttf", size=10)
        self.font = ImageFont.truetype("Quicksand-Medium.ttf", size=10)


    def __del__(self):

        if self.files:
            self.save_loss_file()

        self.writer.close()

    def _draw_bbox(self, draw_obj, bbox, color, text):

        draw_obj.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color)
        w,h = self.font.getsize(text)
        
        x0, y0 = max(bbox[0], 0), max(bbox[1] - h, 0)
        x1, y1 = min(x0 + w, config.input_img_size[0]), min(y0 + h, config.input_img_size[1])
        x0, y0 = x1 - w, y1 - h
        draw_obj.rectangle([x0, y0 + 2, x1, y1], outline=color, fill=color)
        draw_obj.text((x0 + 1, y0), text, fill='black', font=self.font)

    def _draw_anchor_bbox(self, draw_obj, bbox):
        self._draw_bbox(draw_obj, bbox, config.ANCHOR_COLOR, 'anchor')

    def _draw_obj_bbox(self, draw_obj, bbox, objctness):
        text = 'obj {:.0%}'.format(objctness)
        self._draw_bbox(draw_obj, bbox, config.OBJ_COLOR, text)

    def _draw_predictions(self, draw_obj, bboxes, pred_clss_idxs, clss_score):

        for a in range(clss_score.shape[0]):
            clss_idx = pred_clss_idxs[a]
            text = '{} {:.0%}'.format(config.class_names[clss_idx], clss_score[a])
            color = config.COLORS[clss_idx % config.NCOLORS]
            self._draw_bbox(draw_obj, bboxes[a], color, text)
            # print('Bboxes: ', bboxes[a, 0], bboxes[a, 1], bboxes[a, 2], bboxes[a, 3])
            # print("Clss '{}' with score: {}".format(config.class_names[clss_idx], clss_score[a]))
            # print()

    def _draw_annotations(self, draw_obj, annotations, clss_idxs):

        for i in range(annotations.shape[0]):
            self._draw_bbox(draw_obj, annotations[i], config.GT_COLOR, 'gt {}'.format(config.class_names[clss_idxs[i]]))
            # print('Annotation as bbox: ', annotations[i, 0], annotations[i, 1], annotations[i, 2], annotations[i, 3])


    def save_loss_file(self, filepath='output/loss.jpg'):

        def safe_log(xs):
            return [np.log(x + 1.0) for x in xs]

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

        axes[0, 0].plot(self.epochs, safe_log(self.rpn_prob_loss))
        axes[0, 0].set_title("rpn_prob_loss")
        axes[0, 1].plot(self.epochs, safe_log(self.rpn_bbox_loss))
        axes[0, 1].set_title("rpn_bbox_loss")

        axes[1, 0].plot(self.epochs, safe_log(self.clss_reg_prob_loss))
        axes[1, 0].set_title("clss_reg_prob_loss")
        axes[1, 1].plot(self.epochs, safe_log(self.clss_reg_bbox_loss))
        axes[1, 1].set_title("clss_reg_bbox_loss")

        fig.tight_layout()
        plt.savefig(filepath)


    def record(self, epoch, rpn_prob_loss, rpn_bbox_loss, rpn_loss, clss_reg_prob_loss, clss_reg_bbox_loss, clss_reg_loss, total_loss):

        if self.tensorboard:
            self.writer.add_scalar('Loss-RPN/class', rpn_prob_loss, epoch)
            self.writer.add_scalar('Loss-RPN/bbox', rpn_bbox_loss, epoch)

            self.writer.add_scalar('Loss-Regressor/class', clss_reg_prob_loss, epoch)
            self.writer.add_scalar('Loss-Regressor/bbox', clss_reg_bbox_loss, epoch)

            self.writer.add_scalar('Loss/rpn', rpn_loss, epoch)
            self.writer.add_scalar('Loss/regressor', clss_reg_loss, epoch)
            self.writer.add_scalar('Loss/total', total_loss, epoch)

            self.writer.flush()

        if self.screen:
            s = '\nEpoch {}: rpn_prob_loss: {} + rpn_bbox_loss: {} = {}'.format(epoch, rpn_prob_loss, rpn_bbox_loss, rpn_loss)
            s += '\n       : clss_reg_prob_loss: {} + clss_reg_bbox_loss: {} = {}'.format(clss_reg_prob_loss, clss_reg_bbox_loss, clss_reg_loss)
            tqdm.write(s)

        if self.files:

            self.epochs.append(epoch)

            self.rpn_prob_loss.append(rpn_prob_loss)
            self.rpn_bbox_loss.append(rpn_bbox_loss)
            self.rpn_loss.append(rpn_loss)

            self.clss_reg_prob_loss.append(clss_reg_prob_loss)
            self.clss_reg_bbox_loss.append(clss_reg_bbox_loss)
            self.clss_reg_loss.append(clss_reg_loss)


    def record_inference(self, inferences):

        for ith, inference in enumerate(inferences):

            epoch, img, annotation, clss_idxs = inference[:4]
            table_gts_positive_anchors, proposals, all_probs_object, anchors = inference[4:8]
            show_all_results = inference[8]
            probs_object, filtered_proposals = inference[9:11]
            clss_score, pred_clss_idxs, bboxes = inference[11:14]
            
            if epoch == 0:
                os.mkdir('output/rpn/img_{}/'.format(ith))
                os.mkdir('output/final_rpn/img_{}/'.format(ith))
                os.mkdir('output/final/img_{}/'.format(ith))

            # TODO Serio que tenho que fazer esse monte de coisa pra visualizar ?
            img = img.detach().cpu().numpy().copy().transpose(1, 2, 0) * 255
            annotation = annotation.detach().cpu().numpy().copy()
            clss_idxs = clss_idxs.detach().cpu().numpy().copy().astype(np.int)

            table_gts_positive_anchors = table_gts_positive_anchors.detach().cpu().numpy().copy()
            proposals = proposals.detach().cpu().numpy().copy()
            all_probs_object = all_probs_object.detach().cpu().numpy().copy()
            anchors = anchors.detach().cpu().numpy().copy()

            probs_object = probs_object.detach().cpu().numpy().copy()
            filtered_proposals = filtered_proposals.detach().cpu().numpy().copy()
            
            if show_all_results:
                clss_score = clss_score.detach().cpu().numpy().copy()
                pred_clss_idxs = pred_clss_idxs.detach().cpu().numpy().copy() # ja esta como np.int..
                bboxes = bboxes.detach().cpu().numpy().copy()

            img = Image.fromarray(img.astype(np.uint8))
            init_rpn_img = img.copy()
            final_rpn_img = img.copy()
            final_img = img.copy()

            # jogar esses annotatio as bboxx e tal para um arquivo para que consiga saber de ql imagem eh o bbox

            ### INIT RPN ###
            draw = ImageDraw.Draw(init_rpn_img)

            for b in range(annotation.shape[0]):  #### esse trecho de codigo todo aqui dentro pode ser deixado bem mais simples e similar aos outros de modo a ter uma unoca funcao pra tudo

                # self._draw_gt_bbox(draw, annotation[b])
                # draw.rectangle([annotation[b, 0], annotation[b, 1], annotation[b, 2], annotation[b, 3]], outline='green')
                # print('Annotation as bbox: ', annotation[b, 0], annotation[b, 1], annotation[b, 2], annotation[b, 3])

                selected_table_gts_positive_anchors = table_gts_positive_anchors[table_gts_positive_anchors[:, 0] == b, :]

                for a in range(selected_table_gts_positive_anchors.shape[0]):

                    anchor_idx = selected_table_gts_positive_anchors[a, 1]

                    self._draw_anchor_bbox(draw, anchors[anchor_idx])
                    self._draw_obj_bbox(draw, proposals[anchor_idx], all_probs_object[anchor_idx, 1])    
                    
                    # print('Anchors as bbox: ', anchors[anchor_idx, 0], anchors[anchor_idx, 1], anchors[anchor_idx, 2], anchors[anchor_idx, 3])               
                    # print('Proposals as bbox: ', proposals[anchor_idx, 0], proposals[anchor_idx, 1], proposals[anchor_idx, 2], proposals[anchor_idx, 3])
                    # print('Prob (not obj, obj): ', all_probs_object[anchor_idx, 0], all_probs_object[anchor_idx, 1])
                    # print()

            # init_rpn_img.show()
            init_rpn_img.save('output/rpn/img_{}/epoch_{}.jpg'.format(ith, epoch))
            self.writer.add_image('img_{}/begin-RPN'.format(ith), np.array(init_rpn_img), epoch, dataformats='HWC')


            if show_all_results: # if there is an inferred bbox

                ### FINAL RPN ###
                draw = ImageDraw.Draw(final_rpn_img)

                self._draw_annotations(draw, annotation, clss_idxs)

                for a in range(probs_object.shape[0]):

                    self._draw_obj_bbox(draw, filtered_proposals[a], probs_object[a])    

                    # print('Proposals as bbox: ', filtered_proposals[a, 0], filtered_proposals[a, 1], filtered_proposals[a, 2], filtered_proposals[a, 3])
                    # print('Prob: ', probs_object[a])            
                    # print()

                final_rpn_img.save('output/final_rpn/img_{}/epoch_{}.jpg'.format(ith, epoch))
                self.writer.add_image('img_{}/end-RPN'.format(ith), np.array(final_rpn_img), epoch, dataformats='HWC')

                
                ### FINAL ###
                draw = ImageDraw.Draw(final_img)    

                self._draw_annotations(draw, annotation, clss_idxs)

                assert pred_clss_idxs.shape == clss_score.shape

                self._draw_predictions(draw, bboxes, pred_clss_idxs, clss_score)
                
                final_img.save('output/final/img_{}/epoch_{}.jpg'.format(ith, epoch))
                self.writer.add_image('img_{}/final-output'.format(ith), np.array(final_img), epoch, dataformats='HWC')
        
        self.writer.flush()


    def show_anchors(self, rpn_anchors, image_size):

        def _show_anchors(anchors_np, image_size, filename, just_center=False):
            offset = 128
            img_np = np.zeros((image_size[1] + 2*offset, image_size[0] + 2*offset, 3))
            real = Image.fromarray(img_np.astype(np.uint8))
            real_draw = ImageDraw.Draw(real)
            # -1 and +1 for draw the box outside the boundary. The inside content is the content of the bbox/anchor
            real_draw.rectangle([offset - 1, offset - 1, offset + image_size[0] - 1 + 1, offset + image_size[1] - 1 + 1], outline='yellow')

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

        anchors_np = rpn_anchors.detach().numpy().copy()

        _show_anchors(anchors_np, image_size, 'valid_anchors')
        _show_anchors(anchors_np, image_size, 'valid_centers', just_center=True)



    def show_associated_positive_anchors(self, img_number, anchors_np, table_gts_positive_anchors_np, annotation_np, image_size):

        for gti in range(annotation_np.shape[0]):

            offset = 128
            img_np = np.zeros((image_size[1] + 2*offset, image_size[0] + 2*offset, 3))
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

                acw = anchors_np[positive_anchor_idx, 0]
                ach = anchors_np[positive_anchor_idx, 1]
                aw = anchors_np[positive_anchor_idx, 2]
                ah = anchors_np[positive_anchor_idx, 3]

                a0 = acw - 0.5 * (aw - 1)
                a1 = ach - 0.5 * (ah - 1)
                a2 = aw + a0 - 1
                a3 = ah + a1 - 1

                real_draw.rectangle([offset + a0, offset + a1, offset + a2, offset + a3], outline='magenta')

            real.save('output/anchors/associated_positive_anchors/{}_{}.jpg'.format(img_number, gti))


    def show_masked_anchors(self, e, anchors, mask, table_gts_positive_anchors, annotation, image_size):

        anchors_np = anchors.detach().numpy().copy()
        mask_np = mask.detach().numpy().copy()
        table_gts_positive_anchors_np = table_gts_positive_anchors.detach().numpy().copy()
        annotation_np = annotation.detach().numpy().copy()

        self.show_associated_positive_anchors(e, anchors_np, table_gts_positive_anchors_np, annotation_np, image_size)
        
        for mask, mask_name in zip([-1.0, 0.0, 1.0], ['middle', 'negative', 'positive']):

            masked_anchors_np = anchors_np[mask_np == mask, :]

            for i in range(masked_anchors_np.shape[0]):

                offset = 128
                img_np = np.zeros((image_size[1] + 2*offset, image_size[0] + 2*offset, 3))
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

                real_draw.rectangle([offset + a0, offset + a1, offset + a2, offset + a3], outline='magenta')

                a0 = acw - 1.0
                a1 = ach - 1.0
                a2 = acw + 1.0
                a3 = ach + 1.0
                real_draw.rectangle([offset + a0, offset + a1, offset + a2, offset + a3], outline='yellow')

                real.save('output/anchors/{}/{}_{}.jpg'.format(mask_name, e, i))
