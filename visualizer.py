import config
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

        self.learning_rate = []

        self.font = ImageFont.truetype("arial.ttf", size=12)

        self.create_folder = True

    def __del__(self):

        if self.files:
            self.save_loss_file()
            self.save_learning_rate_file()

        self.writer.close()

    def _draw_bbox(self, draw_obj, bbox, color, text):

        draw_obj.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color)
        w, h = self.font.getsize(text)

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

        for a in reversed(range(clss_score.shape[0])):
            clss_idx = pred_clss_idxs[a]
            text = '{} {:.0%}'.format(config.class_names[clss_idx], clss_score[a])
            color = config.COLORS[clss_idx % config.NCOLORS]
            self._draw_bbox(draw_obj, bboxes[a], color, text)
            # print('Bboxes: ', bboxes[a, 0], bboxes[a, 1], bboxes[a, 2], bboxes[a, 3])
            # print("Clss '{}' with score: {}".format(config.class_names[clss_idx], clss_score[a]))
            # print()

    def _draw_annotations(self, draw_obj, annotations):

        for i in range(annotations.shape[0]):
            self._draw_bbox(draw_obj, annotations[i], config.GT_COLOR, 'gt {}'.format(config.class_names[int(annotations[i, -1])]))
            # print('Annotation as bbox: ', annotations[i, 0], annotations[i, 1], annotations[i, 2], annotations[i, 3])

    def save_loss_file(self, filepath='output/loss.jpg', filepath_log='output/log_loss.jpg'):

        def safe_log(xs):
            return [np.log(x + 1.0) for x in xs]

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

        axes[0, 0].plot(self.epochs, self.rpn_prob_loss)
        axes[0, 0].set_title("rpn_prob_loss")
        axes[0, 1].plot(self.epochs, self.rpn_bbox_loss)
        axes[0, 1].set_title("rpn_bbox_loss")

        axes[1, 0].plot(self.epochs, self.clss_reg_prob_loss)
        axes[1, 0].set_title("clss_reg_prob_loss")
        axes[1, 1].plot(self.epochs, self.clss_reg_bbox_loss)
        axes[1, 1].set_title("clss_reg_bbox_loss")

        fig.tight_layout()
        plt.savefig(filepath)
        del fig, axes

        fig_log, axes_log = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

        axes_log[0, 0].plot(self.epochs, safe_log(self.rpn_prob_loss))
        axes_log[0, 0].set_title("rpn_prob_loss")
        axes_log[0, 1].plot(self.epochs, safe_log(self.rpn_bbox_loss))
        axes_log[0, 1].set_title("rpn_bbox_loss")

        axes_log[1, 0].plot(self.epochs, safe_log(self.clss_reg_prob_loss))
        axes_log[1, 0].set_title("clss_reg_prob_loss")
        axes_log[1, 1].plot(self.epochs, safe_log(self.clss_reg_bbox_loss))
        axes_log[1, 1].set_title("clss_reg_bbox_loss")

        fig_log.tight_layout()
        plt.savefig(filepath_log)

    def save_learning_rate_file(self, filepath='output/learning_rate.jpg'):

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

        axes.plot(self.epochs, self.learning_rate)
        axes.set_title("learning_rate")

        fig.tight_layout()
        plt.savefig(filepath)

    def save_graph(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model=input_to_model)

    def record_losses(self, epoch, iteration, display_on, recorded_losses, learning_rate):

        if self.tensorboard:
            self.writer.add_scalar('Loss-RPN/class', recorded_losses['rpn_prob'], iteration)
            self.writer.add_scalar('Loss-RPN/bbox', recorded_losses['rpn_bbox'], iteration)

            self.writer.add_scalar('Loss-Regressor/class', recorded_losses['clss_reg_prob'], iteration)
            self.writer.add_scalar('Loss-Regressor/bbox', recorded_losses['clss_reg_bbox'], iteration)

            self.writer.add_scalar('Loss/rpn', recorded_losses['rpn'], iteration)
            self.writer.add_scalar('Loss/regressor', recorded_losses['clss_reg'], iteration)
            self.writer.add_scalar('Loss/total', recorded_losses['total'], iteration)

            self.writer.add_scalar('Learning-rate', learning_rate, iteration)

            # self.writer.flush()

        if self.screen and display_on:
            s = '\nEpoch {} | Iteration {}'.format(epoch, iteration)
            s += '\n       : total_loss: {:.3f}'.format(recorded_losses['total'])
            s += '\n       : rpn_prob_loss: {:.3f} + rpn_bbox_loss: {:.3f} = {:.3f}'.format(recorded_losses['rpn_prob'], recorded_losses['rpn_bbox'], recorded_losses['rpn'])
            s += '\n       : clss_reg_prob_loss: {:.3f} + clss_reg_bbox_loss: {:.3f} = {:.3f}'.format(recorded_losses['clss_reg_prob'], recorded_losses['clss_reg_bbox'], recorded_losses['clss_reg'])
            s += '\n       : learning rate: {}'.format(learning_rate)
            tqdm.write(s)

        if self.files:

            self.epochs.append(iteration)

            self.rpn_prob_loss.append(recorded_losses['rpn_prob'])
            self.rpn_bbox_loss.append(recorded_losses['rpn_bbox'])
            self.rpn_loss.append(recorded_losses['rpn'])

            self.clss_reg_prob_loss.append(recorded_losses['clss_reg_prob'])
            self.clss_reg_bbox_loss.append(recorded_losses['clss_reg_bbox'])
            self.clss_reg_loss.append(recorded_losses['clss_reg'])

            self.learning_rate.append(learning_rate)

    def record_inference(self, inferences):

        for ith, inference in enumerate(inferences):

            epoch, img, annotations = inference[:3]
            expanded_annotations, table_annotations_dbg, proposals, all_probs_object, anchors = inference[3:8]
            probs_object, filtered_proposals = inference[8:10]
            clss_score, pred_clss_idxs, bboxes = inference[10:13]

            if self.create_folder:
                os.mkdir('output/rpn/img_{}/'.format(ith))
                os.mkdir('output/final_rpn/img_{}/'.format(ith))
                os.mkdir('output/final/img_{}/'.format(ith))

            img = Image.fromarray(img.astype(np.uint8))
            init_rpn_img = img.copy()
            final_rpn_img = img.copy()
            final_img = img.copy()

            # jogar esses annotatio as bboxx e tal para um arquivo para que consiga saber de ql imagem eh o bbox

            # ## INIT RPN ###
            draw = ImageDraw.Draw(init_rpn_img)

            idxs = expanded_annotations[:, -1] > 0.0
            anchors = anchors[idxs]
            proposals = proposals[idxs]
            all_probs_object = all_probs_object[idxs]

            for b in range(annotations.shape[0]):  # ### esse trecho de codigo todo aqui dentro pode ser deixado bem mais simples e similar aos outros de modo a ter uma unoca funcao pra tudo

                # self._draw_gt_bbox(draw, annotation[b])
                # draw.rectangle([annotation[b, 0], annotation[b, 1], annotation[b, 2], annotation[b, 3]], outline='green')
                # print('Annotation as bbox: ', annotation[b, 0], annotation[b, 1], annotation[b, 2], annotation[b, 3])

                for a in np.argwhere(table_annotations_dbg == b):

                    anchor_idx = a[0]
                    self._draw_anchor_bbox(draw, anchors[anchor_idx])
                    self._draw_obj_bbox(draw, proposals[anchor_idx], all_probs_object[anchor_idx, 1])
                    # print('Anchors as bbox: ', anchors[anchor_idx, 0], anchors[anchor_idx, 1], anchors[anchor_idx, 2], anchors[anchor_idx, 3])
                    # print('Proposals as bbox: ', proposals[anchor_idx, 0], proposals[anchor_idx, 1], proposals[anchor_idx, 2], proposals[anchor_idx, 3])
                    # print('Prob (not obj, obj): ', all_probs_object[anchor_idx, 0], all_probs_object[anchor_idx, 1])
                    # print()

            # init_rpn_img.show()
            init_rpn_img.save('output/rpn/img_{}/epoch_{}.jpg'.format(ith, epoch))
            self.writer.add_image('img_{}/begin-RPN'.format(ith), np.array(init_rpn_img), epoch, dataformats='HWC')

            # ## FINAL RPN ###
            draw = ImageDraw.Draw(final_rpn_img)

            self._draw_annotations(draw, annotations)

            # Since it is sorted by probs, the highest ones are drawn last to a better visualization.
            for a in reversed(range(probs_object.shape[0])):
                self._draw_obj_bbox(draw, filtered_proposals[a], probs_object[a])

                # print('Proposals as bbox: ', filtered_proposals[a, 0], filtered_proposals[a, 1], filtered_proposals[a, 2], filtered_proposals[a, 3])
                # print('Prob: ', probs_object[a])
                # print()

            final_rpn_img.save('output/final_rpn/img_{}/epoch_{}.jpg'.format(ith, epoch))
            self.writer.add_image('img_{}/end-RPN'.format(ith), np.array(final_rpn_img), epoch, dataformats='HWC')

            # ## FINAL ###
            draw = ImageDraw.Draw(final_img)

            self._draw_annotations(draw, annotations)

            assert pred_clss_idxs.shape == clss_score.shape

            self._draw_predictions(draw, bboxes, pred_clss_idxs, clss_score)

            final_img.save('output/final/img_{}/epoch_{}.jpg'.format(ith, epoch))
            self.writer.add_image('img_{}/final-output'.format(ith), np.array(final_img), epoch, dataformats='HWC')

        if self.create_folder:
            self.create_folder = False

        self.writer.flush()

    def show_anchors(self, rpn_anchors, image_size):

        def _show_anchors(anchors_np, image_size, filename, just_center=False):
            offset = 128
            img_np = np.zeros((image_size[1] + 2 * offset, image_size[0] + 2 * offset, 3))
            real = Image.fromarray(img_np.astype(np.uint8))
            real_draw = ImageDraw.Draw(real)
            # -1 and +1 for draw the box outside the boundary. The inside content is the content of the bbox/anchor
            real_draw.rectangle([offset - 1, offset - 1, offset + image_size[0] - 1 + 1, offset + image_size[1] - 1 + 1], outline='yellow')

            for i in range(anchors_np.shape[0]):

                if just_center:
                    aw = anchors_np[i, 2] - anchors_np[i, 0] + 1.0
                    ah = anchors_np[i, 3] - anchors_np[i, 1] + 1.0
                    acw = anchors_np[i, 0] + 0.5 * (aw - 1.0)
                    ach = anchors_np[i, 1] + 0.5 * (ah - 1.0)

                    a0 = acw - 1.0
                    a1 = ach - 1.0
                    a2 = acw + 1.0
                    a3 = ach + 1.0
                    real_draw.rectangle([offset + a0, offset + a1, offset + a2, offset + a3], outline='magenta')
                else:
                    self._draw_anchor(real_draw, anchors_np, i, offset, 'magenta')

            # real.show()
            real.save('output/anchors/{}.jpg'.format(filename))

        anchors_np = rpn_anchors.detach().cpu().numpy()

        _show_anchors(anchors_np, image_size, 'valid_anchors')
        _show_anchors(anchors_np, image_size, 'valid_centers', just_center=True)

    def show_associated_positive_anchors(self, img_number, anchors_np, expanded_annotations_np, annotations_np, image_size, table_annotations_dbg_np):

        # Filter out the background paddings (due to background anchors)
        # TODO now, expanded_annotations_np has no background, so just put an assertion here.
        anchors_np = anchors_np[expanded_annotations_np[:, -1] > 0.0]

        for gti in range(annotations_np.shape[0]):

            offset = 128
            img_np = np.zeros((image_size[1] + 2 * offset, image_size[0] + 2 * offset, 3))
            real = Image.fromarray(img_np.astype(np.uint8))
            real_draw = ImageDraw.Draw(real)
            real_draw.rectangle([offset - 1, offset - 1, offset + image_size[0] - 1 + 1, offset + image_size[1] - 1 + 1], outline='yellow')  # atencao para o -1 e +1 e seu significado !

            for pai in np.argwhere(table_annotations_dbg_np == gti):
                self._draw_anchor(real_draw, anchors_np, pai[0], offset, 'magenta')

            self._draw_anchor(real_draw, annotations_np, gti, offset, 'green')

            real.save('output/anchors/associated_positive_anchors/{}_{}.jpg'.format(img_number, gti))

    def show_masked_anchors(self, e, anchors, rpn_labels, expanded_annotations, annotations, image_size, table_annotations_dbg):

        e = e.detach().cpu().numpy()[0]
        anchors_np = anchors.detach().cpu().numpy()
        mask_np = rpn_labels.detach().cpu().numpy()
        expanded_annotations_np = expanded_annotations.detach().cpu().numpy()
        annotations_np = annotations.detach().cpu().numpy()
        table_annotations_dbg_np = table_annotations_dbg.detach().cpu().numpy()

        self.show_associated_positive_anchors(e, anchors_np, expanded_annotations_np, annotations_np, image_size, table_annotations_dbg_np)

        for mask, mask_name in zip([-1.0, 0.0, 1.0], ['middle', 'negative', 'positive']):

            masked_anchors_np = anchors_np[mask_np == mask, :]

            for i in range(masked_anchors_np.shape[0]):

                offset = 128
                img_np = np.zeros((image_size[1] + 2 * offset, image_size[0] + 2 * offset, 3))
                real = Image.fromarray(img_np.astype(np.uint8))
                real_draw = ImageDraw.Draw(real)
                real_draw.rectangle([offset - 1, offset - 1, offset + image_size[0] - 1 + 1, offset + image_size[1] - 1 + 1], outline='yellow')  # atencao para o -1 e +1 e seu significado !

                for bi in range(annotations_np.shape[0]):

                    self._draw_anchor(real_draw, annotations_np, bi, offset, 'green')

                self._draw_anchor(real_draw, masked_anchors_np, i, offset, 'magenta')

                aw = masked_anchors_np[i, 2] - masked_anchors_np[i, 0] + 1.0
                ah = masked_anchors_np[i, 3] - masked_anchors_np[i, 1] + 1.0
                acw = masked_anchors_np[i, 0] + 0.5 * (aw - 1.0)
                ach = masked_anchors_np[i, 1] + 0.5 * (ah - 1.0)

                a0 = acw - 1.0
                a1 = ach - 1.0
                a2 = acw + 1.0
                a3 = ach + 1.0
                real_draw.rectangle([offset + a0, offset + a1, offset + a2, offset + a3], outline='yellow')

                real.save('output/anchors/{}/{}_{}.jpg'.format(mask_name, e, i))

    def _draw_anchor(self, draw_obj, bboxes, idx, offset, color):

        x0 = bboxes[idx, 0]
        y0 = bboxes[idx, 1]
        x1 = bboxes[idx, 2]
        y1 = bboxes[idx, 3]
        draw_obj.rectangle([offset + x0, offset + y0, offset + x1, offset + y1], outline=color)
