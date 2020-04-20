import torch
from feature_extractor import FeatureExtractorNet
from feature_extractor_complete import FeatureExtractorNetComplete
from rpn import RPN
from roi import ROI
from classifier_regressor import ClassifierRegressor
from loss import anchor_labels, get_target_distance, get_target_distance2, get_target_mask, compute_prob_loss
import config
from visualizer import Viz
from tqdm import tqdm, trange
from dataset_loader import inv_normalize
from bbox_utils import offset2bbox, anchors_offset2bbox
import torch.nn.functional as F

class FasterRCNN:
    
    def __init__(self, device):

        self.device = device

        # define the net components
        self.fe_net = FeatureExtractorNet().to(self.device)
        # self.fe_net = FeatureExtractorNetComplete().to(self.device)
        self.rpn_net = RPN(input_img_size=config.input_img_size, feature_extractor_out_dim=self.fe_net.out_dim, feature_extractor_size=self.fe_net.feature_extractor_size, receptive_field_size=self.fe_net.receptive_field_size, device=device).to(self.device)
        self.roi_net = ROI(input_img_size=config.input_img_size).to(self.device)
        self.clss_reg = ClassifierRegressor(input_img_size=config.input_img_size, input_size=7*7*self.fe_net.out_dim, n_classes=1).to(self.device)

        self.viz = Viz(tensorboard=True, files=True, screen=True)


    def train(self, train_dataloader, test_dataset):

        params = list(self.fe_net.parameters())  + \
                 list(self.rpn_net.parameters()) + \
                 list(self.roi_net.parameters()) + \
                 list(self.clss_reg.parameters())

        params = [p for p in params if p.requires_grad == True]

        optimizer = torch.optim.Adam(params, lr=0.001)

        for net in [self.fe_net, self.rpn_net, self.roi_net, self.clss_reg]:
            net.train()

        l = len(train_dataloader)

        for e in trange(1, config.epochs+1):

            rpn_prob_loss_epoch, rpn_bbox_loss_epoch, rpn_loss_epoch = 0, 0, 0
            clss_reg_prob_loss_epoch, clss_reg_bbox_loss_epoch, clss_reg_loss_epoch = 0, 0, 0
            total_loss_epoch = 0
            
            # change variable names to be more intuitive
            # label -> class annotation
            # gtbbox -> bbox annotation
            for img, annotation, _, labels_objectness, labels_class, table_gts_positive_anchors in train_dataloader:

                # show_training_sample(inv_normalize(img[0, :, :, :].clone().detach()).permute(1, 2, 0).numpy().copy(), annotation[0].detach().numpy().copy())

                # img.size()                        -> torch.Size([1, 3, input_img_size[0], input_img_size[1]])
                # annotation.size()                 -> torch.Size([1, #bboxes_in_img, 4])
                # labels.size()                     -> torch.Size([1, #valid_anchors])
                #                                      -1, 0 and 1 for dont care, negative and positive, respectively
                # table_gts_positive_anchors.size() -> torch.Size([1, #positive_anchors, 2]) 
                #                                      [idx of gt box, idxs of its assigned anchor on labels]

                # This implemention only supports one image per batch
                # Every batch channel is removed except for the image which will be forwarded through the feature extractor
                assert img.size(0) == annotation.size(0) == labels_objectness.size(0) == labels_class.size(0) == table_gts_positive_anchors.size(0) == 1
                img, annotation = img.to(self.device), annotation[0, :, :].to(self.device)
                labels_objectness, labels_class, table_gts_positive_anchors = labels_objectness[0, :].to(self.device), labels_class[0, :].to(self.device), table_gts_positive_anchors[0, :, :].to(self.device)
                # img.size()                        -> torch.Size([1, 3, input_img_size[0], input_img_size[1]])
                # annotation.size()                 -> torch.Size([#bboxes_in_img, 4])
                # labels.size()                     -> torch.Size([#valid_anchors])
                # table_gts_positive_anchors.size() -> torch.Size([#positive_anchors, 2]) 

                # print(table_gts_positive_anchors)
                # print(labels_objectness) -> already balanced
                # print(labels_class)      -> not balanced yet
                # exit()

                optimizer.zero_grad()

                features = self.fe_net.forward(img)
                # features.size() -> torch.Size([1, fe.out_dim, fe.feature_extractor_size, fe.feature_extractor_size])

                # The RPN handles the batch channel. The input (features) has the batch channel (asserted to be 1)
                # and outputs all the objects already handled
                proposals, cls_out, filtered_proposals, probs_object, filtered_labels_class = self.rpn_net.forward(features, labels_class)
                # proposals.size()          -> torch.Size([#valid_anchors, 4])
                # cls_out.size()            -> torch.Size([#valid_anchors, 2])
                # filtered_proposals.size() -> torch.Size([#filtered_proposals, 4])
                # probs_object.size()       -> torch.Size([#filtered_proposals]) #NOTE just for visualization.. temporary
                # The features object has its batch channel kept due to later use

                # The filtered_proposals will act as the anchors in the RPN
                # and the table_gts_positive_proposals will act as the table_gts_positive_anchors in the RPN

                ## Compute RPN loss ##
                rpn_bbox_loss = get_target_distance(proposals, self.rpn_net.anchors, annotation, table_gts_positive_anchors)
                rpn_prob_loss = compute_prob_loss(cls_out, labels_objectness)
                #####

                # rpn_loss = 10 * rpn_prob_loss + rpn_bbox_loss
                rpn_loss = rpn_prob_loss + rpn_bbox_loss

                rpn_prob_loss_epoch += rpn_prob_loss.item()
                rpn_bbox_loss_epoch += rpn_bbox_loss.item()
                rpn_loss_epoch += rpn_loss.item()

                # if there is any proposal which is classified as an object
                if filtered_proposals.size(0) > 0: 

                    rois = self.roi_net.forward(filtered_proposals, features)
                    # rois.size()      -> torch.Size([#filtered_proposals, fe.out_dim, roi_net.out_dim, roi_net.out_dim])

                    raw_reg, raw_cls = self.clss_reg.forward(rois)
                    # raw_reg.size()   -> torch.Size([#filtered_proposals, 4])
                    # raw_cls.size()   -> torch.Size([#filtered_proposals, 2])

                    #####
                    ## Compute class_reg loss ##
                    # essa linha de baixo tinha quee star logo abaixo da liha do rpn forward apos a filtragem
                    # e no rpn n deveria ter aquela filtragem..
                    # todo (ver se eh isso msm):
                    # tirar parte de filtragem do rpn e colocar aqui (depois pensa em função)
                    # na filtragem, vai filtrar tbm com a COND (uma matriz de filtragem) para filtrar usando os proprios indices da table_gts_positive_anchors 
                    # para saber se vai pra frente ou n, ou seja, gerando uma nova table_gts_positive_anchors para o regressor. com isso, a primeira coluna consegue indexas as classes.

                    table_fgs_positive_proposals, cls_mask = get_target_mask(filtered_proposals, annotation, filtered_labels_class)
                    clss_reg_bbox_loss = get_target_distance2(raw_reg, filtered_proposals, annotation, table_fgs_positive_proposals)
                    if (cls_mask != -1.0).sum() > 0:
                        clss_reg_prob_loss = compute_prob_loss(raw_cls, cls_mask)
                        clss_reg_loss = clss_reg_prob_loss + clss_reg_bbox_loss
                        clss_reg_prob_loss_epoch += clss_reg_prob_loss.item()
                    else:
                        clss_reg_loss = clss_reg_bbox_loss
                    #####

                    clss_reg_bbox_loss_epoch += clss_reg_bbox_loss.item()
                    clss_reg_loss_epoch += clss_reg_loss.item()

                    total_loss = rpn_loss + clss_reg_loss
                    total_loss_epoch += total_loss.item()
                    show_all_results = True
                
                else:

                    total_loss = rpn_loss
                    show_all_results = False

                total_loss.backward()

                optimizer.step()

            self.viz.record(e, rpn_prob_loss_epoch / l, rpn_bbox_loss_epoch / l, rpn_loss_epoch / l, clss_reg_prob_loss_epoch / l, clss_reg_bbox_loss_epoch / l, clss_reg_loss_epoch / l, total_loss_epoch / l)

            if e % 10 == 0:
                
                output = self.infer(e, test_dataset)
                self.viz.record_inference(output)

                for net in [self.fe_net, self.rpn_net, self.roi_net, self.clss_reg]: net.train()


    # NOTE note que este codigo eh identico ao do treino porem sem a loss e backward.. teria como fazer essa funcao funcionar para ambos treino e inferencia?
    # quero mostrar tbm na iter zero, antes de iniciar o treino
    def infer(self, epoch, dataset):

        output = []

        for net in [self.fe_net, self.rpn_net, self.roi_net, self.clss_reg]: net.eval()
        
        with torch.no_grad():
            
            # for ith, (img, annotation, labels, table_gts_positive_anchors) in enumerate(dataloader):
            # there is a random number being generated inside the Dataloader: https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
            # in the final version, use the dataloader if is more fancy
            for ith in range(len(dataset)):
                
                img, annotation, clss_idxs, labels_objectness, labels_class, table_gts_positive_anchors = dataset[ith]
                
                img = img.unsqueeze(0)
                annotation = annotation.unsqueeze(0)
                clss_idxs = clss_idxs.unsqueeze(0)
                labels_objectness = labels_objectness.unsqueeze(0)
                labels_class = labels_class.unsqueeze(0)
                table_gts_positive_anchors = table_gts_positive_anchors.unsqueeze(0)

                assert img.size(0) == annotation.size(0) == clss_idxs.size(0) == labels_objectness.size(0) == labels_class.size(0) == table_gts_positive_anchors.size(0) == 1
                img, annotation, clss_idxs = img.to(self.device), annotation[0, :, :].to(self.device), clss_idxs[0, :].to(self.device)
                labels_objectness, labels_class, table_gts_positive_anchors = labels_objectness[0, :].to(self.device), labels_class[0, :].to(self.device), table_gts_positive_anchors[0, :, :].to(self.device)

                features = self.fe_net.forward(img)
                # proposals, cls_out, filtered_proposals, probs_object = self.rpn_net.forward(features)
                proposals, cls_out, filtered_proposals, probs_object, filtered_labels_class = self.rpn_net.forward(features, labels_class)

                # if there is any proposal which is classified as an object
                if filtered_proposals.size(0) > 0: # this if has to be implemented inside the visualization?

                    rois = self.roi_net.forward(filtered_proposals, features)
                    raw_reg, raw_cls = self.clss_reg.forward(rois)

                    show_all_results = True

                    refined_proposals, clss_score, pred_clss_idxs = self.clss_reg.infer_bboxes(filtered_proposals, raw_reg, raw_cls)

                else:

                    clss_score = None
                    pred_clss_idxs = None
                    show_all_results = False

                ith_output = [epoch]
                    
                ith_output += [inv_normalize(img[0, :, :, :].clone().detach())]
                ith_output += [offset2bbox(annotation)]
                ith_output += [clss_idxs]
                
                ith_output += [table_gts_positive_anchors]
                ith_output += [offset2bbox(proposals)]
                ith_output += [F.softmax(cls_out, dim=1)]
                ith_output += [anchors_offset2bbox(self.rpn_net.anchors)]

                ith_output += [show_all_results]

                ith_output += [probs_object]
                ith_output += [offset2bbox(filtered_proposals)]

                if show_all_results:
                
                    ith_output += [clss_score]
                    ith_output += [pred_clss_idxs]
                    ith_output += [offset2bbox(refined_proposals)]

                else:

                    ith_output += [None]
                    ith_output += [None]
                    ith_output += [None]

                output.append(ith_output)

        return output
