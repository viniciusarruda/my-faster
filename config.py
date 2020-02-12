

# Dataset used to train
img_folder = 'dataset/one_day/'
annotations_file = 'dataset/one_three_day.csv'

# Size that all images will be resized to
# The annotations will be adjusted accordinly
input_img_size = (224, 224) # (600, 600)

# RPN batch size
rpn_batch_size = 256

# Number of epochs to run
epochs = 1000

# RPN anchor ratios and scales
rpn_anchor_ratios = [0.8, 1, 1.2]  #[1] #[0.5, 1, 2] 
rpn_anchor_scales = [3.5, 4, 4.5]  #[4] #[8, 16, 32]

# Number of top (according to score) proposals kept before NMS. Zero to not filter.
pre_nms_top_n = 30

# Number of top (according to score) proposals kept after NMS. Zero to not filter.
pos_nms_top_n = 0

# NMS threshold
nms_threshold = 0.7
