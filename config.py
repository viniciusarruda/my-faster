

# Dataset used to train
img_folder = 'dataset/mini_day/'
annotations_file = 'dataset/mini_day.csv'

# Size that all images will be resized to
# The annotations will be adjusted accordinly
input_img_size = (600, 600) #(224, 224)

# RPN batch size
rpn_batch_size = 256

# Number of epochs to run
epochs = 1000

# RPN anchor ratios and scales
rpn_anchor_ratios = [0.5, 1, 2]  #[0.8, 1, 1.2]  #[1] #[0.5, 1, 2] 
rpn_anchor_scales = [8, 16, 32]  #[3.5, 4, 4.5]  #[4] #[8, 16, 32]

# Number of proposals kept before NMS
pre_nms_top_n = 30