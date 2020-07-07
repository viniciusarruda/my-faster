# TODO: standardize to more intuitive names and capslock?

verbose = False

# Dataset used to train
# img_folder = 'dataset/Cityscapes/all_val/'
# annotations_file = 'dataset/Cityscapes/one_val.csv'

voc_folder = 'dataset/CityscapesVOC/VOC/'
set_type = 'val'

# Dataset used to validate
# val_img_folder = 'dataset/Cityscapes/all_val/'
# val_annotations_file = 'dataset/Cityscapes/one_val.csv'

val_voc_folder = 'dataset/CityscapesVOC/VOC/'
val_set_type = 'val'

# The class '__background__' must always be at zero index and the user should not use it to label your data, unless you know what you are doing.
class_names = ['__background__', 'car', 'rider', 'person', 'motorcycle', 'bicycle', 'train', 'bus', 'truck']
# class_names = ['__background__', 'car', 'person']
n_classes = len(class_names)

original_img_size = (2048, 1024)  # should get from data, maybe in the __getitem__ function
# original_img_size = (256, 256)  # should get from data, maybe in the __getitem__ function
# Size that all images will be resized to
# The annotations will be adjusted accordinly
# (width, height) format
# input_img_size = (1200, 600)
input_img_size = (1024, 512)
# input_img_size = (512, 256)
# input_img_size = (224, 224)

# minimun bbox width/height size
min_size = 4.0  # training only
# min_size = 8.0  # training only
# min_size = 16.0  # inference only

# Available backbones: Toy, ResNet
backbone = 'ResNet18'

# RPN batch size
rpn_batch_size = 256
max_positive_batch_ratio = 0.5

# Classifier/Regressor batch size
batch_size = 128
fg_fraction = 0.25

# Number of epochs to run
epochs = 10 #25  # 1000

# RPN anchor ratios and scales
rpn_anchor_ratios = [0.5, 1, 2]
rpn_anchor_scales = [4, 8] #[4, 8, 16, 32]
# rpn_anchor_ratios = [0.8, 1, 1.2]
# rpn_anchor_scales = [3.5, 4, 4.5]

# Number of top (according to score) proposals kept before NMS. Zero to not filter.
train_pre_nms_top_n = 12000
test_pre_nms_top_n = 6000

# Number of top (according to score) proposals kept after NMS. Zero to not filter.
train_pos_nms_top_n = 2000
test_pos_nms_top_n = 300


# Visualization

GT_COLOR = 'Chartreuse'
ANCHOR_COLOR = 'Magenta'
OBJ_COLOR = 'DodgerBlue'

COLORS = [
    'AliceBlue', 'Aqua', 'Yellow', 'BlueViolet',
    'BurlyWood', 'CadetBlue', 'AntiqueWhite', 'Bisque',
    'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Green', 'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'YellowGreen', 'Azure', 'Beige', 'BlanchedAlmond', 'Chocolate',
    'Black', 'Blue', 'Aquamarine'
]

NCOLORS = len(COLORS)
