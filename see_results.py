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
