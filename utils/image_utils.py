from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

class ImgSizer:
    def __init__(self):
        pass
    def sizer(self, imPath, angle):
        desired_size = 512
        im = Image.open(imPath)
        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        # use thumbnail() or resize() method to resize the input image

        # thumbnail is a in-place operation

        # im.thumbnail(new_size, Image.ANTIALIAS)
        wpercent = (desired_size/float(im.size[0]))
        hsize = int((float(im.size[1])*float(wpercent)))
        im = im.resize((desired_size,hsize), Image.Resampling.LANCZOS)

        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        # use thumbnail() or resize() method to resize the input image

        new_im = Image.new('L',(desired_size, desired_size))
        new_im.paste(im, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))
        new_im.save(f'{angle}_mask.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", type=str, required= True, help="path to front view image")
    parser.add_argument("--side", type=str, required= True, help="path to side view image")
    args = parser.parse_args()

    ImgSizer().sizer(args.front, 'front')
    ImgSizer().sizer(args.side, 'side')