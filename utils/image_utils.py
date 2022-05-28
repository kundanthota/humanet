from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default='demo')
    parser.add_argument("--output_folder", type=str, default='final_demo')
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()


    try:
        os.mkdir(args.output_folder)
    except:
        pass

    for subject in os.listdir(args.input_folder):
        desired_size = 512
        im_pth = f"{args.input_folder}/{subject}"

        im = Image.open(im_pth)
        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        # use thumbnail() or resize() method to resize the input image

        # thumbnail is a in-place operation

        # im.thumbnail(new_size, Image.ANTIALIAS)

        im = im.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = Image.new("L", (desired_size, desired_size))
        new_im.paste(im, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))

        plt.imsave(f'{args.output_folder}/{subject}', new_im, cmap='gray')