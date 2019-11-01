from PIL import Image
import numpy as np
import tqdm
# https://stackoverflow.com/questions/53658255/how-to-get-the-background-from-multiple-images-by-removing-moving-objects

im1 = Image.open('/home/tducnguyen/Pictures/1.png')
im2 = Image.open('/home/tducnguyen/Pictures/2.png')
pix1 = im1.load()
pix2 = im2.load()


merge = im1
pix_merge = merge.load()

# print(im.size)  # Get the width and hight of the image for iterating over
# print(pix[x, y])  # Get the RGBA Value of the a pixel of an image
# pix[x, y] = value  # Set the RGBA Value of the image (tuple)
width = im1.size[0]
hight = im1.size[1]

for w in range(width):
    for h in range(hight):
        pix_rgb = np.zeros(3, dtype=int)
        for color in range(3):
            pix_val = int((pix1[w, h][color] + pix2[w, h][color]) / 2)
            if pix_val > 255:
                pix_val = 255
            pix_rgb[color] = pix_val

        pix_merge[w, h] = tuple(pix_rgb)

merge.save('/home/tducnguyen/Pictures/merge.png')  # Save the modified pixels as .png
print()

