import os
import shutil
import matplotlib.pyplot as plt

def setup_dir(dir):
    shutil.rmtree(dir, ignore_errors=True)
    os.mkdir(dir)

def show_img(img):
    plt.imshow(img.permute(1,2,0))
    plt.xticks([])
    plt.yticks([])
    plt.show()