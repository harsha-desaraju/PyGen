# This file is to draw using genetic algorithm

import sys
sys.path.append("D://Projects//PyGen")
import cv2
import numpy as np
from optimizers._binary import DiscreteOptimizer
from crossover._crossover import UniformCrossover, KPointCrossover
from selection._selection import RouletteWheelSelection, TounamentSelection
import matplotlib.pyplot as plt



def lab2bgr(arr):
    bgr = cv2.cvtColor(
        np.uint8([[arr]]),
        cv2.COLOR_LAB2BGR
    )[0][0]
    return bgr.tolist()

def bgr2lab(arr):
    lab = cv2.cvtColor(
        np.uint8([[arr]]),
        cv2.COLOR_BGR2LAB
    )[0][0]
    return lab.tolist()


def generate_bounds(max_r=50):
    # Variables for a cirle are 
    # `position`, `radius`, `color`
    # `position`: (x,y)
    # `radius`  : r
    # `color`   : L,a,b (CIELAB color space)

    # Bounds for the variables
    mn_x, mx_x = 0, img_size[0]
    mn_y, mx_y = 0, img_size[1]
    mn_r, mx_r = 0, max_r
    mn_l, mx_l = 0, 255
    # mn_a, mx_a = -127, 127
    # mn_b, mx_b = -127, 127

    mn = np.array([mn_x, mn_y, mn_r, mn_l])
    mx = np.array([mx_x, mx_y, mx_r, mx_l])

    mn_bound = [mn for _ in range(n_bch)]
    mx_bound = [mx for _ in range(n_bch)]

    mn_bound = np.concatenate(mn_bound)
    mx_bound = np.concatenate(mx_bound)

    return (mn_bound, mx_bound)


def generate_image(chrom):
    # print(chrom)
    img = np.zeros(ref_img.shape, dtype=np.uint8)

    for i in range(n_bch):
        img = cv2.circle(
            img, 
            center=(chrom[i*n_dgf+0], chrom[i*n_dgf+1]),
            radius= chrom[i*n_dgf+2],
            color=int(chrom[i*n_dgf+3]),
            thickness=-1
        )

    return img


class fitness:
    def __init__(self, ref_img, prev_gen_img, mask):
        self.ref_img = ref_img
        self.prev_gen_img = prev_gen_img
        self.mask = mask
        
    def cost(self, chrom):
        # A simple pixel to pixel difference
        # between the generated image and the
        # reference image
        img = generate_image(chrom)
        tar = self.ref_img

        return np.sum(np.abs(img - tar)**2)

        # img = self.mask*img
        # tar = self.mask*tar

        # diff1 = cv2.subtract(tar, img)
        # diff2 = cv2.subtract(img, tar)
        # err = cv2.add(diff1, diff2)
        # return np.sum(err)

        # return np.sum(np.abs(tar - img))




def create_mask(img, kernel):
    img = cv2.GaussianBlur(img, kernel, 0)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    fig, ax = plt.subplots(1, 2)

    # ax[0].imshow(hsv_img[:, :, 0])
    # ax[1].imshow(hsv_img[:, :, 1])
    # ax[2].imshow(hsv_img[:, :, 2])

    thres = 5
    mask = np.where(hsv_img[:,:,1]>thres, 1, 0)
    mask = mask.astype(np.uint8)
    # ax[0].imshow(mask)
    # ax[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)*mask, cmap='grey')
    # plt.show()

    return mask
    


def calc_sampling_mask(img_grey, blur_percent):
    img = np.copy(img_grey)
    # Calculate gradient 
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    # Python Calculate gradient magnitude and direction ( in degrees ) 
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    #calculate blur level
    w = img.shape[0] * blur_percent
    if w > 1:
        mag = cv2.GaussianBlur(mag,(0,0), w, cv2.BORDER_DEFAULT)
    #ensure range from 0-255 (mostly for visual debugging, since in sampling we will renormalize it anyway)
    scale = 255.0/mag.max()

    # fig, ax = plt.subplots(1,2)

    # ax[0].imshow(mag*scale, cmap='grey')
    # ax[1].imshow(mag*scale*img_grey, cmap='grey')
    # plt.show()

    return mag*scale



if __name__ == '__main__':
    
    # img_size = (500, 500)
    img_pth ="draw/target_small.png"
    n_dgf = 4
    n_gen = 300
    n_stg = 1
    n_pop = 100
    n_bch = 1


    clr_ref_img = cv2.imread(img_pth,cv2.IMREAD_COLOR)
    ref_img = cv2.cvtColor(clr_ref_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Img", ref_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_size = ref_img.shape
    ref_img = ref_img.astype(np.uint8)

    
    prev_gen_image = np.full(ref_img.shape, 0, dtype=np.uint8)

    print(np.sum(np.abs(prev_gen_image - ref_img)**2))

    for stg in range(n_stg):
        mask = create_mask(clr_ref_img, (45-stg*6, 45-stg*6))
        bnds = generate_bounds(50-stg*5)
        optimizer = DiscreteOptimizer(
            n_pop, n_gen, chrom_len=n_dgf*n_bch, 
            bounds=bnds, selection_rate= 0.5,
            elite= 0.2, n_child=2, n_jobs=None,
            rnd_state=None, mu=0.05
            )
        scores, pop = optimizer.optimize(
            fitness(ref_img, prev_gen_image, mask), 
            KPointCrossover, TounamentSelection, 
            verbose=True, crossover_args={'k':1}
        )

        prev_gen_image = generate_image(pop[0])
        print(f"The error after stage {stg+1} is {scores[0]}")



    cv2.imshow("Gen Image", prev_gen_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

   