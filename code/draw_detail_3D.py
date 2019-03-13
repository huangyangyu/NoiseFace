#!/usr/bin/env python
#coding: utf-8

from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import cv2
import math
import json
import numpy as np
import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
plt.switch_backend("agg")

bins = 201


def choose_iter(_iter, interval):
    if _iter == 1000:
        return True
    if _iter == 200000:
        return True

    k = _iter / interval
    if k % 1 != 0 or (k / 1) % (_iter / 40000 + 1) != 0:
        return False
    else:
        return True


def get_pdf(X):
    pdf = bins * [0.0]
    for x in X:
        pdf[int(x*100)] += 1
    return pdf


def mean_filter(pdf, fr=2):
    filter_pdf = bins * [0.0]
    for i in xrange(fr, bins-fr):
        for j in xrange(i-fr, i+fr+1):
            filter_pdf[i] += pdf[j] / (fr+fr+1)
    return filter_pdf


def load_lfw(prefix):
    items = list()
    for line in open("./data/%s_lfw.txt" % prefix):
        item = line.strip().split()
        items.append((int(item[0]), float(item[1]), 100.0 * float(item[2])))
    items.append(items[-1])
    return items


def draw_imgs(prefix, ratio, mode):
    lfw = load_lfw(prefix)

    frames = 0
    _X = list()
    _Y = list()
    _Z = list()
    iters = list()
    threds = list()
    accs = list()
    alphas = list()
    betas = list()
    gammas = list()
    for _iter in xrange(1000, 200000+1, 1000):
        _iter1, _thred1, _acc1 = lfw[_iter/10000]
        _iter2, _thred2, _acc2 = lfw[_iter/10000+1]
        _thred = _thred1 + (_thred2 - _thred1) * (_iter - _iter1) / max(1, (_iter2 - _iter1))
        _acc = _acc1 + (_acc2 - _acc1) * (_iter - _iter1) / max(1, (_iter2 - _iter1))
        iters.append(_iter)
        threds.append(_thred)
        accs.append(_acc)
        
        log_file = "./data/%s_%d.txt" % (prefix, _iter)
        if not os.path.exists(log_file):
            print "miss: %s" % log_file
            continue
        lines = open(log_file).readlines()
        
        # pdf
        pdf = list()
        for line in lines[3:bins+3]:
            item = line.strip().split()
            item = map(lambda x: float(x), item)
            pdf.append(item[1])

        # clean pdf
        clean_pdf = list()
        for line in lines[bins+3+1:bins+bins+3+1]:
            item = line.strip().split()
            item = map(lambda x: float(x), item)
            clean_pdf.append(item[1])

        # noise pdf
        noise_pdf = list()
        for line in lines[bins+bins+3+1+1:bins+bins+bins+3+1+1]:
            item = line.strip().split()
            item = map(lambda x: float(x), item)
            noise_pdf.append(item[1])

        # pcf
        pcf = list()
        for line in lines[bins+bins+bins+3+1+1+1:bins+bins+bins+bins+3+1+1+1]:
            item = line.strip().split()
            item = map(lambda x: float(x), item)
            pcf.append(item[1])

        # weight
        W = list()
        for line in lines[bins+bins+bins+bins+3+1+1+1+1:bins+bins+bins+bins+bins+3+1+1+1+1]:
            item = line.strip().split()
            item = map(lambda x: float(x), item)
            W.append(item[1])

        X = list()
        for i in xrange(bins):
            X.append(i * 0.01 - 1.0)
        _X.append(X)
        _Y.append(bins * [_iter])
        _Z.append(mean_filter(pdf))
        
        if not choose_iter(_iter, 1000):
            continue

        titlesize = 44
        asize = 44
        glinewidth = 2

        fig = plt.figure(0)
        fig.set_size_inches(24, 18)
        ax = Axes3D(fig)
        #ax.set_title(r"$The\ cos\theta\ distribution\ of\ $" + str(ratio) + "%" + r"$\ noisy\ training\ data\ over\ iteration$", fontsize=titlesize)
        ax.set_xlabel(r"$cos\theta$", fontsize=asize)
        ax.set_ylabel(r"$Iter$", fontsize=asize)
        ax.set_zlabel(r"$Numbers$", fontsize=asize)
        ax.tick_params(labelsize=32)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(0, 200000)
        ax.set_zlim(0.0, 6000.0)
        ax.grid(True, linewidth=glinewidth)
        surf = ax.plot_surface(_X, _Y, _Z, rstride=3, cstride=3, cmap=plt.cm.coolwarm, linewidth=0.1, antialiased=False)
        surf.set_clim([0, 6000])
        cbar = fig.colorbar(surf, shrink=0.5, aspect=10, norm=plt.Normalize(0, 6000))
        cbar.set_ticks([0, 1000, 2000, 3000, 4000, 5000, 6000])
        cbar.set_ticklabels(["0", "1k", "2k", "3k", "4k", "5k", "6k"])
        #cbar.locator = ticker.MaxNLocator(nbins=6)
        #cbar.update_ticks()
        cbar.ax.tick_params(labelsize=24)
    
        #print dir(ax)
        #_ax = ax.twiny()
        #_ax.set_ylim(0.0, 1.0)
        #_ax.plot(bins * [-1.0], iters, accs, label="LFW")
        #_ax.legend()
        #ax.plot(len(iters) * [-1.0], iters, 100.0 * np.array(accs), color="k", label="LFW")
        #ax.plot(len(iters) * [-1.0], iters, 60.0 * np.array(accs), color="k", label="LFW")
        #ax.legend()
    
        plt.savefig("./figures/%s_3D_dist_%d.jpg" % (prefix, _iter))
        plt.close()

        frames += 1
        print "frames:", frames
        print "processed:", _iter
        sys.stdout.flush()


def draw_video(prefix1, prefix2, ratio):
    draw_imgs(prefix1, ratio, mode=1)
    draw_imgs(prefix2, ratio, mode=2)

    fps = 25
    #size = (4800, 1800) 
    #size = (2400, 900) 
    size = (2000, 750) 
    videowriter = cv2.VideoWriter("./figures/demo_3D_distribution_noise-%d%%.avi" % ratio, cv2.cv.CV_FOURCC(*"MJPG"), fps, size)
    for _iter in xrange(1000, 200000+1, 1000):
        if not choose_iter(_iter, 1000):
            continue
        image_file1 = "./figures/%s_3D_dist_%d.jpg" % (prefix1, _iter)
        img1 = cv2.imread(image_file1)
        img1 = cv2.resize(img1, (1000, 750))##
        h1, w1, c1 = img1.shape

        image_file2 = "./figures/%s_3D_dist_%d.jpg" % (prefix2, _iter)
        img2 = cv2.imread(image_file2)
        img2 = cv2.resize(img2, (1000, 750))##
        h2, w2, c2 = img2.shape

        assert h1 == h2 and w1 == w2
        img = np.zeros((size[1], size[0], 3), dtype=img1.dtype)

        img[0:h1, 0:w1, :] = img1
        img[0:h1, w1:w1+w2, :] = img2
        videowriter.write(img)


if __name__ == "__main__":
    prefixs = [
        (0, "p_casia-webface_noise-flip-outlier-1_1-0_Nsoftmax_exp", "p_casia-webface_noise-flip-outlier-1_1-0_Nsoftmax_FIT_exp"), \
        (20, "p_casia-webface_noise-flip-outlier-1_1-20_Nsoftmax_exp", "p_casia-webface_noise-flip-outlier-1_1-20_Nsoftmax_FIT_exp"), \
        (40, "p_casia-webface_noise-flip-outlier-1_1-40_Nsoftmax_exp", "p_casia-webface_noise-flip-outlier-1_1-40_Nsoftmax_FIT_exp"), \
        (60, "p_casia-webface_noise-flip-outlier-1_1-60_Nsoftmax_exp", "p_casia-webface_noise-flip-outlier-1_1-60_Nsoftmax_FIT_exp")
    ]

    for ratio, prefix1, prefix2 in prefixs:
        draw_video(prefix1, prefix2, ratio)
        print "processing noise-%d" % ratio

