#!/usr/bin/env python
#coding: utf-8

import os
import sys
import cv2
import math
import json
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")

bins = 201


def choose_iter(_iter, interval):
    if _iter == 1000:
        return True
    if _iter == 200000:
        return True

    k = _iter / interval
    if k % 2 != 0 or (k / 2) % (_iter / 20000 + 1) != 0:
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
    iters = list()
    threds = list()
    accs = list()
    alphas = list()
    betas = list()
    gammas = list()
    for _iter in xrange(1000, 200000+1, 100):
        if not choose_iter(_iter, 100):
            continue
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
        
        # X
        X = list()
        for i in xrange(bins):
            X.append(i * 0.01 - 1.0)
        
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

        # mean filtering
        filter_pdf = mean_filter(pdf)
        filter_clean_pdf = mean_filter(clean_pdf)
        filter_noise_pdf = mean_filter(noise_pdf)
        # pcf
        sum_filter_pdf = sum(filter_pdf)
        pcf[0] = filter_pdf[0] / sum_filter_pdf
        for i in xrange(1, bins):
            pcf[i] = pcf[i-1] + filter_pdf[i] / sum_filter_pdf
        # mountain
        l_bin_id_ = 0
        r_bin_id_ = bins-1
        while pcf[l_bin_id_] < 0.005:
            l_bin_id_ += 1
        while pcf[r_bin_id_] > 0.995:
            r_bin_id_ -= 1
        
        m_bin_id_ = (l_bin_id_ + r_bin_id_) / 2
        t_bin_id_ = 0
        for i in xrange(bins):
            if filter_pdf[t_bin_id_] < filter_pdf[i]:
                t_bin_id_ = i
        t_bin_ids_ = list()
        for i in xrange(max(l_bin_id_, 5), min(r_bin_id_, bins-5)):
            if filter_pdf[i] >= filter_pdf[i-1] and filter_pdf[i] >= filter_pdf[i+1] and \
               filter_pdf[i] >= filter_pdf[i-2] and filter_pdf[i] >= filter_pdf[i+2] and \
               filter_pdf[i]  > filter_pdf[i-3] and filter_pdf[i]  > filter_pdf[i+3] and \
               filter_pdf[i]  > filter_pdf[i-4] and filter_pdf[i]  > filter_pdf[i+4] and \
               filter_pdf[i]  > filter_pdf[i-5] and filter_pdf[i]  > filter_pdf[i+5]:
                t_bin_ids_.append(i)
                i += 5
        if len(t_bin_ids_) == 0:
            t_bin_ids_.append(t_bin_id_)
        if t_bin_id_ < m_bin_id_:
            lt_bin_id_ = t_bin_id_
            rt_bin_id_ = t_bin_ids_[-1] if t_bin_ids_[-1] > m_bin_id_ else None
            #rt_bin_id_ = max(t_bin_ids_[-1], m_bin_id_)
            #rt_bin_id_ = t_bin_ids_[-1]
        else:
            rt_bin_id_ = t_bin_id_
            lt_bin_id_ = t_bin_ids_[0] if t_bin_ids_[0] < m_bin_id_ else None
            #lt_bin_id_ = min(t_bin_ids_[0], m_bin_id_)
            #lt_bin_id_ = t_bin_ids_[0]
        
        # weight1-3
        weights1 = list()
        weights2 = list()
        weights3 = list()
        weights = list()
        r = 2.576
        alpha = np.clip((r_bin_id_ - 100.0) / 100.0, 0.0, 1.0)
        #beta = abs(2.0 * alpha - 1.0)
        beta = 2.0 - 1.0 / (1.0 + math.exp(5-20*alpha)) - 1.0 / (1.0 + math.exp(20*alpha-15))
        if mode == 1:
            alphas.append(100.0)
            betas.append(0.0)
            gammas.append(0.0)
        else:
            alphas.append(100.0*beta*(alpha<0.5))
            betas.append(100.0*(1-beta))
            gammas.append(100.0*beta*(alpha>0.5))
        for bin_id in xrange(bins):
            # w1
            weight1 = 1.0
            # w2
            if lt_bin_id_ is None:
                weight2 = 0.0
            else:
                #weight2 = np.clip(1.0 * (bin_id - lt_bin_id_) / (r_bin_id_ - lt_bin_id_), 0.0, 1.0)# relu
                weight2 = np.clip(math.log(1.0 + math.exp(10.0 * (bin_id - lt_bin_id_) / (r_bin_id_ - lt_bin_id_))) / math.log(1.0 + math.exp(10.0)), 0.0, 1.0)# softplus
            # w3
            if rt_bin_id_ is None:
                weight3 = 0.0
            else:
                x = bin_id
                u = rt_bin_id_
                #a = ((r_bin_id_ - u) / r) if x > u else ((u - l_bin_id_) / r)# asymmetric
                a = (r_bin_id_ - u) / r# symmetric
                weight3 = math.exp(-1.0 * (x - u) * (x - u) / (2 * a * a))
            # w: merge w1, w2 and w3
            weight = alphas[-1]/100.0 * weight1 + betas[-1]/100.0 * weight2 + gammas[-1]/100.0 * weight3

            weights1.append(weight1)
            weights2.append(weight2)
            weights3.append(weight3)
            weights.append(weight)
        
        asize= 26
        ticksize = 24
        lfont1 = {
            'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 18,
        }
        lfont2 = {
            'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 24,
        }
        titlesize = 28
        glinewidth = 2
        
        fig = plt.figure(0)
        fig.set_size_inches(10, 20)
        ax1 = plt.subplot2grid((4,2), (0,0), rowspan=2, colspan=2)
        ax2 = plt.subplot2grid((4,2), (2,0), rowspan=1, colspan=2)
        ax3 = plt.subplot2grid((4,2), (3,0), rowspan=1, colspan=2)

        ax1.set_title(r"$The\ cos\theta\ distribution\ of\ WebFace-All(64K\ samples),\ $" + "\n" + r"$current\ iteration\ is\ %d$" % _iter, fontsize=titlesize)
        ax1.set_xlabel(r"$cos\theta$", fontsize=asize)
        ax1.set_ylabel(r"$Numbers$", fontsize=asize)
        ax1.tick_params(labelsize=ticksize)
        ax1.set_xlim(-0.5, 1.0)
        ax1.set_ylim(0.0, 6000.0)
        ax1.grid(True, linewidth=glinewidth)
        
        ax1.plot(X, filter_pdf, "y-", linewidth=2, label="Hist-all")
        ax1.plot(X, filter_clean_pdf, "g-", linewidth=2, label="Hist-clean")
        ax1.plot(X, filter_noise_pdf, "r-", linewidth=2, label="Hist-noisy")
        ax1.fill(X, filter_pdf, "y", alpha=0.25)
        ax1.fill(X, filter_clean_pdf, "g", alpha=0.50)
        ax1.fill(X, filter_noise_pdf, "r", alpha=0.50)
        # points
        if mode != 1:
            #p_X = [X[l_bin_id_], X[lt_bin_id_], X[rt_bin_id_], X[r_bin_id_]]
            #p_Y = [filter_pdf[l_bin_id_], filter_pdf[lt_bin_id_], filter_pdf[rt_bin_id_], filter_pdf[r_bin_id_]]
            p_X = [X[l_bin_id_], X[r_bin_id_]]
            p_Y = [filter_pdf[l_bin_id_], filter_pdf[r_bin_id_]]
            if lt_bin_id_ is not None:
                p_X.append(X[lt_bin_id_])
                p_Y.append(filter_pdf[lt_bin_id_])
            if rt_bin_id_ is not None:
                p_X.append(X[rt_bin_id_])
                p_Y.append(filter_pdf[rt_bin_id_])
            ax1.scatter(p_X, p_Y, color="y", linewidths=5)
        ax1.legend(["Hist-all", "Hist-clean", "Hist-noisy"], loc=1, ncol=1, prop=lfont1)

        _ax1 = ax1.twinx()
        _ax1.set_ylabel(r"$\omega$", fontsize=asize)
        _ax1.tick_params(labelsize=ticksize)
        _ax1.set_xlim(-0.5, 1.0)
        _ax1.set_ylim(0.0, 1.2)
        
        #_ax1.bar([1.0], [beta*(alpha<0.5)], color="r", width=0.1, align="center")
        #_ax1.bar([1.0], [(1-beta)], bottom=[beta*(alpha<0.5)], color="g", width=0.1, align="center")
        #_ax1.bar([1.0], [beta*(alpha>0.5)], bottom=[beta*(alpha<0.5) + (1-beta)], color="b", width=0.1, align="center")
        
        if mode == 1:
            _ax1.plot(X,  weights,  "k-", linewidth=2, label=r"$\omega$")
            _ax1.legend([r"$\omega$"], loc=2, ncol=2, prop=lfont2)
        else:
            _ax1.plot(X, weights1,  "r:", linewidth=5, label=r"$\omega1$")
            _ax1.plot(X, weights2, "g--", linewidth=5, label=r"$\omega2$")
            _ax1.plot(X, weights3, "b-.", linewidth=5, label=r"$\omega3$")
            _ax1.plot(X,  weights,  "k-", linewidth=2, label=r"$\omega$")
            _ax1.legend([r"$\omega1$", r"$\omega2$", r"$\omega3$", r"$\omega$"], loc=2, ncol=2, prop=lfont2)

        ax2.set_title(r"$The\ accuracy\ on\ LFW,\ $" + "\n" + r"$current\ value\ is\ %.2f$%%" % accs[-1], fontsize=titlesize)
        ax2.yaxis.grid(True, linewidth=glinewidth)
        ax2.set_xlabel(r"$Iter$", fontsize=asize)
        ax2.set_ylabel(r"$Acc(\%)$", fontsize=asize)
        ax2.tick_params(labelsize=ticksize)
        ax2.set_xlim(0, 200000)
        ax2.set_ylim(50.0, 100.0)
        ax2.plot(iters, accs, "k-", linewidth=3, label="LFW")
        ax2.legend(["LFW"], loc=7, ncol=1, prop=lfont1)# center right

        ax3.set_title(r"$The\ Proportion\ of\ \omega1,\ \omega2\ and\ \omega3,\ $" + "\n" + r"$current\ value\ is\ $%.0f%%, %.0f%% and %.0f%%" % (alphas[-1], betas[-1], gammas[-1]), fontsize=titlesize)
        ax3.yaxis.grid(True, linewidth=glinewidth)
        ax3.set_xlabel(r"$Iter$", fontsize=asize)
        ax3.set_ylabel(r"$Proportion(\%)$", fontsize=asize)
        ax3.tick_params(labelsize=ticksize)
        ax3.set_xlim(0, 200000)
        ax3.set_ylim(0.0, 100.0)
        ax3.plot(iters, np.array(alphas), "r-", linewidth=1, label=r"$\alpha$")
        ax3.plot(iters, np.array(alphas) + np.array(betas), "g-", linewidth=1, label=r"$\beta$")
        ax3.plot(iters, np.array(alphas) + np.array(betas) + np.array(gammas), "b-", linewidth=1, label=r"$\gamma$")
        ax3.fill_between(iters, 0, np.array(alphas) + np.array(betas) + np.array(gammas), color="b", alpha=1.0)
        ax3.fill_between(iters, 0, np.array(alphas) + np.array(betas), color="g", alpha=1.0)
        ax3.fill_between(iters, 0, np.array(alphas), color="r", alpha=1.0)
        ax3.legend([r"$\alpha$", r"$\beta$", r"$\gamma$"], loc=7, ncol=1, prop=lfont2)# center right
        
        fig.tight_layout()
        
        plt.savefig("./figures/%s_2D_dist_%d.jpg" % (prefix, _iter))
        plt.close()

        frames += 1
        print "frames:", frames
        print "processed:", _iter
        sys.stdout.flush()


def draw_video(prefix1, prefix2, ratio):
    draw_imgs(prefix1, ratio, mode=1)
    draw_imgs(prefix2, ratio, mode=2)

    fps = 25
    size = (2000, 2000) 
    videowriter = cv2.VideoWriter("./figures/demo_2D_distribution_noise-%d%%.avi" % ratio, cv2.cv.CV_FOURCC(*"MJPG"), fps, size)
    for _iter in xrange(1000, 200000+1, 100):
        if not choose_iter(_iter, 100):
            continue
        image_file1 = "./figures/%s_2D_dist_%d.jpg" % (prefix1, _iter)
        img1 = cv2.imread(image_file1)
        h1, w1, c1 = img1.shape

        image_file2 = "./figures/%s_2D_dist_%d.jpg" % (prefix2, _iter)
        img2 = cv2.imread(image_file2)
        h2, w2, c2 = img2.shape

        assert h1 == h2 and w1 == w2
        img = np.zeros((h1, w1+w2, c1), dtype=img1.dtype)

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

