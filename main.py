'''
CluNILM: Appliance Taxonomy Derivation Using Clustering for NILM Purpose
by Heeseung Yun & Turhaner Saner

Original Paper: Taxonomy Derivation of Household and Industrial Appliances
Based on Optimal Set of Features for NILM

This project is part of seminar DAISE in Technical University of Munich.
(Data Analytics and Intelligent Systems in Energy Informatics)

'''

import os
import re
import argparse
import itertools
import progressbar

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.io as sio
from scipy import signal
from sklearn import linear_model, metrics
from sklearn.cluster import AgglomerativeClustering as Agnes
from sklearn.cluster import Birch, DBSCAN, KMeans, AffinityPropagation
from PIL import Image


# Parameters
SAMPLING = 12000    # Data sampling frequency
FREQ = 60           # Voltage frequency
SQRT_TWO = 1.414
EULER = 2.718
APPLIANCES = 26     # No. of appliances
WINDOW_SIZE = 9     # Window for filters
LARGE_VALUE = 1000000.
# Regular expression for formatting file name
REGEX_STRING = "[0-9]{4}_|_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]*\.mat"

# Environment Variable
NORMALIZATION = True

# Appliance Mapping
APPLIANCE_MAP = {
    "Computer-1": 0,
    "Iron": 1,
    "Laptop-1": 2,
    "Basement-Receiver-DVR-Blueray-Player": 3,
    "Living-Room-A-V-System": 4,
    "Monitor-2": 5,
    "TV": 6,
    "LCD-Monitor-1": 7,
    "Printer": 8,
    "Living-Room-Empty-Socket": 9,
    "Garage-Door": 10,
    "Basement-lights": 11,
    "Closet-light": 12,
    "Dining-Room-overhead-light": 13,
    "Hallway-stairs-light": 14,
    "Kitchen-hallway-light": 15,
    "Kitchen-overhead-light": 16,
    "Office-lights": 17,
    "Upstairs-hallway-light": 18,
    "Living-Room-Desk-Lamp": 19,
    "Living-Room-Tall-Desk-Lamp": 20,
    "Circuit-4": 21,
    "Circuit-9": 22,
    "Circuit-10": 23,
    "Circuit-11": 24,
    "unknown": 25,
}

# Progress bar for data preprocessing
WIDGETS = [
    ' [', progressbar.FormatLabel('Fetch: %(value)4d / %(max_value)4d'), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
]


def preprocess(data_dir, verbose):
    '''
    preprocesses the data in a given directory. Extracts required features and
    return the result in appliance_set and feature_set
    If verbose=True, every possible features are visualized while processing.
    '''
    # Initialize progressbar
    max_idx = sum([len(files) for r, d, files in os.walk(data_dir)])
    progress_bar = progressbar.ProgressBar(WIDGETS=WIDGETS, max_value=max_idx)
    bar_idx = 0
    progress_bar.start()

    # Appliance dictionary with index as a key
    appliance_set = []

    # List of extracted features with index as a key
    feature_set = np.zeros((max_idx, 20))

    for folder, _, files in os.walk(data_dir):
        for filename in files:
            if '.mat' not in filename:
                print('Warning: files non .mat file {} will be ignored.'.format(filename))
                continue

            # Fetch current and voltage
            sample = sio.loadmat(os.path.join(folder, filename))
            current = sample['ROICurrent']
            voltage = sample['ROIVoltage']
            feature = np.zeros(20)

            # Building blocks for features
            max_current = max(current)
            max_voltage = max(voltage)
            rms_voltage = max_voltage / SQRT_TWO
            sample_length = len(current)
            fund_freq = int(FREQ * sample_length / SAMPLING)

            # Feature 0: Current root mean square
            cur_sum = 0
            for cur_val in current:
                cur_sum = cur_sum + cur_val**2

            feature[0] = np.sqrt(cur_sum / sample_length)

            if verbose is True:
                fig = plt.figure()
                fig.suptitle(filename)
                plt1 = fig.add_subplot(2, 1, 1)
                plt1.set_xscale('log')
                plt1.set_title('Voltage')
                plt1.plot(voltage, markersize=0.1)
                plt2 = fig.add_subplot(2, 1, 2)
                plt2.set_xscale('log')
                plt2.set_title('Current')
                plt2.plot(current, markersize=0.1)
                plt2.plot(np.full(sample_length, feature[0]), color='r')
                plt.show()
                plt.close()

            # Feature 1: Normalized average power
            feature[1] = 120 * 120 * feature[0] / rms_voltage

            # Feature 2-6: Current harmonic coefficient
            # Fundamental frequency, with 3rd, 5th, 7th, 9th harmonic coefficients
            current_fft = abs(np.fft.rfft(current))

            feature[2] = current_fft[1 * fund_freq]
            feature[3] = current_fft[3 * fund_freq]
            feature[4] = current_fft[5 * fund_freq]
            feature[5] = current_fft[7 * fund_freq]
            feature[6] = current_fft[9 * fund_freq]

            if verbose is True:
                x_points = np.array([1, 3, 5, 7, 9]) * fund_freq
                plt.figure()
                plt.xscale('log')
                plt.title('Current harmonic coefficient')
                plt.plot(current_fft[:10*fund_freq], markersize=0.1, zorder=1)
                plt.scatter(x_points, feature[2:7], marker='X', color='r', zorder=2)
                plt.show()
                plt.close()

            # Feature 7-10: Nonactive current harmonic coefficient
            # 7th, 11th, 13th, 17th harmonic coefficients
            avg_power = feature[0] * rms_voltage
            active_current = (avg_power / (rms_voltage * rms_voltage)) * voltage
            non_current = current - active_current
            non_current_fft = abs(np.fft.rfft(non_current))

            if verbose is True:
                fig = plt.figure()
                fig.suptitle(filename)
                plt1 = fig.add_subplot(2, 1, 1)
                plt1.set_xscale('log')
                plt1.set_title('Current')
                plt1.plot(current, markersize=0.1)
                plt2 = fig.add_subplot(2, 1, 2)
                plt2.set_xscale('log')
                plt2.set_title('Active and Non-active Current')
                plt2.plot(active_current, markersize=0.1, color='r')
                plt2.plot(non_current, markersize=0.1, color='b')
                plt.show()
                plt.close()

            feature[7] = non_current_fft[7 * fund_freq]
            feature[8] = non_current_fft[11 * fund_freq]
            feature[9] = non_current_fft[13 * fund_freq]
            feature[10] = non_current_fft[17 * fund_freq]

            if verbose is True:
                x_point = np.array([7, 11, 13, 17]) * fund_freq
                plt.figure()
                plt.xscale('log')
                plt.title('Nonactive Current harmonic coefficient')
                plt.plot(non_current_fft, markersize=0.1, zorder=1)
                plt.scatter(x_point, feature[7:11], marker='X', color='r', zorder=2)
                plt.show()
                plt.close()

            # Feature 11-12: Total harmonic distortion
            cur_fund = 0.
            non_fund = 0.
            for i in np.arange(2 * fund_freq, sample_length, fund_freq):
                cur_fund = cur_fund + current_fft[i]**2
                non_fund = non_fund + non_current_fft[i]**2

            feature[11] = np.sqrt(cur_fund) / current_fft[fund_freq]
            feature[12] = np.sqrt(non_fund) / non_current_fft[fund_freq]

            # Feature 13-14: Transient current
            # feature[13] = max_current * (1 - 1 / EULER)
            # feature[14] = max_current * (1 - 1)
            # Using peak currents as alternative
            feature[13] = max_current
            feature[14] = max(current_fft)

            # Feature 15: Area
            # Performance benchmark: 50ms per cycle using CPU
            # Single cycle is calculated, instead of all
            period = int(SAMPLING / FREQ)
            # Select maxima and minima w.r.t voltage
            max_pt = np.argmax(voltage[0:period])
            min_pt = np.argmin(voltage[0:period])

            # Calculate scale of area
            diag = np.sqrt((current[max_pt] - current[min_pt]) ** 2 + \
                    (voltage[max_pt] - voltage[min_pt])**2)
            # Calculate rotation using cross product
            sgn = 1
            # For loop is used to guarantee non-zero cross product by iteration
            # sgn = 1 if clockwise, otherwise -1
            for i in range(100):
                pt_1 = min(max_pt, min_pt) + i
                pt_2 = pt_1 + 50
                rot_dir = voltage[pt_1] * (current[pt_2] - current[pt_1]) - \
                    current[pt_1] * (voltage[pt_2] - voltage[pt_1])
                if rot_dir < 0:
                    break
                elif rot_dir > 0:
                    sgn = -1
                    break

            # Plot area in 5 by 5 (inch) coordinate
            plt.figure(figsize=(5, 5))
            plt.fill(voltage[0:period], current[0:period])
            plt.axis('off')
            plt.savefig("temp_trajectory.png", bbox_inches='tight', pad_inches=0)
            plt.close()

            # Count the number of pixels colored and multiply by scale (diag)
            img = np.asarray(Image.open("temp_trajectory.png").convert('L'))
            img = 1 * (img < 127)
            feature[15] = sgn * diag * img.sum() / (img.shape[0] * img.shape[1])

            # Feature 16: Curvature
            start_pt = min(max_pt, min_pt)
            ml_current = [current[start_pt]]
            ml_voltage = [current[start_pt]]

            # Get average of two facing points
            for i in np.arange(1, int(period/2), 1):
                ml_current.append((current[start_pt+i] + current[start_pt-i]) / 2)
                ml_voltage.append((voltage[start_pt+i] + voltage[start_pt-i]) / 2)

            ml_current.append(current[start_pt + int(period/2)])
            ml_voltage.append(voltage[start_pt + int(period/2)])

            lin_reg = linear_model.LinearRegression().fit(ml_current, ml_voltage)

            if verbose is True:
                plt.figure()
                plt.title('Area, Regression and Curvature')
                plt.plot(current[:200], voltage[:200])
                plt.scatter(ml_current, ml_voltage, color='black')
                plt.plot(ml_current, lin_reg.predict(ml_current))
                plt.show()
                plt.close()

            feature[16] = lin_reg.score(ml_current, ml_voltage)

            # Feature 17-19: Wavelet
            hp_filter = np.blackman(WINDOW_SIZE)
            lp_range = np.arange(-4, 5, 1)
            # Low-pass filter from https://plot.ly/python/fft-filters
            lp_filter = 0.42 - 0.5 * np.cos(2 * np.pi * lp_range/8) + \
                    0.08 * np.cos(4 * np.pi * lp_range/8)

            if verbose is True:
                fig = plt.figure()
                fig.suptitle(filename)

            f_i = current[:, 0]
            for i in range(8):
                wd_i = signal.convolve(f_i, hp_filter)[::2]
                f_i = signal.convolve(f_i, lp_filter)[::2]
                if i == 2:
                    if verbose is True:
                        plt1 = fig.add_subplot(3, 1, 1)
                        plt1.set_xscale('log')
                        plt1.set_title('2nd Detail Wavelet')
                        plt1.plot(wd_i, markersize=0.1)

                    feature[17] = sum(np.square(wd_i))
                elif i == 4:
                    if verbose is True:
                        plt2 = fig.add_subplot(3, 1, 2)
                        plt2.set_xscale('log')
                        plt2.set_title('4th Detail Wavelet')
                        plt2.plot(wd_i, markersize=0.1)

                    feature[18] = sum(np.square(wd_i))

            feature[19] = sum(np.square(wd_i))

            if verbose is True:
                plt3 = fig.add_subplot(3, 1, 3)
                plt3.set_xscale('log')
                plt3.set_title('8th Detail Wavelet')
                plt3.plot(wd_i, markersize=0.1)
                plt.show()
                plt.close()

            # Update
            feature_set[bar_idx, :] = feature
            appliance_name = re.sub(REGEX_STRING, '', filename)
            appliance_set.append(appliance_name)

            bar_idx = bar_idx + 1
            progress_bar.update(bar_idx)

    progress_bar.finish()

    if NORMALIZATION:
        for i in range(feature_set.shape[1]):
            feature_set[:, i] = (feature_set[:, i] - np.average(feature_set[:, i])) / \
                    np.std(feature_set[:, i])
    # For pure steady-state analysis
    # feature_set = np.delete(feature_set, [13, 14, 17, 18, 19], 1)

    return appliance_set, feature_set


def clustering(feature_set, appliance_set, method, verbose):
    '''
    For given feature_set and appliance_set, clustering is performed
    based on the indicated method.
    If verbose=True, analysis data of each cluster is displayed
    '''
    # Algorithm that allows iteration
    if method == 'AGNES' or method == 'BIRCH' or method == 'KMeans':
        for clu_size in np.arange(2, 10, 1):
            if method == 'AGNES':
                # linkage: distance metric (ward)
                clu = Agnes(n_clusters=clu_size, linkage='ward').fit(feature_set)
            elif method == 'BIRCH':
                # threshold: splitting tendency (lower ~ more split)
                # branching_factor: maximum CF subcluster
                clu = Birch(n_clusters=int(clu_size), threshold=0.1, \
                        branching_factor=10).fit(feature_set)
            else:
                # init: seed initialization
                # n_init: repetition of seeding
                # max_iter: maximum iteration
                # tol: tolerance of convergence
                # algorithm: "full" or "elkan" for k-means algorithm
                clu = KMeans(n_clusters=clu_size, init='k-means++', \
                        max_iter=600).fit(feature_set)

            if verbose is True:
                print("Cluster Analysis of {}_{}".format(method, clu_size))
                # Calculate coefficient
                silscore = metrics.silhouette_score(feature_set, clu.labels_)
                print("Silhouette score: {}".format(silscore))
                # Arrays for cluster-wise summary
                summary_siz = np.zeros(clu_size)
                summary_max = np.full((feature_set.shape[1], clu_size), -LARGE_VALUE)
                summary_min = np.full((feature_set.shape[1], clu_size), LARGE_VALUE)
                summary_sum = np.zeros((feature_set.shape[1], clu_size))

            # Confusion matrix construction
            conf_matrix = np.zeros((clu_size, APPLIANCES))

            if verbose is True:
                for i in range(feature_set.shape[0]):
                    summary_siz[clu.labels_[i]] = summary_siz[clu.labels_[i]] + 1
                    for j in range(feature_set.shape[1]):
                        summary_max[j, clu.labels_[i]] = \
                                max(summary_max[j, clu.labels_[i]], feature_set[i, j])
                        summary_min[j, clu.labels_[i]] = \
                                min(summary_min[j, clu.labels_[i]], feature_set[i, j])
                        summary_sum[j, clu.labels_[i]] = \
                                summary_sum[j, clu.labels_[i]] + feature_set[i, j]
                for i in range(clu_size):
                    np.set_printoptions(precision=2, linewidth=200, suppress=True)
                    print("Cluster {} ({})".format(i, summary_siz[i]))
                    print("  * Max: {}".format(summary_max[:, i]))
                    print("  * Min: {}".format(summary_min[:, i]))
                    print("  * Avg: {}".format(summary_sum[:, i]/summary_siz[i]))

            for i in range(feature_set.shape[0]):
                conf_matrix[clu.labels_[i], APPLIANCE_MAP[appliance_set[i]]] = \
                    conf_matrix[clu.labels_[i], APPLIANCE_MAP[appliance_set[i]]] + 1

            plot_confusion_matrix(cm=conf_matrix, classes=list(APPLIANCE_MAP.keys()), \
                    method=method)

        if method == 'AGNES':
            dendrogram_visualize(clu, appliance_set, method)

    elif method == 'DBSCAN':
        # Density-Based Spatial Clustering of Applications with Noise
        # eps: max_distance
        # min_samples: minimum number of samples for core
        # metric: 'euclidean'
        # algorithm: auto, ball_tree, kd_tree, brute
        # leaf_size: 30
        # p: power of minkowski metric
        clu = DBSCAN().fit(feature_set)
        clu_size = len(set(clu.labels_))

        conf_matrix = np.zeros((clu_size, APPLIANCES))
        # Last index indicates noise set (ragbag)
        for i in range(feature_set.shape[0]):
            conf_matrix[clu.labels_[i], APPLIANCE_MAP[appliance_set[i]]] = \
                conf_matrix[clu.labels_[i], APPLIANCE_MAP[appliance_set[i]]] + 1

        plot_confusion_matrix(cm=conf_matrix, classes=list(APPLIANCE_MAP.keys()), \
                method=method)

    elif method == 'AffinityPropagation':
        # damping (0.5 ~ 1)
        # max_iter (200)
        # convergence_iter (15)
        clu = AffinityPropagation(max_iter=50).fit(feature_set)
        clu_size = len(set(clu.labels_))

        conf_matrix = np.zeros((clu_size, APPLIANCES))
        # Last index indicates noise set (orphan)
        for i in range(feature_set.shape[0]):
            conf_matrix[clu.labels_[i], APPLIANCE_MAP[appliance_set[i]]] = \
                conf_matrix[clu.labels_[i], APPLIANCE_MAP[appliance_set[i]]] + 1

        plot_confusion_matrix(cm=conf_matrix, classes=list(APPLIANCE_MAP.keys()), \
                method=method)

    else:
        print("Error: Method {} is not supported.".format(method))


def dendrogram_visualize(cluster_result, appliance_set, method):
    '''
    Visualize dendrogram for AGNES clustering.
    This code is modified from github.com/scikit-learn/scikit-learn
    (examples/cluster/plot_hierarchical_clustering_dendrogram.py)
    '''
    plt.figure(figsize=(30, 30))
    plt.title('{} Dendrogram'.format(method))

    children = cluster_result.children_
    distance = np.arange(children.shape[0])
    observations = np.arange(2, children.shape[0] + 2)
    linkage_matrix = np.column_stack([children, distance, observations]).astype(float)

    sp.cluster.hierarchy.dendrogram(linkage_matrix, labels=appliance_set)

    plt.savefig('./visualization/{}_Dendrogram.png'.format(method))
    plt.close()


def plot_confusion_matrix(cm, classes, method, normalize=False, cmap=plt.cm.Blues):
    '''
    Visualize confusion matrix
    This code is modified from:
    scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        cm_norm = cm
    else:
        cm = cm.astype('int')
        cm_norm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]

    # Set up the size of confusion matrix
    plt.figure(figsize=(14, 5) if method != 'AffinityPropagation' else (20, 30))

    for i, _ in enumerate(classes):
        classes[i] = (classes[i][:10] + '..') if len(classes[i]) > 10 else classes[i]

    plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    # plt.title('{} Confusion Matrix'.format(method))
    plt.colorbar()
    plt.xticks(np.arange(cm_norm.shape[1]), classes, rotation=90)
    plt.yticks(np.arange(cm_norm.shape[0]))

    fmt = '.2f' if normalize else 'd'
    thresh = cm_norm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", \
                color="white" if cm_norm[i, j] > thresh else "black")

    plt.ylabel('Clusters')
    plt.xlabel('Appliances')
    plt.tight_layout()

    plt.savefig('./visualization/{}_{}_Confusion.png'.format(method, cm.shape[0]), \
            bbox_inches='tight', pad_inches=0.5)
    plt.close()


def main():
    '''
    Main routine of this program
    Executes preprocessing and clustering based on parsed arguments
    '''
    # Define argument parser for execution
    parser = argparse.ArgumentParser(description=\
            'CluNILM: Appliance Taxonomy Derivation Using Clustering for NILM Purpose')
    parser.add_argument("-f", "--feature", type=str, \
            help="Directory of extracted feature")
    parser.add_argument("-d", "--dataset", type=str, \
            help="Directory of dataset", default="./dataset")
    parser.add_argument("-c", "--clustering", type=str, \
            help="Clustering method to be applied (AGNES, BIRCH, KMeans, DBSCAN, AffinityPropagation)")
    parser.add_argument("-v", "--verbose", type=bool, \
            help="Show detailed log and visuals while processing (default: False)", \
            default=False)
    args = parser.parse_args()

    if args.clustering is None:
        parser.print_help()
    else:
        if args.feature is None:
            # No existing feature available, thereby start preprocessing
            if sum([len(files) for r, d, files in os.walk(args.dataset)]) <= 0:
                # Check validity of dataset directory
                print("Error: Make sure you've entered right directory for dataset")
            else:
                appliance_set, feature_set = preprocess(args.dataset, args.verbose)
                np.save('./save/savepoint.npy', feature_set)
                np.save('./save/label.npy', appliance_set)
                clustering(feature_set, appliance_set, args.clustering, args.verbose)
        else:
            # Use existing feature set
            try:
                feature_set = np.load(os.path.join(args.feature, 'savepoint.npy'))
                appliance_set = np.load(os.path.join(args.feature, 'label.npy'))
            except:
                print("Error: Cannot open savepoint.npy and label.npy")
            clustering(feature_set, appliance_set, args.clustering, args.verbose)


if __name__ == '__main__':
    main()
