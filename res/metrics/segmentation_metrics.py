import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

major = cv2.__version__.split('.')[0]     # Get opencv version
bDebug = False

def get_ious(pred,gt,debug=False):
    '''
    This function calculates IOU for predictions. Currently it only has multilabel support. Check for binary support (our labels are loaded differently for both cases)
    Checks if there's more than 2 classes, if not, returns 0 as that is not yet implemented.

    Please note that this function expects prediction of the size [NUM_CLASSES x W x h] andHgt of the size [W x H]

    For each label prediction, cuts of at 80%, i.e. if the prediction score is > 0.8 takes the pixel as being predicted yes for the class.

    Example Inputs:
    gt: torch tensor: ([0,1],
    [2,3],
    [0,1],
    [2,3]])
    pred: torch tensor: torch.stack([gt==0,gt==1,gt==2,gt==3])

    Thus prediction has as many channels as classes, while gt has only 1 channel, with pixel value = class number.

    Known Failure Mode: If class numbers don't start with 0 it will not work.
    '''
    if pred.size()[0] < 2:
        #implement binary case here
        return 0

    num_classes = pred.size()[0]

    ious = np.zeros(num_classes)

    if debug:
        print('get_ious() run in debug mode, below are debug print statements:')
        print('---------------------------------------------------------------\n')
        print("num classes is", num_classes)

    for i in range(num_classes):
        pred_category = pred[i] > 0.8 # this is W x H
        gt_category = gt == i # Again W x H

        if debug:
            print("category %s prediction is \n%s"%(i,pred_category))
            print("category %s ground truth is \n%s"%(i,gt_category))

        intersection = torch.sum(pred_category & gt_category).item()
        union = torch.sum((pred_category + gt_category) > 0).item()

        if union != 0:
            iou = intersection/union
        else:
            iou = None

        ious[i] = iou

    return ious


# def calc_precision_recall(contours_a, contours_b, threshold):
#
#     top_count = 0
#
#     try:
#         for b in range(len(contours_b)):
#
#             # find the nearest distance
#             for a in range(len(contours_a)):
#                 dist = (contours_a[a][0] - contours_b[b][0]) *                     (contours_a[a][0] - contours_b[b][0])
#                 dist = dist +                     (contours_a[a][1] - contours_b[b][1]) *                     (contours_a[a][1] - contours_b[b][1])
#                 if dist < threshold*threshold:
#                     top_count = top_count + 1
#                     break
#
#         precision_recall = top_count/len(contours_b)
#     except Exception as e:
#         precision_recall = 0
#
#     return precision_recall, top_count, len(contours_b)
#
#
# # In[145]:
#
#
# def displacement_metric(contours_a, contours_b, threshold):
#
#     top_count = 0
#     sum_closest_dist = 0
#     for b in range(len(contours_b)):
#         closest_distance = inf
#         # find the nearest distance
#         for a in range(len(contours_a)):
#             dist = (contours_a[a][0] - contours_b[b][0]) *                 (contours_a[a][0] - contours_b[b][0])
#             dist = dist +                 (contours_a[a][1] - contours_b[b][1]) *                 (contours_a[a][1] - contours_b[b][1])
#             if dist < closest_distance:
#                 closest_dist = dist
#         sum_closest_dist += closest_dist
#     return sum_closest_dist
#
#
# # In[146]:
#
#
# def format_label(imarray):
# #     imarray = imarray[0,:,:]
#     imarray[imarray>150] = 255
#     imarray[imarray<150] = 0
#     imarray[imarray==255] = 1
#     return imarray
#
#
# # In[137]:
#
#
# def get_contours(gtfile,prfile):
#     gt__ = cv2.imread(gtfile)    # Read GT segmentation
#     gt_ = cv2.cvtColor(gt__, cv2.COLOR_BGR2GRAY)    # Convert color space
#
#
#     gt_ = format_label(gt_)
#
#     pr_ = cv2.imread(prfile)    # Read predicted segmentation
#     pr_ = cv2.cvtColor(pr_, cv2.COLOR_BGR2GRAY)    # Convert color space
#
#     pr_ = format_label(pr_)
#     gt_ = gtfile
#     pr_ = prfile
#
#     classes_gt = np.unique(gt_)    # Get GT classes
#     classes_pr = np.unique(pr_)    # Get predicted classes
#     # Check classes from GT and prediction
#     if not np.array_equiv(classes_gt, classes_pr):
#         print('Classes are not same! GT:', classes_gt, 'Pred:', classes_pr)
#
#         classes = np.concatenate((classes_gt, classes_pr))
#         classes = np.unique(classes)
#         classes = np.sort(classes)
#         print('Merged classes :', classes)
#     else:
# #         print('Classes :', classes_gt)
#         classes = classes_gt    # Get matched classes
#
#     m = np.max(classes)    # Get max of classes (number of classes)
#     # Define bfscore variable (initialized with zeros)
#     bfscores = np.zeros((m+1), dtype=float)
#
#     for i in range(m+1):
#         bfscores[i] = np.nan
#
#     target_class = 1
#
#     gt = gt_.copy()
#     gt[gt != target_class] = 0
#     _, contours, _ = cv2.findContours(
#     gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#     contours_gt = []
#     for i in range(len(contours)):
#         for j in range(len(contours[i])):
#             contours_gt.append(contours[i][j][0].tolist())
#
#     # Draw GT contours
#     img = np.zeros_like(gt__)
#     # print(img.shape)
#     img[gt == target_class, 0] = 128  # Blue
#     img = cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
#
#     pr = pr_.copy()
#     pr[pr != target_class] = 0
#     # print(pr.shape)
#
#     # Draw predicted contours
#     img[pr == target_class, 2] = 128  # Red
#     img = cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
#     _, contours, _ = cv2.findContours(
#             pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#     contours_pr = []
#     for i in range(len(contours)):
#         for j in range(len(contours[i])):
#             contours_pr.append(contours[i][j][0].tolist())
#     return contours_gt,contours_pr
#
#
# # In[141]:
#
#
# def bfscore(contours_gt,contours_pr,threshold = 2):
#     # 3. calculate
#     precision, numerator, denominator = calc_precision_recall(
#         contours_gt, contours_pr, threshold)    # Precision
# #     print("\tprecision:", denominator, numerator)
#
#     recall, numerator, denominator = calc_precision_recall(
#         contours_pr, contours_gt, threshold)    # Recall
# #     print("\trecall:", denominator, numerator)
#     f1 = 0
#     try:
#         f1 = 2*recall*precision/(recall + precision)    # F1 score
#     except:
#         #f1 = 0
#         f1 = np.nan
# #     print("\tf1:", f1)
#     return f1
#
#
# # In[ ]:
#
#
# def displacement_score(contours_gt,contours_pr,threshold = 2):
#     # 3. calculate
#     precision, numerator, denominator = calc_precision_recall(
#         contours_gt, contours_pr, threshold)    # Precision
# #     print("\tprecision:", denominator, numerator)
#
#     recall, numerator, denominator = calc_precision_recall(
#         contours_pr, contours_gt, threshold)    # Recall
# #     print("\trecall:", denominator, numerator)
#     f1 = 0
#     try:
#         f1 = 2*recall*precision/(recall + precision)    # F1 score
#     except:
#         #f1 = 0
#         f1 = np.nan
# #     print("\tf1:", f1)
#     return f1
