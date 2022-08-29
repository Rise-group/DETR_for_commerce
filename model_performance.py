# %%
# import the necessary packages
import os
import six.moves.urllib as urllib
import sys
import tarfile
# import tensorflow as tf
# import zipfile
# import cv2 #descomentar
import numpy as np
import pandas as pd
#from .prediction import drop_same_building
# from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt

# %%


def drop_same_building(bbox, scores, classes, trshld_px=3):
    #Drop a instance if has another with higher score
    #this funcion needs the bounding_boxes be no relative, with the nominal value
    #i.e., whit weight=332, height=133
    list_same=[]
    centers = get_center(bbox)
    Xc = centers[:,0]
    for i in range(len(Xc)):
       Xrest = Xc[i]-Xc
       if len(Xrest[np.abs(Xrest)<trshld_px])> 1:
           list_same.append(i)
    if len(list_same)>0:
        ind_max = np.where(scores==np.max(scores))[0]
        list_to_drop=list(set(list_same)-set(ind_max))
        print("Hay varias detecciones de la misma fachada")
        print("centroides: ",centers, "y scores: ", scores)
        for d in list_to_drop:
            print("Se mantendra el bounding_box {} con centroide {} y score {}".format(ind_max[0],
                  tuple(centers[ind_max[0]]), scores[ind_max[0]]))
            print("Borrando el bounding_box {} con centroides {} y con scores {}".format(d,
                  tuple(centers[d]), scores[d]))
            new_bbox = np.delete(bbox, d, axis=0)
            new_scores = np.delete(scores, d, axis=0)
            new_classes = np.delete(classes, d, axis=0)
            return new_bbox,new_scores,new_classes
    else:
        print("No hay varias detecciones de la misma fachada, puede continuar")
        return bbox, scores, classes


def read_yolo_bbox(image, width, height):
    txt_path = 'output/db_bbox/tuning/' + image[:-4] + '.txt'
    classes = []
    with open(txt_path) as f:
        content = f.readlines()
    bbox = np.zeros((len(content), 4), dtype=int)
    for line in content:
        values_str = line.split()
        class_index, x_center, y_center, x_width, y_height = map(float, values_str)
        class_index = int(class_index)
        classes.append(class_index)
        # convert yolo to points
        x1, y1, x2, y2 = yolo_to_x_y(x_center, y_center, x_width, y_height, width, height)
        x1, y1, x2, y2 = x1 - 1, y1 - 1, x2 - 1, y2 - 1
        bbox[content.index(line)] = x1, y1, x2, y2
        # img_objects.append([class_index, x1, y1, x2, y2])
        # print(bbox)
    return bbox, classes


def read_yolo_bbox2(image, path_csv_labels):
    # txt_path = 'output/db_bbox/tuning/' + image[:-4] + '.txt'

    labels = pd.read_csv(path_csv_labels)

    list_index_labels = np.where(labels.filename == image)[0].tolist()

    #print(labels.iloc[list_index_labels])

    content = labels.iloc[list_index_labels]
    bbox = np.zeros((len(content), 4), dtype=int)
    label_map = ["comercial", "no_comercial"]
    classes = []
    for i in list_index_labels:
        classe, x1, x2, y1, y2 = labels.iloc[[i]][["class", "xmin", "xmax", "ymin", "ymax"]].values[0]

        class_index = label_map.index(classe)
        classes.append(class_index)
        bbox[list_index_labels.index(i)] = x1, y1, x2, y2

    return bbox, classes


def draw_text_bb(img, bbox, classes, scores=None, color=(0, 255, 0), size=0.25):
    class_list = ["comercial", "no comercial"]
    # Allow draw bounding boxex, and text inside them
    # the bbox must be (x1, y1, x2, y2) where 1 is topleft and 2 is botright
    # tu run for just a bbox:
    # draw_text_bb(image, np.expand_dims(bbo_5_full[1], axis=0),np.expand_dims(scores_5[1], axis=0), np.expand_dims(classes_5[1], axis=0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    centers = get_center(bbox)
    print(centers)
    for i in range(bbox.shape[0]):
        center = centers[i]
        xc, yc = center[0], center[1]
        center = tuple(center)
        (x1, y1, x2, y2) = bbox[i]

        top = (x1, y1 - 5)
        if classes[i] == 1:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, '%s' % (class_list[classes[i]]), top, font, size, color)
        if not scores is None:
            cv2.putText(img, '{}%'.format((int(100 * scores[i]))), (xc - 5, yc), font, size, color)
    return img


def match_pred_gt(bbox_gt, bbox_p, classes_gt, classes_p):
    list_mgt = []
    list_mp = []
    centers_gt = get_center(bbox_gt)
    centers_p = get_center(bbox_p)
    Xc_gt = centers_gt[:, 0]
    Xc_p = centers_p[:, 0]
    ious = np.zeros((len(classes_gt), 1))
    classes = np.unique(classes_gt)
    for cl in classes:
        ids_gt = np.where(classes_gt == cl)[0]
        for i in ids_gt:
            Xrest = Xc_gt[i] - Xc_p
            if len(Xrest[np.abs(Xrest) < 15]) > 0:
                try:
                    id_mp = np.where(Xrest == Xrest[np.abs(Xrest) < 15])[0][0]
                    list_mp.append(id_mp)
                    list_mgt.append(i)
                except:
                    continue
            else:
                continue
    for bb in range(len(list_mgt)):
        # print(Xc_gt[list_mgt[bb]], Xc_p[list_mp[bb]])
        print(bb_intersection_over_union(bbox_gt[list_mgt[bb]], bbox_p[list_mp[bb]]))
        ious[list_mgt[bb]] = bb_intersection_over_union(bbox_gt[list_mgt[bb]], bbox_p[list_mp[bb]])
    return ious


def iou_all_pred_gt(bbox_gt, bbox_p, classes_gt, classes_p):
    list_mgt = []
    list_mp = []
    centers_gt = get_center(bbox_gt)
    centers_p = get_center(bbox_p)
    Xc_gt = centers_gt[:, 0]
    Xc_p = centers_p[:, 0]

    classes = np.unique(classes_gt)
    # for cl in classes:
    #    ids_gt=np.where(classes_gt==cl)[0]
    for i in range(len(Xc_gt)):
        Xrest = Xc_gt[i] - Xc_p
        if len(Xrest[np.abs(Xrest) < 15]) > 0:
            try:
                id_mp = np.where(Xrest == Xrest[np.abs(Xrest) < 15])[0][0]
                list_mp.append(id_mp)
                list_mgt.append(i)
            except:
                continue
        else:
            continue
    ious = np.zeros((len(list_mgt), 1))
    for bb in range(len(list_mgt)):
        # print(Xc_gt[list_mgt[bb]], Xc_p[list_mp[bb]])
        print(bb_intersection_over_union(bbox_gt[list_mgt[bb]], bbox_p[list_mp[bb]]))
        ious[bb] = bb_intersection_over_union(bbox_gt[list_mgt[bb]], bbox_p[list_mp[bb]])
    return ious


def yolo_to_x_y(x_center, y_center, x_width, y_height, width=332, height=133):
    x_center *= width
    y_center *= height
    x_width *= width
    y_height *= height
    x_width /= 2.0
    y_height /= 2.0
    return int(x_center - x_width), int(y_center - y_height), int(x_center + x_width), int(y_center + y_height)


def get_center(bbox):
    centers = np.zeros((bbox.shape[0], 2), dtype=int)
    for i in range(bbox.shape[0]):
        (x1, y1, x2, y2) = bbox[i]
        center = (int((x2 - x1) / 2) + x1), (int((y2 - y1) / 2) + y1)
        centers[i] = center
    return centers


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def compute_iou(image_name, boxes, scores, classes):
    # Predictions with a scores higher than 50%
    bbo_5 = boxes[scores > 0.5]
    classes_5 = classes[scores > 0.5].astype(np.int32)
    scores_5 = scores[scores > 0.5]

    # Convert Bounding_boxes from relative to nominal
    bbo_5_full = (bbo_5[:, [1, 0, 3, 2]] * (332, 133, 332, 133)).astype(np.int32)

    # Delete the detections of the instances if has another with higher score
    box_c, scores_c, classes_c = drop_same_building(bbo_5_full, scores_5, classes_5, trshld_px=3)

    # read the bounding boxes from the ground truth
    box_gt, classes_gt = read_yolo_bbox(image_name, 332, 133)

    # match the bounding of the ground truth and predictions and compute the
    # Intersection over union
    ious = match_pred_gt(box_gt, box_c, classes_gt, classes_c - 1)

    accuracy = ious.mean()
    print("the accuraccy for the image {} is {:.2f}%".format(image_name, accuracy * 100))
    return accuracy


def compute_iou_relaxed(image_name, boxes, scores, classes):
    # Predictions with a scores higher than 50%
    bbo_5 = boxes[scores > 0.5]
    classes_5 = classes[scores > 0.5].astype(np.int32)
    scores_5 = scores[scores > 0.5]

    # Convert Bounding_boxes from relative to nominal
    bbo_5_full = (bbo_5[:, [1, 0, 3, 2]] * (332, 133, 332, 133)).astype(np.int32)

    # Delete the detections of the instances if has another with higher score
    box_c, scores_c, classes_c = drop_same_building(bbo_5_full, scores_5, classes_5, trshld_px=3)

    # read the bounding boxes from the ground truth
    box_gt, classes_gt = read_yolo_bbox(image_name, 332, 133)

    # match the bounding of the ground truth and predictions and compute the
    # Intersection over union
    ious = iou_all_pred_gt(box_gt, box_c, classes_gt, classes_c - 1)

    accuracy = ious.mean()
    print("the accuraccy for the image {} is {:.2f}%".format(image_name, accuracy * 100))
    return accuracy


def draw_iou(image_name, boxes, scores, classes):
    test_path = "/home/jessiycami/Pictures/resize/tuning"
    IMAGE_SIZE = (12, 8)

    # Predictions with a scores higher than 50%
    bbo_5 = boxes[scores > 0.5]
    classes_5 = classes[scores > 0.5].astype(np.int32)
    scores_5 = scores[scores > 0.5]

    # Convert Bounding_boxes from relative to nominal
    bbo_5_full = (bbo_5[:, [1, 0, 3, 2]] * (332, 133, 332, 133)).astype(np.int32)

    # Delete the detections of the instances if has another with higher score
    box_c, scores_c, classes_c = drop_same_building(bbo_5_full, scores_5, classes_5, trshld_px=3)

    # read the bounding boxes from the ground truth
    box_gt, classes_gt = read_yolo_bbox(image_name, 332, 133)
    image = cv2.imread("output/" + image_name)
    draw_text_bb(image, box_c, classes_c - 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    box_gt, classes_gt = read_yolo_bbox(image_name, 332, 133)

    ious = match_pred_gt(box_gt, box_c, classes_gt, classes_c - 1)

    draw_text_bb(image, box_gt, classes_gt, ious)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_rgb)


def performance_metrics_from_binary_classification(target, prediction):
    """
    This funcion computes a set of performance metrics for binary classification of images.

    Parameters
    ----------
    target :      2D numpy array with ground truth values (i.e: target values).
    prediction :  2D numpy array estimated values (i.e: predicted values)

    Returns
    -------
    TN_map : array
        True negative map.
    FP_map : array
        False positive map.
    FN_map : array
        False negative map.
    TP_map : array
        True positive map.
    TN : int
        Number of true negatives.
    FP : int
        Number of false positives.
    FN : int
        Number of false negatives.
    TP : int
        Number of true positives.
    P : int
        Number of positives.
    N : int
        Number of negatives.
    CM : array
        Confusion matrix.
    observations : int
        Total number of observations.
    FP_rate : float
        False positive rate.
    TP_rate : float
        True positive rate.
    precision : float
        Precision.
    recall : float
        Recall.
    accuracy : float
        Accuracy.
    F_measure : float
        F1 measure.
    specificity : float
        Specificity.
    error_rate : float
        Error rate.

    """

    TN_map = np.logical_and(np.logical_not(target), np.logical_not(prediction))  # target0_prediction0. This is good.
    FP_map = np.logical_and(np.logical_not(target), prediction)  # target0_prediction1. This is bad.
    FN_map = np.logical_and(target, np.logical_not(prediction))  # target1_prediction0. This is bad
    TP_map = np.logical_and(target, prediction)  # target1_prediction1. This is good.

    TN = np.sum(TN_map)  # True negative
    FP = np.sum(FP_map)  # False positive
    FN = np.sum(FN_map)  # False negative
    TP = np.sum(TP_map)  # True positive
    P = TP + FN
    N = FP + TN
    CM = np.array([[TP, FN],
                   [FP, TN]])  # Confusion matrix
    observations = P + N

    FP_rate = FP / N  # False positive rate
    TP_rate = TP / P  # True positive rate
    precision = TP / (TP + FP)  # Precision
    recall = TP / P  # Recall
    accuracy = (TP + TN) / (P + N)  # Accuracy
    F_measure = 2.0 / ((1.0 / precision) + (1.0 / recall))  # F1 measure
    specificity = TN / (FP + TN)  # Specificity
    error_rate = (FP + FN) / (P + N)  # error rate
    #print(df_final[df_final.num_com_labeled>0].difference.value_counts())
    print("CM", CM)
    print("observations", observations)
    print("FP_rate", FP_rate)
    print("TP_rate", TP_rate)
    print("precision", precision)
    print("recall", recall)
    print("accuracy", accuracy)
    print("F_measure", F_measure)
    print("specificity", specificity)
    print("error_rate", error_rate)

    return TN_map, FP_map, FN_map, TP_map, TN, FP, FN, TP, P, N, CM, observations, FP_rate, TP_rate, precision, recall,\
           accuracy, F_measure, specificity, error_rate


def join_real_prediction(path_csv_labels, path_csv_pred):
    """
        This funcion computes a set of performance metrics for binary classification of images.

        Parameters
        ----------
        path_csv_labels :      2D numpy array with ground truth values (i.e: target values).
        path_csv_pred :  2D numpy array estimated values (i.e: predicted values)

        Returns
        -------
        df_final : DataFrame
            DataFrame with columns: "num_labels","num_com_labeled", "num_com_detected", "difference".
    """
    bboxes = pd.read_csv(path_csv_labels)
    csv_inference = pd.read_csv(path_csv_pred)
    bboxes_commercial = bboxes[bboxes['class']=="comercial"]
    df_train = pd.DataFrame({"num_labels": bboxes.filename.value_counts()})
    df_train=df_train.join(bboxes_commercial.filename.value_counts()).fillna(0)
    df_train=df_train.join(csv_inference.image_name.value_counts()).fillna(0)
    df_train.loc[:,"num_com_labeled"]=df_train.filename.astype(np.int)
    df_train.loc[:,"num_com_detected"]=df_train.image_name.astype(np.int)
    df_train.loc[:,"difference"]=df_train.num_com_labeled- df_train.num_com_detected
    df_final =df_train[["num_labels","num_com_labeled", "num_com_detected", "difference"]].sort_index()


    lista_scores_concat = list()
    for image in csv_inference.image_name.value_counts().index:
        lista = csv_inference.scores[csv_inference.image_name == image].values
        # print(lista)
        cadena = str()
        for i in lista:
            cadena = cadena + str(i) + ","
        lista_scores_concat.append(cadena)
    df_detections = pd.DataFrame(
        {"num_detecciones": csv_inference.image_name.value_counts(), "scores": lista_scores_concat},
        csv_inference.image_name.value_counts().index)
    df_final.loc[:, "num_tot_labeled"] = df_final["num_labels"]
    df_final_scores = df_final.join(df_detections).fillna(0)[
        ["num_tot_labeled", "num_com_labeled", "num_com_detected", "difference", "scores", ]]
    df_final_scores.to_csv('2_Code/output/requerimiento_daniel_septiembre.csv')

    return df_final_scores


def show_error_histogram_and_loss(e):
    """
    This function shows the histogram of the error and the corresponding Loss value.

    Parameters
    ----------
    e : array
        numpy array with error values (can be positive or negative).
    low : int
        lowest value of the error to show in the the histogram.
    high : int
        highest value of the error to show in the the histogram.


    """
    n_points = len(e)   
    low = np.min(e)
    high = np.max(e)
    # Number of observations.
    plt.hist(e,
             bins=np.arange(low + 0.5, high + 1 + 0.5, 1),
             density=False, facecolor='b', alpha=0.75, ec='black')  # Histogram.
    plt.xlabel('Error in counts')
    plt.ylabel('Histogram')
    title = 'Histogram of error. '  # Show the Loss function in the title.
    plt.title(title)
    plt.xlim(low, high)
    plt.ylim(0, n_points)

    vals = pd.Series(e).value_counts()
    plt.xticks(vals.index.tolist())
    plt.suptitle("Train Error Histogram", size=16)
    plt.title("Real - Prediction", size=12)
    for i in vals.index.tolist():
        plt.annotate(str(vals[i]), (i - .25, vals[i] + 10))
    # plt.xticks(np.arange(low, high+1, step=1))
    # plt.grid(True)

    plt.savefig("output/histogram_strau_diff_pred_vs_gtv2.png", ppp=300)

# 1 vals =>th
# 0 vals


# %%
def match_preds_labels_for_folder(folder_path, path_csv_predictions, path_csv_labels):
    images = [i for i in os.listdir(folder_path) if i[-4:] == ".jpg"]
    predictions = pd.read_csv(path_csv_predictions)
    labels = pd.read_csv(path_csv_labels)
    colors = [(0, 255, 0), (0, 0, 255)]
    bbox_thickness = 2
    accuracies = []

    # if not os.path.exists(folder_path + "../facades_predicted/"): os.mkdir(folder_path + "../facades_predicted/")
    for image in images:
        list_index_labels = np.where(labels.filename == image)[0].tolist()
        list_index_preds = np.where(predictions.filename == image)[0].tolist()
        labels.iloc[list_index_labels]
        if "Unnamed: 0" in predictions.columns:
            predictions = predictions.drop(["Unnamed: 0"], axis=1)
        content = predictions.iloc[list_index_preds]
        # draw the predictions of the model
        image_name = image
        print(image_name)
        image_data = cv2.imread("output/" + image_name)

        # draw_text_bb(image, box_c, classes_c - 1, scores_c)
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("output/9999_pred.jpg", image)
        # plt.figure(figsize=IMAGE_SIZE)
        # plt.imshow(image_rgb)
        # draw the ground truth and the iou values
        # image = cv2.imread("output/" + image_name)
        # draw_text_bb(image, box_gt, classes_gt, ious)
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # plt.figure(figsize=IMAGE_SIZE)
        # plt.imshow(image_rgb)

        box_gt, classes_gt = read_yolo_bbox2(image_name,path_csv_labels)

        bbox = np.zeros((len(content), 4), dtype=int)
        classes = []
        scores = []
        for i in list_index_preds:
            class_index, x1, x2, y1, y2= predictions.iloc[[i]][["class_index", "x1", "x2", "y1", "y2"]].values[0]
            class_index = int(class_index)-1
            score = predictions.iloc[[i]][["score"]].values[0]
            scores.append(score)
            classes.append(class_index)
            bbox[list_index_preds.index(i)] = x1, y1, x2, y2

            print(class_index, x1, x2, y1, y2, score[0])
            # color = colors[class_index-1]
            # cv2.rectangle(img_data, (x1, y1), (x2, y2), color, thickness=bbox_thickness)
            # img_data = draw_text_bb(img_data, str(score[0])[:3], (x1, y1 + 15), color, 1)



        # read the bounding boxes from the ground truth

        box_c, scores_c, classes_c = drop_same_building(bbox, scores, classes, trshld_px=3)
        # match the bounding of the ground truth and predictions and compute the
        # Intersection over union
        ious = match_pred_gt(box_gt, box_c, classes_gt, classes_c)

        accuracy = ious.mean()
        print("the accuraccy for the image {} is {:.2f}%".format(image_name, accuracy * 100))
        accuracies.append(accuracy)

    return accuracies


def match_with_iou(labels_commerce, pred_commerce, verbose = True):
    list_index_match_iou_p_labels = []
    list_index_match_iou_p_preds = []
    ious_d2=[]
    list_filenames = labels_commerce.filename.unique().tolist()
    N = len(list_filenames)
    for name_i in list_filenames:
        index = list_filenames.index(name_i)
        list_index_match_preds = np.where(pred_commerce.filename == name_i)[0].tolist()
        list_index_match_labels = np.where(labels_commerce.filename == name_i)[0].tolist()

        classes_gt = labels_commerce.iloc[list_index_match_labels][["class_index"]].values
        bbox_gt = labels_commerce.iloc[list_index_match_labels][["xmin", "ymin", "xmax", "ymax"]].values

        score = pred_commerce.iloc[list_index_match_preds][["score"]].values
        classes_pred = (pred_commerce.iloc[list_index_match_preds][["class_index"]]-1).values
        bbox_pred = pred_commerce.iloc[list_index_match_preds][["x1", "y1", "x2", "y2"]].values

        for i in range(len(classes_gt)):
            for j in range(len(classes_pred)):           
                iou = bb_intersection_over_union(bbox_gt[i], bbox_pred[j])
                if iou>0:
                    ious_d2.append({"id_gt": list_index_match_labels[i] , "id_pred": list_index_match_preds[j],
                                    "IOU": iou})
        # print(index+1,"/", N, name_i, end="\r")
        #print(ious_d2)
    new=pd.DataFrame(ious_d2).sort_values(
        "IOU",ascending=False).drop_duplicates(["id_pred"]).drop_duplicates(["id_gt"])
    N_new = len(new)
    if verbose: print("facade labels matched: "+str(N_new)+" of "+str(len(labels_commerce))+" with iou_mean: "+
          str(new.IOU.mean())+" and TP: "+str(len(new[new.IOU>0.5])))
    new=new.sort_index()
    list_index_match_iou_p_labels = new.id_gt.tolist()
    list_index_match_iou_p_preds = new.id_pred.tolist()
    ious = new.IOU.tolist()
    #print(ious.mean(), len(ious[ious>0.5]))
    return list_index_match_iou_p_labels, list_index_match_iou_p_preds
#list_index_match_iou_p_labels, list_index_match_iou_p_preds= match_with_iou(labels_commerce, pred_commerce)


def compute_eval_table(labels_commerce, pred_commerce, reindex_match_labels,reindex_match_preds, iou_th, verbose = True):
    
    if verbose: print("number of labels: ",len(reindex_match_labels),
          "number of preds: ",len(reindex_match_preds))    
    bboxes_labels = labels_commerce.iloc[reindex_match_labels][["xmin", "ymin","xmax","ymax"]].values
    bboxes_preds = pred_commerce.iloc[reindex_match_preds][["x1", "y1", "x2", "y2"]].values
    
    eval_table = pd.DataFrame()
    eval_table['filename'] = labels_commerce.iloc[reindex_match_labels].filename
    eval_table.loc[:,'IOU'] = [bb_intersection_over_union(bboxes_labels[i], bboxes_preds[i]) for i in range(len(bboxes_labels))]
    eval_table['TP/FP'] = eval_table['IOU'].apply(lambda x: 'TP' if x>=iou_th else 'FP')

    eval_table['class_index'] = labels_commerce.iloc[reindex_match_labels].class_index
    eval_table['class_index2'] = pred_commerce.iloc[reindex_match_preds].class_index.values-1
    eval_table.loc[:,'score'] = pred_commerce.iloc[reindex_match_preds].score.values
    eval_table.loc[:,'IOU'] = [bb_intersection_over_union(bboxes_labels[i], bboxes_preds[i]) for i in range(len(bboxes_labels))]
    eval_table['TP'] = eval_table['IOU'].apply(lambda x: 1 if x>=iou_th else 0)
    eval_table['FP'] = eval_table['IOU'].apply(lambda x: 1 if x<iou_th else 0) 
    eval_table = eval_table.sort_values("score", ascending=False)
    eval_table['acc TP'] = (eval_table.TP == 1).cumsum()
    eval_table['acc FP'] = (eval_table.FP == 1).cumsum()
    eval_table['precision'] = (eval_table["acc TP"])/(eval_table["acc TP"]+eval_table["acc FP"])
    eval_table['recall'] = (eval_table["acc TP"])/len(eval_table)
    

    return eval_table
#compute_eval_table(labels_commerce, pred_commerce, list_index_match_iou_p_labels,list_index_match_iou_p_preds)


def graph_mAP(eval_table, prec_at_rec):
    #fig, ax = plt.subplots(figsize=(20,12))
    eval_table.plot.scatter("recall", "precision", figsize=(20,12))#, ylim=[0.68,1.02])
    plt.fill_between(eval_table["recall"], eval_table["precision"])#, [0.68])

    PR = plt.step(eval_table["recall"], eval_table["precision"], lw=4, where='post')
    IR = plt.step(np.linspace(0.0, 1.0, 11), prec_at_rec, where='post', lw=4, color="orange")
        #recall['micro'], precision['micro'], where='post')

    plt.plot(np.linspace(0.0, 1.0, 11), prec_at_rec, 'or', label='11-point interpolated precision')
    plt.title('Precision x Recall curve \nClass: %s, AP: %s' % ("commerce", np.mean(prec_at_rec)), size="x-large")
    plt.xlabel('recall', fontsize='large')
    plt.xticks(fontsize='large')
    plt.ylabel('precision', fontsize='large')
    plt.yticks(fontsize='large')
    labels=["11-point interpolated precision", "Precision x Recall"]
    plt.legend([IR, PR],labels=labels , loc='upper right', shadow=True, fontsize=14)

    plt.grid()
#graph_mAP(eval_table, prec_at_rec)


def graph_hist_diff_comm(labels_dataset, pred_commerce):
    labels_commerce =labels_dataset[labels_dataset.class_index==0]
    df_tuning = pd.DataFrame({"num_tot_labels": labels_dataset.filename.value_counts()})
    df_tuning=df_tuning.join(labels_commerce.filename.value_counts()).fillna(0)
    df_tuning=df_tuning.join(pred_commerce.filename.value_counts(), rsuffix=2).fillna(0)
    df_tuning.loc[:,"num_com_labeled"]=df_tuning.filename.astype(np.int)
    df_tuning.loc[:,"num_com_detected"]=df_tuning.filename2.astype(np.int)
    df_tuning.loc[:,"difference"]=df_tuning.num_com_labeled- df_tuning.num_com_detected
    df_tuning =df_tuning[["num_tot_labels","num_com_labeled", "num_com_detected", "difference"]].sort_index()
    #df_tuning.difference.value_counts().plot.barh()
    plt.figure(figsize=(12,8))
    vals = np.arange(df_tuning.difference.min()+0.5, df_tuning.difference.max()+0.5+1, 1)
    N_vals = len(vals)
    ax = df_tuning.difference.plot.hist(bins=N_vals,histtype='bar',orientation='vertical', ec='black')
    vals2=df_tuning.difference.value_counts()
    plt.xticks(vals2.index.tolist())
    plt.title("Histogram calculated as: Real - Prediction \n in Model Selection", fontsize='large')
    for i in vals2.index.tolist():
        plt.annotate(str(vals2[i]), (i,vals2[i]+10))


def match_with_centers(labels_tuning, predictions_tuning):
    list_index_match_centers_labels = []
    list_index_match_centers_preds = []
    list_filenames = labels_tuning.filename.unique().tolist()
    N = len(list_filenames)
    for name_i in list_filenames:
        index = list_filenames.index(name_i)
        list_index_match_preds = np.where(predictions_tuning.filename == name_i)[0].tolist()
        list_index_match_labels = np.where(labels_tuning.filename == name_i)[0].tolist()
        box_gt, classes_gt = read_yolo_bbox2(name_i,"output/tuning_labels.csv")
        score = predictions_tuning.iloc[list_index_match_preds][["score"]].values
        classes_pred = (predictions_tuning.iloc[list_index_match_preds][["class_index"]]-1).values
        bbox_pred = predictions_tuning.iloc[list_index_match_preds][["x1", "y1", "x2", "y2"]].values
        centers_p = get_center(bbox_pred)
        centers_gt = get_center(box_gt)
        #match in centers
        Xc_gt = centers_gt[:, 0]
        Xc_p = centers_p[:, 0]
        for i in range(len(classes_gt)):
            Xrest = Xc_gt[i] - Xc_p
            if len(Xrest[np.abs(Xrest) < 15]) > 0:
                list_index_match_centers_labels.append(list_index_match_labels[i])
                list_index_match_centers_preds.append(list_index_match_preds[np.where(np.abs(Xrest) < 15)[0][0]])
        print(index+1,"/", N, name_i, end="\r")
    return list_index_match_centers_labels, list_index_match_centers_preds
#match_with_centers(labels_tuning, predictions_tuning)


def performance_detectors_metrics(predictions_dataset_file, labels_file, score_th, iou_th, plots=True, verbose=True ):
    #path ="output/predictions_databases/"
    #path
    #labels_file= "output/tuning_labels.csv"
    #score_th = 0.75
    predictions_dataset = pd.read_csv(predictions_dataset_file+".csv")#path
    #print(len(predictions_dataset[predictions_dataset.score>score_th]))
    predictions_dataset = predictions_dataset[predictions_dataset.score>score_th]
    #predictions_dataset.sort_values(by="filename")
    labels_dataset = pd.read_csv(labels_file)
    labels_dataset.loc[:,"match_center"]=0
    labels_dataset.loc[:,"match_class"]=0
    labels_dataset.loc[:,"class_index"]= [0 if i=="comercial" else 1 for i in labels_dataset["class"].tolist()]

    labels_commerce =labels_dataset[labels_dataset.class_index==0]
    
    #For Yolo
    if "class_index" not in predictions_dataset.columns:
        predictions_dataset.loc[:,"class_index"]= [0 if i=="comercial" else 1 
                                                   for i in predictions_dataset["clase"].tolist()]
        predictions_dataset.loc[:,"x1"] = predictions_dataset["xmin"]
        predictions_dataset.loc[:,"y1"] = predictions_dataset["ymin"]
        predictions_dataset.loc[:,"x2"] = predictions_dataset["xmax"]
        predictions_dataset.loc[:,"y2"] = predictions_dataset["ymax"]

    pred_commerce =predictions_dataset[predictions_dataset.class_index==1]
    #pred_commerce                
    list_index_match_iou_p_labels, list_index_match_iou_p_preds= match_with_iou(labels_commerce, pred_commerce, verbose=verbose)
    
    eval_table = compute_eval_table(labels_commerce, pred_commerce, 
                                    list_index_match_iou_p_labels,list_index_match_iou_p_preds, iou_th, verbose = verbose)
    eval_table_i50 = compute_eval_table(labels_commerce, pred_commerce, 
                                    list_index_match_iou_p_labels,list_index_match_iou_p_preds, 0.5, verbose = verbose)
    
    #print(eval_table)
    if verbose: print(eval_table['TP/FP'].value_counts())
    if verbose: print(len(eval_table))
    if verbose: print(len(labels_commerce))
    precision = (eval_table['TP/FP'].value_counts()/len(eval_table))["TP"]
    if verbose: print("precision in commerce: ",precision)
    recall= (eval_table['TP/FP'].value_counts()/len(labels_commerce))["TP"]
    if verbose: print("recall in commerce: ",recall)
    f1_score = 2*((precision*recall)/(precision+recall))
    if verbose: print("f1_score in commerce:",f1_score)
    #print((eval_table['TP/FP'].value_counts()/(
    #    len(labels_commerce)+eval_table['TP/FP'].value_counts()["FP"]))["TP"])
    #eval_table['TP/FP'].value_counts()["FP"]+len(labels_commerce)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        #print(recall_level)
        x = eval_table_i50[eval_table_i50['recall'] >= recall_level]['precision']
        prec = max(x) if len(x)>0 else 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)
    #print('11 point precision is ', prec_at_rec)
    if verbose: print('AP for ', predictions_dataset_file, " is ", avg_prec)

    if plots: 
        graph_mAP(eval_table, prec_at_rec)
        graph_hist_diff_comm(labels_dataset, pred_commerce)
    
    return avg_prec, precision, recall, f1_score, eval_table.IOU.mean(), len(eval_table)


def performance_commerce(predictions_dataset_file, path, labels_file, score_th):
    #labels_file= "output/tuning_labels.csv"
    #path ="output/predictions_databases/"
    predictions_dataset = pd.read_csv(path+predictions_dataset_file+".csv")
    predictions_dataset = predictions_dataset[predictions_dataset.score>score_th]
    #predictions_dataset.sort_values(by="filename")
    labels_dataset = pd.read_csv(labels_file)
    labels_dataset.loc[:,"class_index"]= [0 if i=="comercial" else 1 for i in labels_dataset["class"].tolist()]
    #labels_dataset
    labels_commerce =labels_dataset[labels_dataset.class_index==0]
    
    #labels_commerce
    #print(predictions_dataset_file, predictions_dataset.columns)
    if "class_index" not in predictions_dataset.columns:
        predictions_dataset.loc[:,"class_index"]= [0 if i=="comercial" else 1 
                                                   for i in predictions_dataset["clase"].tolist()]
    pred_commerce =predictions_dataset[predictions_dataset.class_index==1]
    df_tuning = pd.DataFrame({"num_tot_labels": labels_dataset.filename.value_counts()})
    df_tuning=df_tuning.join(labels_commerce.filename.value_counts()).fillna(0)
    df_tuning=df_tuning.join(pred_commerce.filename.value_counts(), rsuffix=2).fillna(0)
    df_tuning.loc[:,"num_com_labeled"]=df_tuning.filename.astype(np.int)
    df_tuning.loc[:,"num_com_detected"]=df_tuning.filename2.astype(np.int)
    df_tuning.loc[:,"difference"]=df_tuning.num_com_labeled- df_tuning.num_com_detected
    df_tuning =df_tuning[["num_tot_labels","num_com_labeled", "num_com_detected", "difference"]].sort_index()
    p_com_img=len(df_tuning[(df_tuning.difference==0)])/len(df_tuning)
    print("%2.2f percent of commerce, %d of %d images "%( p_com_img*100, len(df_tuning[(df_tuning.difference==0)]), len(df_tuning)))
    coinc_commer = len(df_tuning[(df_tuning.difference==0) & (df_tuning.num_com_labeled>0)])
    print("%d coincidences of %d images where there is commerce"%(coinc_commer, len(df_tuning[df_tuning.num_com_labeled>0])))
    mrse_commer = (np.power(df_tuning.difference.tolist(),2).sum()/len(df_tuning))**0.5
    print("Root mean squared error: %2.4f "%(mrse_commer))
    #return (len(df_tuning[(df_tuning.difference==0)])/len(df_tuning)
    #return (len(df_tuning[(df_tuning.difference==0) & (df_tuning.num_com_labeled>0)]))
    return p_com_img, coinc_commer, mrse_commer


def performance_commerce2(predictions_dataset_file, labels_file, score_th, verbose = True):
    """
    Compute the commerce metrics proposed in the thesis methodology.
    """
    #predictions_dataset_file = "prediction_tuning_inference_graph_base_line"
    #path = "../4_Outputs/predictions_databases/preds_0_1_th/"
    #labels_file = "../4_Outputs/tuning_labels.csv"
    #score_th = 0.45

    predictions_dataset = pd.read_csv(predictions_dataset_file+".csv")
    predictions_dataset = predictions_dataset[predictions_dataset.score>score_th]
    #predictions_dataset.sort_values(by="filename")
    labels_dataset = pd.read_csv(labels_file)
    labels_dataset.loc[:,"class_index"]= [0 if i=="comercial" else 1 for i in labels_dataset["class"].tolist()]
    #labels_commerce =
    labels_dataset[labels_dataset.class_index==0] ##
    if "class_index" not in predictions_dataset.columns:
        predictions_dataset.loc[:,"class_index"]= [0 if i=="comercial" else 1 
                                                   for i in predictions_dataset["clase"].tolist()]    
    #pred_commerce_dataset =predictions_dataset[(predictions_dataset.class_index==1) &(predictions_dataset.score>score_th)]##
    df_dataset = pd.DataFrame({"num_tot_labels": labels_dataset.filename.value_counts()})
    df_dataset=df_dataset.join(labels_dataset[labels_dataset.class_index==0].filename.value_counts()).fillna(0)##
    df_dataset=df_dataset.join(predictions_dataset[(predictions_dataset.class_index==1) &(predictions_dataset.score>score_th)
                                                  ].filename.value_counts(), rsuffix=2).fillna(0)##
    df_dataset.loc[:,"num_com_labeled"]=df_dataset.filename.astype(np.int)
    df_dataset.loc[:,"num_com_detected"]=df_dataset.filename2.astype(np.int)
    df_dataset.loc[:,"difference"]=df_dataset.num_com_labeled- df_dataset.num_com_detected
    df_dataset =df_dataset[["num_tot_labels","num_com_labeled", "num_com_detected", "difference"]].sort_index()
    
    #df_dataset_real_comm = df_dataset[df_dataset.num_com_labeled>0]##
    #df_dataset_not_real_comm = df_dataset[df_dataset.num_com_labeled==0]

    com_pos_det = df_dataset[df_dataset.num_com_labeled>0].num_com_detected
    com_pos_lab = df_dataset[df_dataset.num_com_labeled>0].num_com_labeled
    com_neg_det = df_dataset[df_dataset.num_com_labeled==0].num_com_detected
    com_neg_lab = df_dataset[df_dataset.num_com_labeled==0].num_com_labeled
    df_dataset.loc[df_dataset.num_com_labeled>0,"p_comm"]= com_pos_det/df_dataset[
    df_dataset.num_com_labeled>0].num_com_labeled
    com_pos_p_c = df_dataset[df_dataset.num_com_labeled>0].p_comm
    
    if verbose: print("model_name: ",predictions_dataset_file)
    if verbose: print((com_pos_det/com_pos_lab).sum())
    if verbose: print(len(df_dataset[df_dataset.num_com_labeled>0]))
    ML = (com_pos_det/com_pos_lab).sum()/len(df_dataset[df_dataset.num_com_labeled>0])
    if verbose: print("Mean P_comm", com_pos_p_c.mean())#, ML)
    if verbose: print("Standard Desviation P_comm", com_pos_p_c.std())
    E = com_neg_det.sum()/len(df_dataset[df_dataset.num_com_labeled==0])
    if verbose: print("Mean ERR_comm", com_neg_det.mean())#, E)
    if verbose: print("Standard Desviation ERR_comm", com_neg_det.std())
    return com_pos_p_c.mean(), com_pos_p_c.std(),com_neg_det.mean(), com_neg_det.std()  #ML, E


def compute_map_for_score_iou(labels_file, predictions_file, path, score_th, iou_th):
    #labels_file= "output/tuning_labels.csv"
    #score_th = 0.75
    predictions_dataset = pd.read_csv(path+predictions_file+".csv")
    #print(len(predictions_dataset[predictions_dataset.score>score_th]))
    predictions_dataset = predictions_dataset[predictions_dataset.score>score_th]
    #predictions_dataset.sort_values(by="filename")
    labels_dataset = pd.read_csv(labels_file)
    labels_dataset.loc[:,"match_center"]=0
    labels_dataset.loc[:,"match_class"]=0
    labels_dataset.loc[:,"class_index"]= [0 if i=="comercial" else 1 for i in labels_dataset["class"].tolist()]

    labels_commerce =labels_dataset[labels_dataset.class_index==0]
    
    #For Yolo
    if "class_index" not in predictions_dataset.columns:
        predictions_dataset.loc[:,"class_index"]= [0 if i=="comercial" else 1 
                                                   for i in predictions_dataset["clase"].tolist()]
        predictions_dataset.loc[:,"x1"] = predictions_dataset["xmin"]
        predictions_dataset.loc[:,"y1"] = predictions_dataset["ymin"]
        predictions_dataset.loc[:,"x2"] = predictions_dataset["xmax"]
        predictions_dataset.loc[:,"y2"] = predictions_dataset["ymax"]

    pred_commerce =predictions_dataset[predictions_dataset.class_index==1]
    
    list_index_match_iou_p_labels, list_index_match_iou_p_preds= match_with_iou(labels_commerce, pred_commerce, verbose=verbose)
    
    eval_table = compute_eval_table(labels_commerce, pred_commerce, 
                                    list_index_match_iou_p_labels,list_index_match_iou_p_preds, iou_th)
    
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        #print(recall_level)
        x = eval_table[eval_table['recall'] >= recall_level]['precision']
        prec = max(x) if len(x)>0 else 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)
    #print('11 point precision is ', prec_at_rec)
    print('mAP for ', predictions_file, " is ", avg_prec)
    max_prec, max_recall = (eval_table[len(eval_table)-1:len(eval_table)][["precision", "recall"]].values.tolist()[0][0],
     eval_table[len(eval_table)-1:len(eval_table)][["precision", "recall"]].values.tolist()[0][1])
    print(max_prec, max_recall)
    return max_prec, max_recall


def comparate_models_by_split(split, prediction_folder,labels_file, score_th=0.45,iou_th =0.1):
    list_comparison = []
    #split = "tuning"
    #performance_commerce(i[:-4])[2]] 
    #prediction_folder = "../4_Outputs/predictions_databases/preds_0_1_th/"
    for i in os.listdir(prediction_folder):
        if i.startswith("prediction_"+split+"_"):
            #print(i)
            p_com_img, p_com_img_std, E, E_std = \
            performance_commerce2(prediction_folder+"/"+i[:-4], # i[:-4],
                                  labels_file, # "../4_Outputs/"+split+"_labels.csv",
                                  score_th=score_th, verbose=False)
            avg_prec, precision, recall, f1_score, IOU, n_eval = \
            performance_detectors_metrics(prediction_folder+"/"+i[:-4],labels_file,score_th=score_th, #i[:-4],
                                          iou_th =iou_th, plots=False, verbose = False)
            #list_comparison.append((i[34:], p_com_img, E))
            list_comparison.append(("_".join(i.split(".")[0].split("_")[2:]), p_com_img, p_com_img_std, E, E_std, avg_prec, 
                                    precision, recall, f1_score))

    comparison_split = pd.DataFrame(list_comparison,columns= ["name", "p_com_mean", "p_com_std", "E", "E_std", "avg_prec", 
                                                               "precision",    "recall", "f1_score"])
    return comparison_split


# %%Main
if __name__ == '__main__':
    prediction_folder = sys.argv[1]
    labels_file = sys.argv[2]
    score_th = sys.argv[3]
    IOU_th = sys.argv[4]
    print(sys.argv[1], sys.argv[2], sys.argv[3])
    # df_final = join_real_prediction("2_Code/output/all_train_labels.csv","2_Code/output/csv_para_Daniel.csv")
    # #df_final_tuning = join_real_prediction("2_Code/output/all_train_labels.csv","2_Code/output/csv_para_Daniel.csv")
    # #pd.read_csv(path_csv_pred)
    # TN_map, FP_map, FN_map, TP_map, TN, FP, FN, TP, P, N, CM, observations, FP_rate, TP_rate, precision, recall, \
    # accuracy, F_measure, specificity, error_rate = performance_metrics_from_binary_classification(
    #     df_final.num_com_labeled.values, df_final.num_com_detected.values)
    #
    # show_error_histogram_and_loss(df_final.difference.fillna(0).values)

    # accuracies = match_preds_labels_for_folder("1_Data/images_resize/tuning/",
    #                                            "1_Data/images_resize/prediction_tuning_baseline.csv",
    #                                            "2_Code/output/tuning_labels.csv")
    # print(pd.DataFrame(accuracies).mean())
    performance_commerce2(prediction_folder,#"data/", "prediction_tuning_inference_graph_train_80_l2_001",
                          labels_file, #"data/tuning_labels.csv",
                          float(score_th))# 0.45)
    avg_prec, precision, recall, f1_score, IOU, n_eval = performance_detectors_metrics(prediction_folder,
        #"prediction_tuning_inference_graph_train_80_l2_001",
        #"../4_Outputs/predictions_databases/preds_0_1_th/",
        labels_file, #"../4_Outputs/tuning_labels.csv",
        score_th=float(score_th), iou_th=float(IOU_th))

    comparison_tuning = comparate_models_by_split("tuning",
                                                  "/".join(prediction_folder.split("/")[:-1]),#"../4_Outputs/predictions_databases/preds_0_1_th/",
                                                  labels_file,
                                                  float(score_th),
                                                  float(IOU_th))
    print(comparison_tuning)
    # score_th=0.45, iou_th =0.1  TESIS
    name_model = "frozen_graphs"  # "train_dropout_l2_1_p1.csv"#
    comparison_table = comparison_tuning.set_index("name").loc[name_model].to_frame().rename(
        columns={name_model: "tuning"})
    comparison_table.columns.name = name_model
    #comparison_table.loc[:, "tuning"] = comparison_tuning.set_index("name").loc[name_model]
    #comparison_table.loc[:, "test"] = comparison_test.set_index("name").loc[name_model]
    print(comparison_table[:4])
    print(comparison_table[4:])

