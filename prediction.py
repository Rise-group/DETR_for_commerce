import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import pandas as pd
#import zipfile
import cv2
from PIL import Image
# from imgaug import augmenters as iaa
from glob import glob


# %%
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# %%
def get_center(bbox):
    centers = np.zeros((bbox.shape[0],2), dtype=int)
    for i in range(bbox.shape[0]):
        (x1, y1, x2, y2) = bbox[i]
        center = (int((x2-x1)/2)+x1),(int((y2-y1)/2)+y1)
        centers[i] = center
    return centers


def scaling(img, p):
    return cv2.resize(img, (int(p * img.shape[1]), int(p * img.shape[0])))
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


def scale_image_in_folder(folder_path, sub_folder=''):
    # images = os.listdir(folder_path + sub_folder)
    images = [i for i in os.listdir(folder_path + sub_folder) if i[-4:] == ".jpg"]
    aug = iaa.GaussianBlur(sigma=(15))
    if not os.path.exists(folder_path + sub_folder + "resize/"): os.mkdir(folder_path + sub_folder + "resize/")
    for image in images:
        print("resizing image %3d / %3d" % (images.index(image), len(images)), end="\r")
        image_name = image
        filename, original_extension = os.path.splitext(image_name)
        # sub_folder = "fachadas_laterales/"
        img_data = cv2.imread(folder_path + sub_folder + filename + original_extension, cv2.COLOR_BGR2RGB)
        # img_data =cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

        img_data_blur = aug.augment_image(img_data)

        img_part = scaling(img_data_blur, 0.04)
        cv2.imwrite(folder_path + sub_folder + "resize/" + filename + original_extension, img_part)


# %%
def make_prediction_for_a_folder(detector_name, folder, output_path=None, th=0.5):

    # MODEL_NAME = 'pretrained_models/inference_graph'
    # MODEL_NAME = 'pretrained_models/inf9'
    MODEL_NAME = detector_name
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    # DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('pretrained_models', 'labelmap.pbtxt')

    root_detector = os.path.dirname(detector_name)
    root_folder = os.path.dirname(folder)

    root_list_detector = root_detector.split("/")
    root_list_folder = root_folder.split("/")
    if output_path is None: output_path = "predictions/prediction_" + root_list_folder[
        -1] + "_" + detector_name + ".csv"


    detections = []
    cont = 0

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    TEST_IMAGE_PATHS = os.listdir(folder)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            ops = tf.get_default_graph().get_operations()
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # detection_SSP = detection_graph.get_tensor_by_name('SecondStagePostprocessor/ToFloat_1:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in TEST_IMAGE_PATHS:
                print("predicting images:   %d/%d" % (TEST_IMAGE_PATHS.index(image_path), len(TEST_IMAGE_PATHS)), end="\r")
                image = Image.open(folder+image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                bbo_5 = boxes[scores > th]
                classes_5 = classes[scores > th].astype(np.int32)
                scores_5 = scores[scores > th]

                # Convert Bounding_boxes from relative to nominal
                bbo_5_full = (bbo_5[:, [1, 0, 3, 2]] * (332, 133, 332, 133)).astype(np.int32)

                # Delete the detections of the instances if has another with higher score
                # box_c, scores_c, classes_c = drop_same_building(bbo_5_full, scores_5, classes_5, trshld_px=3)

                centers = get_center(bbo_5_full)
                xc = centers[:, 0]
                for i in range(len(xc)):
                    (x1, y1, x2, y2) = bbo_5_full[i]
                    detection_image = {'filename': image_path, "image_id":TEST_IMAGE_PATHS.index(image_path), 'id_bbox': cont, 'class_index': classes_5[i], "score":scores_5[i], 'x1': x1,
                                       'y1': y1, 'x2': x2, 'y2': y2}
                    detections.append(detection_image)
                    cont += 1
                if (TEST_IMAGE_PATHS.index(image_path) % 1000 == 0) and (TEST_IMAGE_PATHS.index(image_path) != 0 ):
                    labels_all = pd.DataFrame.from_dict(detections)
                    labels_all.to_csv(output_path, index=False)

    labels_all = pd.DataFrame.from_dict(detections)

    labels_all.to_csv(output_path, index = False)
    return labels_all


# %%
def make_prediction_for_a_folder2(detector_name, folder, th=0.5):

    # MODEL_NAME = 'pretrained_models/inference_graph'
    # MODEL_NAME = 'pretrained_models/inf9'
    MODEL_NAME = detector_name
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    # DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('pretrained_models', 'labelmap.pbtxt')

    NUM_CLASSES = 2

    class_list = ["comercial", "no comercial"]
    image_name_rsz = "9999_rsz.jpg"

    root_detector = os.path.dirname(detector_name)
    root_folder = os.path.dirname(folder)

    root_list_detector = root_detector.split("/")
    root_list_folder = root_folder.split("/")



    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # PATH_TO_TEST_IMAGES_DIR = 'output'
    #PATH_TO_TEST_IMAGES_DIR = folder
    IMAGE_SIZE = (12, 8)

    #TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, image_name_rsz)]
    #PATH_TO_TEST_IMAGES_DIR
    # TEST_IMAGE_PATHS = os.listdir(folder)


    subfolders = os.listdir(folder)
    print(subfolders)
    images_in_subfolder = []
    for subfolder in subfolders[18:]:
        print("####  predicting for the folder: "+folder+subfolder)
        images_in_subfolder = glob(folder+subfolder+"/" + '*.jpg')

        TEST_IMAGE_PATHS = images_in_subfolder

        detections = []
        cont = 0
        labels_all = pd.DataFrame()

        if os.path.exists(folder+"/../prediction_"+subfolder+"_"+root_list_detector[-1]+".csv"):
            labels_all = pd.read_csv(folder+"/../prediction_"+subfolder+"_"+root_list_detector[-1]+".csv")
            downloaded = len(sample_points)-1
            print("there is a csv with the metadata, i will start where it finished, in the point: " + str(downloaded))
            predicted = labels_all.filename.unique()
        else:
            downloaded=0
            predicted = []




        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                ops = tf.get_default_graph().get_operations()
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # detection_SSP = detection_graph.get_tensor_by_name('SecondStagePostprocessor/ToFloat_1:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                for image_path in TEST_IMAGE_PATHS:
                    if image_path in predicted: 
                        print(image_path, " was previously_predicted")
                        continue
                    print("predicting images:   %d/%d,   filename: %s" % (TEST_IMAGE_PATHS.index(image_path), len(TEST_IMAGE_PATHS), image_path), end="\r")
                    image = Image.open(image_path)
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = load_image_into_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    # vis_util.visualize_boxes_and_labels_on_image_array(
                    #     image_np,
                    #     np.squeeze(boxes),
                    #     np.squeeze(classes).astype(np.int32),
                    #     np.squeeze(scores),
                    #     category_index,
                    #     use_normalized_coordinates = True,
                    #                              line_thickness = 4)
                    #plt.figure(figsize=IMAGE_SIZE)
                    #plt.imshow(image_np)
                    bbo_5 = boxes[scores > th]
                    classes_5 = classes[scores > th].astype(np.int32)
                    scores_5 = scores[scores > th]

                    # Convert Bounding_boxes from relative to nominal
                    bbo_5_full = (bbo_5[:, [1, 0, 3, 2]] * (332, 133, 332, 133)).astype(np.int32)

                    # Delete the detections of the instances if has another with higher score
                    box_c, scores_c, classes_c = drop_same_building(bbo_5_full, scores_5, classes_5, trshld_px=3)

                    centers = get_center(box_c)
                    xc = centers[:, 0]
                    for i in range(len(xc)):
                        (x1, y1, x2, y2) = box_c[i]
                        detection_image = {'filename': image_path, "image_id":TEST_IMAGE_PATHS.index(image_path), 'id_bbox': cont, 'class_index': classes_c[i], "score":scores_c[i], 'x1': x1,
                                           'y1': y1, 'x2': x2, 'y2': y2}
                        detections.append(detection_image)
                        cont += 1
                    if (TEST_IMAGE_PATHS.index(image_path) % 1000 == 0) and (TEST_IMAGE_PATHS.index(image_path) != 0 ):
                        labels_all = pd.DataFrame.from_dict(detections)
                        labels_all.to_csv(folder+"/../prediction_"+subfolder+"_"+root_list_detector[-1]+".csv",index = False)

        labels_all = pd.DataFrame.from_dict(detections)

        labels_all.to_csv(folder+"/../prediction_"+subfolder+"_"+root_list_detector[-1]+".csv", index = False)
    return labels_all


# %%
def draw_prediction_for_a_folder(folder_path, path_csv_predictions):
    images = [i for i in os.listdir(folder_path) if i[-4:] == ".jpg"]

    predictions = pd.read_csv(path_csv_predictions)
    colors = [(0, 255, 0), (0, 0, 255)]
    bbox_thickness = 2
    if not os.path.exists(folder_path + "../facades_predicted/"): os.mkdir(folder_path + "../facades_predicted/")
    for image in images:
        list_index_match = np.where(predictions.filename == image)[0].tolist()

        print("drawing predicitions for image %3d / %3d" % (images.index(image), len(images)), end="\r")
        image_name = image
        filename, original_extension = os.path.splitext(image_name)
        img_data = cv2.imread(folder_path + filename + original_extension, cv2.COLOR_BGR2RGB)

        for i in list_index_match:
            class_index, x1, x2, y1, y2= predictions.iloc[[i]][["class_index", "x1", "x2", "y1", "y2"]].values[0]
            score = predictions.iloc[[i]][["score"]].values[0]
            print(class_index, x1, x2, y1, y2, score[0])
            color = colors[class_index-1]
            cv2.rectangle(img_data, (x1, y1), (x2, y2), color, thickness=bbox_thickness)
            img_data = draw_text(img_data, str(score[0])[:3], (x1, y1 + 15), color, 1)

        cv2.imwrite(folder_path + "../facades_predicted/" + filename + original_extension, img_data)

    return img_data


# %%
def draw_text(tmp_img, text, center, color, size):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(tmp_img, text, center, font, 1, color, size, cv2.LINE_AA)
    return tmp_img
# %% # Main


if __name__ == '__main__':
    fg_folder = sys.argv[1]
    score_th = sys.argv[2]
    folder_images = sys.argv[3]
    output_directory = sys.argv[4]

    make_prediction_for_a_folder(fg_folder,
                                 folder_images,
                                 output_path=output_directory,
                                 th = float(score_th))


    # draw_prediction_for_a_folder(folder+'images/facades_resized/', folder+"prediction_database.csv")

    #scale_image_in_folder('images/facades_projection/')
    #make_prediction_for_a_folder("/mnt/datos/juanCamilo/tensorflow-models/research/object_detection/inference_graph",
    #                             'images/facades_projection/resize/', th=0.5)
    #draw_prediction_for_a_folder(folder + 'images/facades_resized/', folder + "prediction_database.csv")