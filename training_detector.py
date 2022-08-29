# P10
# Training Model
# Inputs
# 1. Folder with augmented images for training
# 2. Human Expertise
# 4. Object detector architecture (Faster RCNN)
# 3. Object detector hyperparameters
# 6. Percentages for data splits
# 10. Parameters for data augmentation
# Outputs
# 13. Trained detector of commercial establishments
# 14. Performance metrics for training.
# 15. Performance metrics for model selection
# 16. Best object detector hyperparameters
# 17. Performance metrics for testing


# %%
# P14
# Training Detector
# Inputs
# 11. Folder with images for training (augmented or not)  --> 1_Data/data augmented_csv  OR  7. Folder with images
#     for training--> 1_Data/images_resize/train
# 8.  Folder with images for model selection  --> 1_Data/images_resize/tuning
# 12. File with labels for images (.csv) (augmented or not) OR 5. File with labels (.csv) -->output/all_train_labels.csv
# 4.  Object detector architecture (Faster RCNN)
# 3.  Object detector hyperparameters

# Outputs
# 13. Trained detector of commercial establishments
# 14. Performance metrics for training.
# 15. Performance metrics for model selection
# 16. Best object detector hyperparameters


import os
from glob import glob
import numpy as np
import pandas as pd
import shutil
# import cv2


# %%
def make_training_labels_db(path_img_training_folder):  # , path_output_shp, distance, random):
    """
    Take a shapefile of points, drop the nearest points in a distance values in meters.
    Save a shapefile with the output


    Parameters
    ----------
    path_img_training_folder : string
        Folder with images for training
    # path_output_shp: string
    #     the path to the folder_path in which the output shapefile will be saved
    # distance: Integer
    #     distance threshold in meters

    Returns
    -------
    df_training_labels: DataFrame
        list with the indexes
    """
    image_list = glob(path_img_training_folder + '*.jpg')
    print("there are " + str(len(image_list)) + " images in folder_path ")
    image_list.sort()

    train = []
    df_train_labels = pd.read_csv("2_Code/output/all_train_labels.csv")

    for i in image_list:
        name = i[-8:]
        for j in np.where(df_train_labels.filename == name)[0]:
            train.append(j)

    df_training_labels = df_train_labels.iloc[train]

    print("there are " + str(len(df_training_labels)) + " labels in subset")
    return df_training_labels.reset_index(drop=True)


# %%
def get_just_a_percent(df, percent, shuffle=False):

    files_list = df.filename.unique()   # pd.Serie

    if shuffle:
        idxs = np.random.permutation(range(len(files_list)))
    else:
        idxs = list(range(len(files_list)))

    until = int(percent * len(files_list))
    print("working with " + str(until) + " images")
    train = []
    for i in files_list[idxs[:until]]:
        for j in np.where(df.filename == i)[0]:
            train.append(j)
    # files_list
    print(str(len(df.iloc[train][df.iloc[train]["class"]=="comercial"]))+" comercials, "+
          str(len(df.iloc[train][df.iloc[train]["class"] == "no_comercial"]))+" no_comercials")
    return df.iloc[train].reset_index(drop=True)


def copy_away_images(path_img_training_folder, list_files, new_folder):
    folder = path_img_training_folder + "../" + new_folder + "/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    else:
        print("folder_path "+new_folder+" exist, please delete it")

    for file in list_files:
        print(file, end="\r")
        shutil.copyfile(path_img_training_folder + file, folder + file)


def cut_away_images(path_img_training_folder, list_files, new_folder):
    folder = path_img_training_folder + "../" + new_folder + "/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    else:
        print("folder_path "+new_folder+" exist, please delete it")

    for file in list_files:
        print(file, list_files.index(file),"from "+str(len(list_files)) , end="\r")
        shutil.move(path_img_training_folder + file, folder + file)


def check_for_folder(folder, metadata_shp):
    """
    check all the images in a folder respect to a shapefile, in the field panoid, with images from google api
    read the shape and the files in the folder
    extract de panoids: files[:22], just the uniques
    reindex the files in the shapefile
    create a automatic filename for output and save it as .shp


    Parameters
    ----------
    folder : string
        the path to the folder in which are the images downloaded
    metadata_shp: string
        the path to the GeoDataFrame read by geopandas of the shapefile of points with panoid column

    Returns
    -------
    """    
    
    #folder = "output/shapefiles/Valle_de_Aburra/"
    metadata = gpd.GeoDataFrame.from_file(metadata_shp)
    
    #folder_comunas ="images/comunas/"
    images_in_folder = os.listdir(folder)
    print(len(images_in_folder))
    panoids_in_folder = [i[:22] for i in images_in_folder]
    print(panoids_in_folder)
    u_panoids=pd.Series(panoids_in_folder).unique()
    print(len(u_panoids))
    #cont_comunas_panoids_2018
    print("number of images in folder "+folder+": "+str(len(u_panoids))+"/"+str(len(metadata.panoid.value_counts())))
    reindex = []
    reindex2 = []
    cont=0
    for i in u_panoids:
        cont+=1
        #print(np.where(metadata_2018.panoid==i)[0].tolist())
        reindex = reindex+ np.where(metadata.panoid==i)[0].tolist()
        reindex2.append(np.where(metadata.panoid==i)[0][0]) if len(np.where(metadata.panoid==i)[0])>0 else None
        print(str(cont)+" / "+str(len(u_panoids))+" images reindexed "+str(len(reindex)), end="\r")
    string = ""
    for i in metadata_shp.split("/")[:-1]: string =string+i+"/"
    output_shp= string+folder.replace("/","_")+metadata_shp.split("/")[-2][-4:]+".shp"
    print("saving in: "+output_shp)
    metadata.iloc[reindex].to_file(output_shp)


def check_for_folder_and_subfolder(folder, metadata_shp):
    """
    check all the images in the subfolders of the folder respect to a shapefile, in the field panoid, with images from google api
    read the shape and the files in the folder
    extract de panoids: files[:22], just the uniques
    reindex the files in the shapefile
    create a automatic filename for output and save it as .shp


    Parameters
    ----------
    folder : string
        the path to the folder in which are the images downloaded
    metadata_shp: string
        the path to the GeoDataFrame read by geopandas of the shapefile of points with panoid column

    Returns
    -------
    """    
    #folder = "output/shapefiles/Valle_de_Aburra/"
    metadata = gpd.GeoDataFrame.from_file(metadata_shp)
    
    #folder_comunas ="images/comunas/"
    subfolder = os.listdir(folder)
    print(subfolder)
    images_in_subfolder = []
    for i in subfolder:images_in_subfolder = images_in_subfolder + os.listdir(folder+i)
    print(len(images_in_subfolder))
    panoids_in_subfolder = [i[:22] for i in images_in_subfolder]
    u_panoids=pd.Series(panoids_in_subfolder).unique()
    print(len(u_panoids))
    #cont_comunas_panoids_2018
    print("number of images in folder "+folder+": "+str(len(u_panoids))+"/"+str(len(metadata.panoid.value_counts())))
    reindex = []
    reindex2 = []
    cont=0
    for i in u_panoids:
        cont+=1
        print(np.where(metadata.panoid==i)[0].tolist())
        reindex = reindex+ np.where(metadata.panoid==i)[0].tolist()
        reindex2.append(np.where(metadata.panoid==i)[0][0]) if len(np.where(metadata.panoid==i)[0])>0 else None
        print(str(cont)+" / "+str(len(u_panoids))+" images reindexed", end="\r")
    string = ""
    for i in metadata_shp.split("/")[:-1]: string =string+i+"/"
    output_shp= string+folder.replace("/","_")+metadata_shp.split("/")[-2][-4:]+".shp"
    print("saving in: "+output_shp)
    metadata.iloc[reindex].to_file(output_shp)


def cut_away_images2(path_img_training_folder, list_files, new_folder):
    folder = path_img_training_folder + "../" + new_folder + "/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
        moved = []
    else:
        moved = os.listdir(folder)

    for file in list_files:
        print(file, list_files.index(file),"from "+str(len(list_files)) , end="\r")
        if file in moved:
            print(file+" was in folder, skiping")
            continue
        shutil.move(path_img_training_folder + file, folder + file)


def move_images_to_subfolder_from_shp(folder,metadata_shp):
    metadata = gpd.GeoDataFrame.from_file(metadata_shp)

    #folder = "images/municipios/facades_api/"
    images_in_folder = os.listdir(folder)
    print(len(images_in_folder))

    total = 0
    subfolders=metadata.nombre_car.unique()
    for j in subfolders:
        print("searching for the subfolder: "+j+" the number "+str(subfolders.tolist().index(j))+
              " of "+str(len(subfolders)))
        list_i=[]
        cont=0
        list_u = metadata[metadata.nombre_car==j].panoid.unique().tolist()
        for i in images_in_folder:
            cont+=1
            if i[:22] in list_u:
                list_i.append(i)
            # print(str(cont)+" / "+str(len(images_in_folder))+" images searching", end="\r")
        cut_away_images2(folder,list_i,j)
        print("there are "+str(len(list_i))+" coincidences in the subfolder "+j)
        total+=len(list_i)

    print(total, "/", len(images_in_folder), " of images moved")


# %%
if __name__ == "__main__":
    # input
    # cut_away_images("2_Code/output/shapefiles/Valle_de_Aburra/images/facades_api/", lista_downloads_caldas, "caldas")

    image_training_folder = os.getcwd() + "/" + "1_Data/images_resize/train/"
    df_training_labels = make_training_labels_db(image_training_folder)
    print(str(len(df_training_labels[df_training_labels["class"]=="comercial"]))+" comercials, "
          +str(len(df_training_labels[df_training_labels["class"] == "no_comercial"]))+" no_comercials")

    for i in [0.5, 0.65, 0.8]:
        name = "train_" + str(int(i * 100)) + "p"
        print("creating the csv " + name + ".csv and the folder_path " + name)
        df_training_labels_p = get_just_a_percent(df_training_labels, i, False)


        df_training_labels_p.to_csv(image_training_folder+"../"+name+".csv", index=False)
        files_list = df_training_labels_p.filename.unique()
        copy_away_images(image_training_folder, files_list, name)

# ### Example####
# root = '2_Code'
# inputShp = os.path.join(root,'output/shapefiles/Envigado/street_points_20m_2.shp')
# outputTxt = root
# GSVpanoMetadataCollector(inputShp,1000,outputTxt)

# Procces:

# match entre images and labels      OK --> make_training_labels_db
# %%
# apply jairo's suggestion (50%,75%,100% of data)
# %%
# Generate tf_record(unaugmented)
# train with diference architectures and hyperparameters
# export inference_graph
# something similar to gridsearch for select best hyperparameters

# tf_records, train_p, ssd_mobilenet_v1_coco_2017_11_17, faster_rcnn_inception_v2_coco_2018_01_28
# https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
# https://github.com/tensorflow/models

conda env remove -n tesis_msc
conda create -n tesis_msc python=3.6
activate tesis_msc
python -m pip install --upgrade pip
pip install tensorflow==1.12 tensorboard
# pip install --ignore-installed --upgrade tensorflow-gpu ####
conda install -c anaconda protobuf
pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python
# pip install pillow
# pip install lxml
# pip install Cython
# pip install contextlib2
# pip install jupyter
# pip install matplotlib
# pip install pandas
# pip install opencv-python

#set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
set PYTHONPATH=C:\Users\jucas\Documents\commerce_pixel\models;C:\Users\jucas\Documents\commerce_pixel\models\research;C:\Users\jucas\Documents\commerce_pixel\models\research\slim 

cd Documents\commerce_pixel\models\research

protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto 
#.\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
python setup.py build
python setup.py install

python
import tensorflow as tf
tf.__version__

cd object_detection
jupyter notebook object_detection_tutorial.ipynb

# python3 generate_tfrecord.py --csv_input=../../../images_resize/augmented1_train_labels.csv --image_dir=../../../images_resize/augmented/ --output_path=../../../images_resize/tf_records/train_aug1.record
# python3 train.py --logtostderr --train_dir=training/train_80_l2_001/ --pipeline_config_path=training/train_80_l2_001/faster_rcnn_inception_v2_pets.config
# python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path training/train_80_l2_001/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/train_80_l2_001/model.ckpt-500000 --output_directory frozen_graphs/inference_graph_train_80_l2_001