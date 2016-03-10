from __future__ import print_function

import os
import numpy as np
import dicom
from scipy.misc import imresize
from segment import calc_rois

img_resize = True
img_shape = (64, 64)

def crop_resize(images, circles):
    """
    Crop center and resize.

    :param img: image to be cropped and resized.
    """
    crops = []
    for i in xrange(images.shape[0]):
        center = circles[i][1]
        stack = images[i]
        # we crop image from center
        cen_x = np.round(center[0])
        cen_y = np.round(center[1])
        if cen_x - crop_size[0]/2 < 0:
            x = 0
            xx = crop_size[1]
        else:
            x = cen_x - crop_size[0]/2
            xx = x+crop_size[0]
        if cen_y - crop_size[1]/2 < 0:
            y = 0
            yy = crop_size[1]
        else:
            y = cen_x - crop_size[1]/2
            yy = y+crop_size[1]
            
        crop_img = padded[:, x:xx, y:yy]
        crops.append(crop_img)
    
    return np.array(crops)

# def crop_resize(img):
#     """
#     Crop center and resize.

#     :param img: image to be cropped and resized.
#     """
#     if img.shape[0] < img.shape[1]:
#         img = img.T
#     # we crop image from center
#     short_edge = min(img.shape[:2])
#     yy = int((img.shape[0] - short_edge) / 2)
#     xx = int((img.shape[1] - short_edge) / 2)
#     crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
#     img = crop_img
#     img = imresize(img, img_shape)
#     return img


def load_images(from_dir, verbose=True):
    """
    Load images in the form study x slices x width x height.
    Each image contains 30 time series frames so that it is ready for the convolutional network.

    :param from_dir: directory with images (train or validate)
    :param verbose: if true then print data
    """
    print('-'*50)
    print('Loading all DICOM images from {0}...'.format(from_dir))
    print('-'*50)

    current_study_sub = ''  # saves the current study sub_folder
    current_study = ''  # saves the current study folder
    current_study_images = []  # holds current study images
    ids = []  # keeps the ids of the studies
    study_to_images = dict()  # dictionary for studies to images
    total = 0
    images = []  # saves 30-frame-images
    from_dir = from_dir if from_dir.endswith('/') else from_dir + '/'

    spacings = []
    last_study = None
    for subdir, _, files in os.walk(from_dir):
        subdir = subdir.replace('\\', '/')  # windows path fix
        subdir_split = subdir.split('/')
        study_id = subdir_split[-3]

        pixel_scale = None
        if "sax" in subdir:
            for f in files:
                image_path = os.path.join(subdir, f)
                if not image_path.endswith('.dcm'):
                    continue

                image = dicom.read_file(image_path)
                image = image.pixel_array.astype(float)
                image /= np.max(image)  # scale to [0,1]

                if pixel_scale == None:
                    pixel_scale = image.PixelSpacing
                    spacings.append(pixel_scale)
                # if img_resize:
                #     image = crop_resize(image)

                if current_study_sub != subdir:
                    x = 0
                    try:
                        while len(images) < 30:
                            images.append(images[x])
                            x += 1
                        if len(images) > 30:
                            images = images[0:30]

                    except IndexError:
                        pass
                    current_study_sub = subdir
                    current_study_images.append(images)
                    images = []

                if current_study != study_id:
                    study_to_images[current_study] = np.array(current_study_images)
                    if current_study != "":
                        ids.append(current_study)
                    last_study = current_study
                    current_study = study_id
                    current_study_images = []
                images.append(image)
                if verbose:
                    if total % 1000 == 0:
                        print('Images processed {0}'.format(total))
                total += 1

            all_study_images = study_to_images[last_study]
            rois, circles = calc_rois(all_study_images)
            study_to_images[last_study] = crop_resize(all_study_images, circles)

    x = 0
    try:
        while len(images) < 30:
            images.append(images[x])
            x += 1
        if len(images) > 30:
            images = images[0:30]
    except IndexError:
        pass

    print('-'*50)
    print('All DICOM in {0} images loaded.'.format(from_dir))
    print('-'*50)

    current_study_images.append(images)
    study_to_images[current_study] = np.array(current_study_images)
    if current_study != "":
        ids.append(current_study)

    all_study_images = study_to_images[last_study]
    rois, circles = calc_rois(all_study_images)
    study_to_images[last_study] = crop_resize(all_study_images, circles)

    return ids, study_to_images


def map_studies_results():
    """
    Maps studies to their respective targets.
    """
    id_to_results = dict()
    train_csv = open('D:/Documents/CS231N/dataset/train.csv') # /data/KaggleData/train.csv
    lines = train_csv.readlines()
    i = 0
    for item in lines:
        if i == 0:
            i = 1
            continue
        id, diastole, systole = item.replace('\n', '').split(',')
        id_to_results[id] = [float(diastole), float(systole)]

    return id_to_results


def write_train_npy():
    """
    Loads the training data set including X and y and saves it to .npy file.
    """
    print('-'*50)
    print('Writing training data to .npy file...')
    print('-'*50)

    study_ids, images = load_images('D:/Documents/CS231N/dataset/train')  # /data/KaggleData/train # load images and their ids
    studies_to_results = map_studies_results()  # load the dictionary of studies to targets
    X = []
    y = []

    for study_id in study_ids:
        study = images[study_id]
        outputs = studies_to_results[study_id]
        all_study_images = np.concatentate(study)
        X.append(all_study_images)
        y.append(outputs)
        # for i in range(study.shape[0]):
        #     X.append(study[i, :, :, :])
        #     y.append(outputs)

    X = np.array(X, dtype=np.uint8)
    y = np.array(y)
    np.save('E:data/X_train.npy', X)
    np.save('E:data/y_train.npy', y)
    print('Done.')


def write_validation_npy():
    """
    Loads the validation data set including X and study ids and saves it to .npy file.
    """
    print('-'*50)
    print('Writing validation data to .npy file...')
    print('-'*50)

    ids, images = load_images('D:/Documents/CS231N/dataset/validate') # /data/KaggleData/validate
    study_ids = []
    X = []

    for study_id in ids:
        study = images[study_id]
        all_study_images = np.concatentate(study)
        X.append(all_study_images)
        # for i in range(study.shape[0]):
        #     study_ids.append(study_id)
        #     X.append(study[i, :, :, :])

    X = np.array(X, dtype=np.uint8)
    np.save('E:data/X_validate.npy', X)
    np.save('E:data/ids_validate.npy', study_ids)
    print('Done.')


write_train_npy()
write_validation_npy()
