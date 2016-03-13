from __future__ import print_function

import os
import numpy as np
import dicom
from scipy.misc import imresize
from segment import calc_rois
import pickle

crop_size = (128, 128)
scale_size = (64, 64)

def crop_resize(images, circles):
    """
    Crop center and resize.

    :param img: image to be cropped and resized.
    """
    crops = []
    for i in xrange(images.shape[0]):
        center = circles[i]
        stack = images[i]
        # we crop image from center
        cen_x = np.round(center[1])
        cen_y = np.round(center[0])
        if cen_x - crop_size[0]/2 < 0:
            x = 0
            xx = crop_size[1]
        elif cen_x + crop_size[0]/2 > stack.shape[1]:
            xx = stack.shape[1]
            x = xx - crop_size[0]
        else:
            x = cen_x - crop_size[0]/2
            xx = x+crop_size[0]

        if cen_y - crop_size[1]/2 < 0:
            y = 0
            yy = crop_size[1]
        elif cen_y + crop_size[1]/2 > stack.shape[2]:
            yy = stack.shape[2]
            y = yy - crop_size[1]
        else:
            y = cen_y - crop_size[1]/2
            yy = y+crop_size[1]

        tmp = []
        cropped = stack[:, x:xx, y:yy]
        for i in xrange(cropped.shape[0]):
            tmp.append(imresize(cropped[i,:,:], scale_size))
        crops.append(np.array(tmp))

    return np.array(crops)


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
    metadata = {}
    slice_locations = []
    for subdir, _, files in os.walk(from_dir):
        subdir = subdir.replace('\\', '/')  # windows path fix
        subdir_split = subdir.split('/')
        study_id = subdir_split[-3]

        pixel_scale = None
        slice_thickness = None
        
        if "sax" in subdir:
            for f in files:
                #print(current_study)
                image_path = os.path.join(subdir, f)
                if not image_path.endswith('.dcm'):
                    continue
                
                image = dicom.read_file(image_path)
                if not pixel_scale:
                    pixel_scale = float(image.PixelSpacing[0])*2.0
                if not slice_thickness:
                    slice_thickness = float(image.SliceThickness)
                image = image.pixel_array.astype(float)

                image /= np.max(image)  # scale to [0,1]

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
                    if current_study != "":
                        print(study_id)
                        try:
                            ids.append(current_study)
                            all_study_images = np.array(current_study_images)
                            centers = calc_rois(all_study_images)
                            study_to_images[current_study] = crop_resize(all_study_images, centers)
                            metadata[current_study] = np.array([pixel_scale, slice_thickness])
                            print('shape for : ', current_study, study_to_images[current_study].shape)
                        except:
                            pass

                    current_study = study_id
                    current_study_images = []

                images.append(image)

                if verbose:
                    if total % 1000 == 0:
                        print('Images processed {0}'.format(total))
                total += 1
    x = 0
    try:
        while len(images) < 30:
            images.append(images[x])
            x += 1
        if len(images) > 30:
            images = images[0:30]
    except IndexError:
        pass
    metadata[study_id] = np.array([pixel_scale, slice_thickness])


    print('-'*50)
    print('All DICOM in {0} images loaded.'.format(from_dir))
    print('-'*50)

    try:
        current_study_images.append(images)
        study_to_images[current_study] = np.array(current_study_images)
        if current_study != "":
            ids.append(current_study)
            all_study_images = np.array(current_study_images)
            centers = calc_rois(all_study_images)
            study_to_images[current_study] = crop_resize(all_study_images, centers)
    except:
        pass
    return ids, study_to_images, metadata



def map_studies_results():
    """
    Maps studies to their respective targets.
    """
    id_to_results = dict()
    train_csv = open('/data/KaggleData/train.csv')
    # train_csv = open('D:/Documents/CS231N/dataset/train.csv') # /data/KaggleData/train.csv
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
    # study_ids, images, all_metadata = load_images('D:/Documents/CS231N/dataset/train')
    study_ids, images, all_metadata = load_images('/data/KaggleData/train')

    studies_to_results = map_studies_results()  # load the dictionary of studies to targets
    X = []
    y = []
    metadata = []
    
    for study_id in study_ids:
        print('Processing id: ', study_id)
        try:
            study = images[study_id]
            study_metadata = all_metadata[study_id]
            outputs = studies_to_results[study_id]
            all_study_images = np.concatenate(study)
            X.append(all_study_images)
            y.append(outputs)
            metadata.append(study_metadata)
        except:
            pass

    X_new = []
    maxDepth = np.max([stack.shape[0] for stack in X])
    for stack in X:
        # Concatenate blank images until all stacks are equal size
        stack = np.concatenate((stack, np.zeros((maxDepth - stack.shape[0], stack.shape[1], stack.shape[2]))))
        X_new.append(stack)

    X = np.array(X_new, dtype=np.uint8)
    y = np.array(y)
    study_metadata = np.array(metadata, dtype=np.float64)
    np.save('/data/preprocessed/X_train.npy', X)
    np.save('/data/preprocessed/y_train.npy', y)
    np.save('/data/preprocessed/metadata_train.npy', study_metadata)
    print('Done.')


def write_validation_npy():
    """
    Loads the validation data set including X and study ids and saves it to .npy file.
    """
    print('-'*50)
    print('Writing validation data to .npy file...')
    print('-'*50)

    # ids, images, all_metadata = load_images('D:/Documents/CS231N/dataset/validate')
    ids, images, all_metadata = load_images('/data/KaggleData/validate') # /data/KaggleData/validate
    study_ids = []
    X = []
    metadata = []

    for study_id in ids:
        try:
            study = images[study_id]
            study_metadata = all_metadata[study_id]
            all_study_images = np.concatenate(study)
            X.append(all_study_images)
            metadata.append(study_metadata)
        except:
            pass

    X_new = []
    maxDepth = np.max([stack.shape[0] for stack in X])
    for stack in X:
        # Concatenate blank images until all stacks are equal size
        stack = np.concatenate((stack, np.zeros((maxDepth - stack.shape[0], stack.shape[1], stack.shape[2]))))
        X_new.append(stack)

    X = np.array(X_new, dtype=np.uint8)
    study_metadata = np.array(study_metadata, dtype=np.float64)
    np.save('data/tmp/X_validate.npy', X)
    np.save('data/tmp/ids_validate.npy', study_ids)
    np.save('data/tmp/metadata_val.npy', study_metadata)
    print('Done.')


write_train_npy()
write_validation_npy()
