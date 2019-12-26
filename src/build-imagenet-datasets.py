"""
Build the ImageNet 2012 Challenge training, validation, and testing data set with synset.


After extractiong, the raw ImageNet validation data set resides in JPEG files
located in the following directory structure.

    DATA_DIR/jpeg/val/ILSVRC2012_val_00000001.JPEG
    DATA_DIR/jpeg/val/ILSVRC2012_val_00000002.JPEG
    ...
    DATA_DIR/jpeg/val/ILSVRC2012_val_00050000.JPEG

This script moves the files into a directory structure like such:

    DATA_DIR/jpeg/val/n01440764/ILSVRC2012_val_00000293.JPEG
    DATA_DIR/jpeg/val/n01440764/ILSVRC2012_val_00000543.JPEG
    ...

where 'n01440764' is the unique synset label associated with these images.

This directory reorganization requires a mapping from validation image
ground truth label number to the associated synset label. This mapping is
created using the following files included in the ImageNet development kit.

- "./ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
- "./ILSVRC2012_devkit_t12/data/meta.mat"

Sample usage:

Default behavior is to build all of the train, val, and test datasets.

    $ python build-imagenet-datasets.py

You can also choose to build only some of the datasets.

    $ python build-imagenet-datasets.py --no-val --no-test  # only builds train
    $ python build-imagenet-datasets.py --no-train --no-test  # only build val

"""
import argparse
import glob
import os
import tarfile

import scipy.io as sio


DATA_DIR = "../data"
TESTING_TAR = f"{DATA_DIR}/ILSVRC2012_img_test_v10102019.tar"
TESTING_TAR_MD5 = "13fbf1da088b1f7c80a789d452818daa"

TRAINING_TAR = f"{DATA_DIR}/ILSVRC2012_img_train.tar"
TRAINING_TAR_MD5 = "MD5: 1d675b47d978889d74fa0da5fadfb00e"
 
VALIDATION_TAR = f"{DATA_DIR}/ILSVRC2012_img_val.tar"
VALIDATION_TAR_MD5 = "29b22e2961454d5413ddabcf34fc5622"


def extract_testing_images(prefix):
    with tarfile.open(TESTING_TAR) as tf:
        tf.extractall(path=prefix)
    n_test_images = len(glob.glob(f"{prefix}/test/*.JPEG"))
    assert n_test_images == 100000
    print("Successfully extracted testing images!")


def extract_training_images(prefix):
    with tarfile.open(TRAINING_TAR) as tf:
        for member in tf:
            synset_label, _ = member.name.split('.')
            synset_path = f"{prefix}/{synset_label}"
            os.mkdir(synset_path)
            tf.extract(member, path=synset_path)
            with tarfile.open(f"{synset_path}/{member.name}") as class_tf:
                class_tf.extractall(path=synset_path)
            os.remove(f"{synset_path}/{member.name}")
    n_training_images = len(glob.glob(f"{prefix}/n*/*.JPEG"))
    assert n_training_images == 1281167
    print("Successfully extracted training images!")


def extract_validation_images(prefix):
    with tarfile.open(VALIDATION_TAR) as tf:
        tf.extractall(path=prefix)
    n_validation_images = len(glob.glob(f"{prefix}/*"))
    assert n_validation_images == 50000
    print("Successfully extracted validation images!")


parser = argparse.ArgumentParser()
parser.add_argument("--train", default=True, action="store_true")
parser.add_argument("--no-train", dest="train", action="store_false")
parser.add_argument("--val", default=True, action="store_true")
parser.add_argument("--no-val", dest="val", action="store_false")
parser.add_argument("--test", default=True, action="store_true")
parser.add_argument("--no-test", dest="test", action="store_false")
args = parser.parse_args()

if args.train:
    _prefix = f"{DATA_DIR}/jpeg/train"
    if not os.path.isdir(_prefix):
        os.makedirs(_prefix)
    extract_training_images(_prefix)

if args.val:
    _prefix = f"{DATA_DIR}/jpeg/val"
    if not os.path.isdir(_prefix):
        os.makedirs(_prefix)
    extract_validation_images(_prefix)

    # process the matlab files to create the mapping from ground truth labels to synset labels    
    with open("./ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt") as f:
        ground_truth_labels = [int(label.strip()) for label in f.readlines()]

    meta = sio.loadmat("./ILSVRC2012_devkit_t12/data/meta.mat")
    ground_truth_labels_to_synset_labels = {k: v for (((k,),), (v,), *_), in meta["synsets"]}
    synset_labels = [ground_truth_labels_to_synset_labels[n] for n in ground_truth_labels]

    # re-organize the validation images directory to match the structure of the training images directory
    prefix = f"{DATA_DIR}/jpeg/val"
    src_image_paths = sorted(glob.glob(f"{prefix}/*.JPEG"))
    assert len(synset_labels) == len(src_image_paths)

    for synset_label, src_image_path in zip(synset_labels, src_image_paths):
        synset_path = f"{prefix}/{synset_label}"
        if not os.path.isdir(synset_path):
            os.makedirs(synset_path)
        validation_image = os.path.basename(src_image_path)
        dst_image_path = f"{synset_path}/{validation_image}"
        os.rename(src_image_path, dst_image_path)
    assert(len(glob.glob(f"{prefix}/*.JPEG")) == 0)

if args.test:
    _prefix = f"{DATA_DIR}/jpeg"  # the TESTING_TAR includes a test/ dir already
    if not os.path.isdir(_prefix):
        os.makedirs(_prefix)
    extract_testing_images(_prefix)
