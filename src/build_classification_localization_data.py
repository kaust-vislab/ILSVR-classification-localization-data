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

    $ python build_classification_localization_data.py

"""
import glob
import os,sys
import tarfile

import scipy.io as sio
import argparse 

DATA_DIR = "../data"
TESTING_TAR = f"{DATA_DIR}/ILSVRC2012_img_test_v10102019.tar"
TESTING_TAR_MD5 = "13fbf1da088b1f7c80a789d452818daa"

TRAINING_TAR = f"{DATA_DIR}/ILSVRC2012_img_train.tar"
TRAINING_TAR_MD5 = "MD5: 1d675b47d978889d74fa0da5fadfb00e"
 
VALIDATION_TAR = f"{DATA_DIR}/ILSVRC2012_img_val.tar"
VALIDATION_TAR_MD5 = "29b22e2961454d5413ddabcf34fc5622"


def read_args ():
    parser = argparse.ArgumentParser(prog='build_classification_localization_data.py ',description='Builds directory tree for the Imagenet 2012 raw data into test, train and validation data.')
    parser.add_argument('--test',required=False,action='store_true',help='Extract test data and create subdirectory.\n')
    parser.add_argument('--train',required=False,action='store_true',help='Extract training data and create subdirectory.\n')
    parser.add_argument('--val',required=False,action='store_true',help='Extract validation data and create subdirectory.\n')
    parser.add_argument('--mapping',required=False,action='store_true',help='Creates mapping understandable by the DL frameworks.\n')
    parser.add_argument('--dryrun',required=False,action='store_true',help='Dry run.\n')
    return parser.parse_args()

def extract_testing_images(prefix=f"{DATA_DIR}/jpeg"):
    # the TESTING_TAR includes a test/ dir already
    if not os.path.isdir(prefix):
        os.makedirs(prefix,exist_ok=True)
    with tarfile.open(TESTING_TAR) as tf:
        tf.extractall(path=prefix)
    n_test_images = len(glob.glob(f"{prefix}/test/*.JPEG"))
    assert n_test_images == 100000
    print("Successfully extracted testing images!")


def extract_training_images(prefix=f"{DATA_DIR}/jpeg/train"):
    if not os.path.isdir(prefix):
        os.makedirs(prefix,exist_ok=True)
    with tarfile.open(TRAINING_TAR) as tf:
        for member in tf:
            synset_label, _ = member.name.split('.')
            synset_path = f"{prefix}/{synset_label}"
            os.makedirs(synset_path,exist_ok=True)
            tf.extract(member, path=synset_path)
            with tarfile.open(f"{synset_path}/{member.name}") as class_tf:
                class_tf.extractall(path=synset_path)
            os.remove(f"{synset_path}/{member.name}")
    n_training_images = len(glob.glob(f"{prefix}/n*/*.JPEG"))
    assert n_training_images == 1281167
    print("Successfully extracted training images!")


def extract_validation_images(prefix=f"{DATA_DIR}/jpeg/val"):
    if not os.path.isdir(prefix):
        os.makedirs(prefix,exist_ok=True)
    with tarfile.open(VALIDATION_TAR) as tf:
        tf.extractall(path=prefix)
    n_validation_images = len(glob.glob(f"{prefix}/*"))
    assert n_validation_images == 50000
    print("Successfully extracted validation images!")

def mapping():
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
        print("Successfully built the Imagenet classification and localization data set!")


if __name__ == "__main__":
    args = read_args()
  
    if (len(sys.argv) > 1):
        if args.test:
            extract_testing_images()
            
        if args.train:
            extract_training_images()
        
        if args.val:
            extract_validation_images()
    
        if args.mapping:
            mapping()
        if args.dryrun:
            print("its a dry_run")
            
    else:
        extract_testing_images()
        extract_training_images()
        extract_validation_images()
        mapping()
     
