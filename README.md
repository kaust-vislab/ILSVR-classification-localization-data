# ILSVR classification and localization dataset

This repo contains a build script for the [Imagenet](http://www.image-net.org/) data set. 
After cloning the repository, in order to use the build script, you will first need to 
obtain an account with Imagenet and then download the following files.

1. ILSVRC2012_devkit_t12.tar.gz
2. ILSVRC2012_img_train.tar
3. ILSVRC2012_img_val.tar
4. ILSVRC2012_img_test_v10102019.tar

File 1 containing the Imagenet development kit should be moved into the `./src` directory 
and then extracted.

```bash
$ tar -xzf ILSVRC2012_devkit_t12.tar.gz
```

Files 2-4 contain the raw image files and should be moved to the `./data` directory. **You do 
not need to extract files 2-4 the extraction process will be handled by the build script.** 

## Building the data set

### Creating and activating the Conda environment

The following commands can be used to create and activate the Conda environment containing the 
necessary Python packages to build the Imagenet data set.

```bash
$ conda env create --prefix ./env --file environment.yml
$ conda activate ./env
```
 
### Raw JPEG images

Running the following commands will extract and re-organize the raw *.JPEG images that comprise 
the Imagenet classification and localization data set. The resulting training, validation, and 
testing images can be found in `./data/jpeg/train`, `./data/jpeg/val`, and `./data/jpeg/test`, 
respectively.

```bash
$ cd ./src
$ python build_classification_localization_data.py
```
