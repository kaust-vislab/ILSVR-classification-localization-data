# ILSVR classification and localization dataset

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
