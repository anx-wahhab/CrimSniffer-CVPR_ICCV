# Facial Sketch to Image Generation using Coordinate-preserving Gated Fusion

This project presents a novel approach for generating realistic images from facial sketches using a Coordinate-preserving Gated Fusion technique. The text file contains guide require to: Pre-Processing, Training, and Inference.

## Requirements 
   - Make sure you are using the python version 3.8 or above.
   -Install all the requirements using the following command:	
		pip install -r requirements.txt 
    


## Pre-Processing

1. **Data Collection**: 
   - Gather a dataset containing facial images from either Celeb-A-HQ Dataset, or Chuk face sketch database. "https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html", "https://www.kaggle.com/datasets/arbazkhan971/cuhk-face-sketch-database-cufs".

2. **Face Extraction**: 
   - In order to extract faces from images you can use pre-trained HAAR cascade
classifier (Viola and Jones 2001), which is commonly used for face recognition. The code for it is given inside the dataset directory named as face_extractor.py

3. **Apply Photoshop Photocopy filter**:
   - Apply the photoshop photocopy filter to all the faces extracted from the images.

4. **Apply Sketch Simplification**:
   - Apply the sketch simplification to all the photoshop photocopy filtered images. The code for the sketch simplification is inherited from the 
"https://github.com/Xu-Justin/sketch_simplification/tree/b65de8130dd6352d7f1baa10bb39888903b8bcbf" and placed inside the dataset folder. However, you need to download the weights from the following drive link "https://drive.google.com/file/d/1-16NelGGRQBOBa42OFpgu3QX4y4rSs0q/view".

NOTE: While preparing the dataset place the weight file model_gan.pth to the root directory of the sketch_simplification directory, on the other hand while using it during inference stage place it inside the sketch_simplification directory.

5. **Data Structure**:
   - Place the dataset in the following format, the names of the folder inside the root folders must be same(i.e. photos, sketches).
-i.e. cleb-a
	|____train
	|	|__photos
	|	|__sketches
	| 
	|____val(optional)
		|__photos
		|__sketches



## Training

As discussed in main paper, our architecture consists of two stage training. 

1. **For training stage 1**:

python train_stage_1.py --dataset Dataset/your-data/train/ --dataset_validation Dataset/your-data/val/ --batch_size 2  --epochs 100 --output weight/weight/path-to save-your-weight-files-during-training/ --device cuda 

   - To see more detailed explanation of arguments open the training_stage_1.py 

2. ** After first stage training is done use the following command for training stage 2**:
   
python train_stage_2.py --dataset Dataset/your-data/train/ --dataset_validation Dataset/your-data/val/ --batch_size 2  --epochs 100 --resume_CE weight/weight/path-to-load-your-weight-files-of-training-stage-1/CE/ --output weight/weight/path-to save-your-weight-files-during-training/ --device cuda

   - To see more detailed explanation of arguments open the training_stage_2.py 




## Inference

1. **GFPGAN**:
   - For the inference stage, first of all clone the gfpgan repository from there GitHub using the following command below,
	git clone https://github.com/TencentARC/GFPGAN.git
   - Inside the GFPGAN repository, run the following command to download all the required packages.
	pip install -r requirements.txt 
   - Download their weight files for the V1.4 model and place it inside the GFPGAN\experiments\pretrained_models directory.

2. **Set Python path to avoid conflicts or errors**:
   - After setting up the gfpgan, run the following command before running any of the inference command
		set PYTHONPATH=%PYTHONPATH%;./GFPGAN

3. **Folder Image Inference**:
   - The following commands will inference all images in the folder input folder and then saved the result to output folder.
   
python combined_inference.py --crimsniffer_weight weight/weight/path-to-your-trainied-weights-folder/ --gfpgan_weight GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth --input path-to-input-folder/--output path-to-output-folder/ --upscale 2 --bg_upsampler realesrgan  --suffix enhanced  --ext jpg --weight 0.5 --device cuda --upscale 2 --bg_upsampler realesrgan

  - To see more detailed explanation of arguments open the combined_inference.py file
