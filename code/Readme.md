# Transfer Learning

Folder contains all the necessary Training scripts , Inference scripts , analysis scripts and also data required for Project Transfer Learning.

- Labels used for development:
> NEUTROPHIL, MONOCYTE, LYMPHOCYTE, EOSINOPHIL

## Requirements / required libraires
- python >= 3.9.12 
- Jupyter Notebook (to run the Blood cell analysis ipython notebook)
- imutils
- torch 
- torchvision
- matplotlib
- pandas
- streamlit

==> we can install this libraires with below command
> python3 -m pip install "library name"
#### example: 
>Python3 -m pip install imutils

## Folder Structure
- data
    > Dataset contains  Eosinophil, Lymphocyte, Monocyte, Neutrophil cell images 
   ####  Folder name : "Combined"
        > contains all blood cell images seggreagated by the label.
   ####  Folder name : "Image_files"
        > contains all blood cell images seggreagated by the label and splitted for Train , Test ,validation splits.


## code 

- blood_cell_analysis.ipynb
    > This ipython notebook contains all the Data analysis done for the Dataset.

    - Run 
        > you can run this ipython notebook using jupyter notebook.

- data_split.py
    > Script to split the combined data to Train and Validation sets and save them into a folder.
    - Run
        > you can run this code by "Python3 data_split.py" 
        * please specify the all the paths correctly in the code.

- Resnet_train.py
    > script which reads , loads the data and create's and finetune the pretrained resnet50 model.
    >here model is retrained with custom classification layer by freezing all the pre-trained weights
    - Run
        > you can run this code by "Python3 Resnet_train.py" 
        * please specify the all the paths correctly in the code.
        * This code saves the results statistics into a csv file.

- vgg.py
    > script which reads , loads the data and create's and finetune the pretrained vgg16 model.
    >here model is retrained with custom classification layer by freezing all the pre-trained weights
    - Run
        > you can run this code by "Python3 vgg.py" 
        * please specify the all the paths correctly in the code.
        * This code saves the results statistics into a csv file.

- mobilenet.py
    > script which reads , loads the data and create's and finetune the pretrained mobilenetv2 model.
    >here model is retrained with custom classification layer by freezing all the pre-trained weights
    - Run
        > you can run this code by "Python3 mobilenet.py" 
        * please specify the all the paths correctly in the code.
        * This code saves the results statistics into a csv file.


- Resnet_train_finetuning.py
    > script which reads , loads the data and create's and finetune the pretrained resnet50 model with all the weights.
    >here entire model is retrained by keeping low learning rate on bloodcells dataset.
    - Run
        > you can run this code by "Python3 Resnet_train_finetuning.py" 
        * please specify the all the paths correctly in the code.
        * This code saves the results statistics into a csv file.

- Resnet_train_noweights.py
    > script which reads , loads the data and create's Resnet Architecture model and train it without any pre-trained weights.
    >here we take the reference of the resnet Architecture model,and trained entire model from scracth
    - Run
        > you can run this code by "Resnet_train_noweights.py" 
        * please specify the all the paths correctly in the code.
        * This code saves the results statistics into a csv file.


- inference_streamlit.py
    > uI code for inference , models comparision and Training Metrics visualizaztion
    > Before run this file we need to excute the all the models file so the data required is gererated and saved in metrics folder.
    - Run
        > you can run this code by "streamlit run inference_streamlit.py" 
        * please specify the all the paths correctly in the code.
        * This code opens the UI in the browser.

- plotter.py
    > Script to plot the Training  and  validation statistics to image file.
    - Run
        > you can run this code by "Python3 plotter.py" 
        * please specify the all the paths correctly in the code.

## CODE RUNNING FLOW

1. first make sure all the images and code have been unzipped perfectly.
2. Next run the "data_split.py"  code after providing/ changing all the necessary paths.
3. Next run the "Resnet_train.py" which start training the pre-trained model with the dataset provided , please specify all paths correctly.
# follow the same approach for all the models 
4. Next run the "plotter.py" script which plots the results into separate image file and showcase that.
5. once the Model is completely Trained you can run the 'inference_streamlit.py' for visualization of metrics , comnaprision and inference.



