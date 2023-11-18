import streamlit as st
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # reslove the issues related to the openMP lib




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the transforms for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained models from .pth files
vgg16_model = torch.load('../models/vgg_Best_model.pth', map_location=torch.device('cpu'))
mobilenetv2_model = torch.load('../models/mobilenet_Best_model.pth', map_location=torch.device('cpu'))
resnet50_model = torch.load('../models/Resnet_Best_model.pth', map_location=torch.device('cpu'))
Resnet_scratch_model = torch.load('../models/Resnet_noweights_Best_model.pth', map_location=torch.device('cpu'))
Resnet_finetuned_model = torch.load('../models/Resnet_finetune_Best_model.pth', map_location=torch.device('cpu'))

vgg16_model = vgg16_model.to(device)
mobilenetv2_model = mobilenetv2_model.to(device)
resnet50_model = resnet50_model.to(device)
Resnet_scratch_model = Resnet_scratch_model.to(device)
Resnet_finetuned_model = Resnet_finetuned_model.to(device)

# Set the models to evaluation mode
vgg16_model.eval()
mobilenetv2_model.eval()
resnet50_model.eval()
Resnet_scratch_model.eval()
Resnet_finetuned_model.eval()



def display_images(folder_path):
    image_files = os.listdir(folder_path)
    for file_name in image_files:
        image_path = os.path.join(folder_path, file_name)
        st.markdown("<h4 style='text-align: center;'>"+str(file_name.split('.')[0])+"</h4>", unsafe_allow_html=True)
        image = Image.open(image_path)
        new_image = image.resize((700, 500))
        st.image(new_image, use_column_width=True)

def page_images():
    st.title("Training Metrics and Confusion matrix")
    
    # Folder paths
    folder_path1 = '../metrics/Training_stats'
    folder_path2 = '../metrics/confusion_matrix'
    
    # Display images in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Training Graph")
        display_images(folder_path1)
        
    with col2:
        st.header("Confusion matrix")
        display_images(folder_path2)


def page_inference():
    # Hide the GIF image
    gif_container = st.empty()
    gif_container.empty()

    # Add a message asking the user to upload an image
    st.markdown("<h4 style='text-align: center;'>Please upload an image</h4>", unsafe_allow_html=True)

    # Add an uploader widget to the app
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the input image and apply the transforms
        image = Image.open(uploaded_file)
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

        # Make predictions using the three models
        vgg16_prediction = predict(vgg16_model, image_tensor)
        mobilenetv2_prediction = predict(mobilenetv2_model, image_tensor)
        resnet50_prediction = predict(resnet50_model, image_tensor)
        Resnet_scratch_prediction = predict(Resnet_scratch_model, image_tensor)
        Resnet_finetuned_prediction = predict(Resnet_finetuned_model, image_tensor)

        # Display the input image and the predictions

        st.header("Input Image")
        st.image(image, use_column_width=False , width = 500)
        st.markdown("<h3 style='text-align: center;'>Model Predictions</h4>", unsafe_allow_html=True)

        row1 = st.columns(2)
        row2 = st.columns(2)
        row3 = st.columns(2)
        row4 = st.columns(2)
        row5 = st.columns(2)
        row6 = st.columns(2)

        with row1[0]:
            st.markdown("<h4 style='text-align: center; color: green;'>Model Name</h4>", unsafe_allow_html=True)
        with row1[1]:
            st.markdown("<h4 style='text-align: center;color: green;'>Predicted Class</h4>", unsafe_allow_html=True)

        with row2[0]:
            st.markdown("<h5 style='text-align: center;'>"+str('Feature Extracted based Trained Resnet50 Model')+"</h4>", unsafe_allow_html=True)

        with row2[1]:
            st.markdown("<h5 style='text-align: center;'>"+str(resnet50_prediction)+"</h4>", unsafe_allow_html=True)

        with row3[0]:
            st.markdown("<h5 style='text-align: center;'>"+str('Feature Extracted based Trained Mobilenetv2 Model')+"</h4>", unsafe_allow_html=True)

        with row3[1]:
            st.markdown("<h5 style='text-align: center;'>"+str(mobilenetv2_prediction)+"</h4>", unsafe_allow_html=True)

        with row4[0]:
            st.markdown("<h5 style='text-align: center;'>"+str('Feature Extracted based Trained  VGG16 Model')+"</h4>", unsafe_allow_html=True)

        with row4[1]:
            st.markdown("<h5 style='text-align: center;'>"+str(vgg16_prediction)+"</h4>", unsafe_allow_html=True)

        with row5[0]:
            st.markdown("<h5 style='text-align: center;'>"+str('scratch CNN model')+"</h4>", unsafe_allow_html=True)

        with row5[1]:
            st.markdown("<h5 style='text-align: center;'>"+str(Resnet_scratch_prediction)+"</h4>", unsafe_allow_html=True)

        with row6[0]:
            st.markdown("<h5 style='text-align: center;'>"+str('FineTuned Resnet50 Model')+"</h4>", unsafe_allow_html=True)

        with row6[1]:
            st.markdown("<h5 style='text-align: center;'>"+str(Resnet_finetuned_prediction)+"</h4>", unsafe_allow_html=True)

def page_comparison():
    st.title("Models Comparision ")

    # Model files
    model_files_a ={
        'Feature Extracted Resnet50 Model': '../metrics/performance_metrics/Feature_Extracted_Resnet50.md',
        'Feature Extracted VGG16 Model': '../metrics/performance_metrics/Feature_Extracted_vgg16.md',
        'Feature Extracted Mobilenetv2 Model': '../metrics/performance_metrics/Feature_Extracted_Mobilenetv2.md'
    }

    model_files_b = {
        'Finetuned Resnet Model': '../metrics/performance_metrics/Finetuned_Resnet50.md',
        'Feature Extraction Resnet Model': '../metrics/performance_metrics/Feature_Extracted_Resnet50.md',
        'scratch Trained Resnet Model': '../metrics/performance_metrics/Scratch_cnn.md'
    }

    # Radio button to select models
    model_type = st.radio("Select Model Type", ['Transfer Learning (vgg16 vs Resnet50 vs Mobilenetv2)', 'Custom CNN model vs Trasnfer Learning Models'])

    if model_type == 'Transfer Learning (vgg16 vs Resnet50 vs Mobilenetv2)':
        st.markdown("<h4 style='text-align: center; color: green;'>Comparison of the 3 different models shown which were trained using the feature extraction-based Transfer Learning technique.</h4>", unsafe_allow_html=True)
        st.markdown("- **VGG16** \n- **Mobilenetv2** \n- **Resnet**")
        model_files = model_files_a
    else:
        st.markdown("<h4 style='text-align: center; color: green;'>comparision of custom CNN model and Transfer Learning Models</h4>", unsafe_allow_html=True)
        st.markdown("- **Finetuned Model:** pre-trained ResNet model  which  fine-tuned on the Bloodcells dataset.  \n- **Feature Extracted Resnet Model:** A pre-trained  ResNet model , Trained freezing weights and altering the classification layer accoridng to the bloodcells dataset. \n- **Scratch CNN model:** Custom CNN model without any pre-trained weights on bloodcells dataset")
        model_files = model_files_b

    # Metrics to show
    metrics = ['accuracy', 'f1-score', 'recall']
    model_data = {}


    for model_name, file_path in model_files.items():
        table = pd.read_table(file_path, sep='|', skiprows=[1], usecols=[1, 2, 3, 4, 5])
        table.columns = table.columns.str.strip()
        overall_precision = 0
        overall_recall = 0
        overall_f1score = 0
        for i in range(0,4):
            overall_precision += table.loc[i]['precision'] * table.loc[i]['support']
            overall_recall += table.loc[i]['recall'] * table.loc[i]['support']
            overall_f1score += table.loc[i]['f1-score'] * table.loc[i]['support']

        overall_precision = overall_precision/ table.loc[5]['support']
        overall_recall = overall_recall/ table.loc[5]['support']
        overall_f1score = overall_f1score/ table.loc[5]['support']

        model_data[model_name] = {
            'accuracy': table.loc[4]['precision'],
            'precision': overall_precision,
            'recall': overall_recall,
            'f1-score': overall_f1score,
        }

    # Create dataframe and show table
    df = pd.DataFrame(model_data).T
    df.index.name = 'Model Name'
    st.table(df)
    st.markdown("Every Model trained with below parameters")
    st.markdown("- **Epochs:** 100  \n- **Batch Size:** 128 \n- **Optimizer:** Stochastic Gradient Descent \n- **Learning Rate :** 0.001 \n- **Loss :** Cross Entropy" )






def home():
    st.markdown("<h2 style='text-align: center;'>A Study of Transfer Learning</h2>", unsafe_allow_html=True)
    row1 = st.columns(2)
    with row1[0]:
        st.image('./utils/TL.png',width = 500)
    with row1[1]:
        st.image('./utils/TL.gif', use_column_width=True)



def predict(model, image):
    with torch.no_grad():
        dict_classes = {0:"EOSINOPHIL" , 1:"LYMPHOCYTE" , 2:"MONOCYTE" ,3:"NEUTROPHIL"}
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return dict_classes[predicted.item()]

def main():
    st.set_page_config(page_title="Transfer Learning Study",layout="wide")
    # Add CSS to adjust the width of the sidebar
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: 200px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    width: 200px;
    margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    # Define your pages in a dictionary
    pages = {
        "Home":home,
        "Inference": page_inference,
        "Models comparsion": page_comparison,
        "Training Metrics":page_images
    }
    st.sidebar.title("Navigation Menu")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Display the selected page with the help of the dictionary
    page = pages[selection]
    page()


if __name__ == "__main__":
    main()
