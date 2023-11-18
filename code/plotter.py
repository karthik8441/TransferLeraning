import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the path to the folder containing the CSV files
folder_path = '../metrics/Training_stats_csvs'

# Loop through each file in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a CSV file
    if file_name.endswith('.csv'):
        # Load the CSV file into a pandas dataframe
        df = pd.read_csv(os.path.join(folder_path, file_name))

        # Extract the data from the dataframe
        train_losses = df['train_loss']
        train_accs = df['train_acc']
        val_losses = df['val_loss']
        val_accs = df['val_acc']
        epochs = df.index + 1

        # Create a figure with subplots for loss and accuracy
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot the training and validation losses
        axs[0].plot(epochs, train_losses, label='Training Loss')
        axs[0].plot(epochs, val_losses, label='Validation Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # Plot the training and validation accuracies
        axs[1].plot(epochs, train_accs, label='Training Accuracy')
        axs[1].plot(epochs, val_accs, label='Validation Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        # Save the figure to an image file with the same name as the CSV file to png
       
        # Close the figure to free up memory
        plt.close(fig)
