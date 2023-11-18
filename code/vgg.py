import torch
import csv
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


batch_size = 128
num_epochs=100
models_path = "../models/vgg_Best_model.pth"
data_dir = "../data/image_files"
file_name = "../metrics/Training_stats_csvs/vgg_Training_metric.csv"
report_file = '../metrics/performance_metrics/vgg_report.md'
confusion_file = '../metrics/confusion_matrix/vgg_confusion.png'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def Save_metrics(metrics , file_name):
    train_losses, test_losses, train_accuracy, test_accuracy = metrics
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['train_loss', 'val_loss', 'train_acc', 'val_acc'])
        for i in range(len(train_losses)):
            writer.writerow([train_losses[i], test_losses[i], train_accuracy[i].tolist(), test_accuracy[i].tolist()])

def imshow(image_tensor, title=None, figure_name=None):
    # Matplotlib based image display function
    fig = plt.figure(figure_name)
    image_tensor = image_tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_tensor = std * image_tensor + mean
    image_tensor = np.clip(image_tensor, 0, 1)
    plt.imshow(image_tensor)
    if title is not None:
        plt.title(title)
    plt.pause(0.00001)  # pause a bit so that plots are updated
    plt.show(block=False) 
    plt.draw()
    plt.pause(5)
    plt.close('all')


def samples_visualization(datasets):
    
    train_dataset, train_loader,val_dataset, val_loader,test_dataset , test_loader = datasets
    # Extracting a Batch data from Training set
    inputs1, classes1 = next(iter(train_loader))
    # Make a grid from batch accordingly
    out1 = torchvision.utils.make_grid(inputs1)
    class_names1 = train_dataset.classes
    imshow(out1, title=[class_names1[x] for x in classes1] , figure_name="Training Dataset" )


    # Extracting a Batch data from val set
    inputs2, classes2 = next(iter(val_loader))
    # Make a grid from batch accordingly
    out2 = torchvision.utils.make_grid(inputs2)
    class_names2 = val_dataset.classes
    imshow(out2, title=[class_names2[x] for x in classes2] , figure_name="validation Dataset")

    # Extracting a Batch data from Test set
    inputs2, classes2 = next(iter(test_loader))
    # Make a grid from batch accordingly
    out2 = torchvision.utils.make_grid(inputs2)
    class_names2 = test_dataset.classes
    imshow(out2, title=[class_names2[x] for x in classes2] , figure_name="Test Dataset")

def create_data_loaders(data_dir ,batch_size):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(f'{data_dir}/train', transform=train_transforms)
    val_dataset = datasets.ImageFolder(f'{data_dir}/val', transform=test_transforms)
    test_dataset = datasets.ImageFolder(f'{data_dir}/test', transform=test_transforms)
    
    #creating data loders for the model
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

    return [train_dataset, train_loader,val_dataset, val_loader,test_dataset , test_loader]

def train_model(datasets, device , model, criterion, optimizer, scheduler, num_epochs,models_path):
    train_dataset, train_loader,val_dataset, val_loader,test_dataset , test_loader = datasets
    best_model_wts = copy.deepcopy(model)
    best_acc = 0.0
    best_loss = float('inf')

    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        # Training phase
        model.train()
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()  #set's the grad to zero 

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

            if i % 10 == 9:    # print every 10 mini-batches
                mini_batch_loss = running_loss / ((i+1) * train_loader.batch_size)
                print('[Epoch: %d, Batch: %d] Loss: %.4f' %
                    (epoch + 1, i + 1, mini_batch_loss))

        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        epoch_acc = running_corrects.double() / len(train_dataset)
        train_accuracy.append(epoch_acc)

        print('Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # Testing phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(val_dataset)
            test_losses.append(epoch_loss)
            epoch_acc = running_corrects.double() / len(val_dataset)
            test_accuracy.append(epoch_acc)

            print('Test Loss: {:.4f} Test Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # Save the model with the best accuracy and lowest loss
            if epoch_acc > best_acc or (epoch_acc == best_acc and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model)
                torch.save(best_model_wts, models_path)

        print('---epoch completed----',epoch+1)

        scheduler.step()

    print('Best val Acc: {:.4f}'.format(best_acc))
    print('Best val Loss: {:.4f}'.format(best_loss))

    return model, [train_losses, test_losses, train_accuracy, test_accuracy]


def evaluate_model(model, test_loader, device, report_file, confusion_file):
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true += labels.cpu().tolist()
            y_pred += preds.cpu().tolist()

    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True ,target_names=test_loader.dataset.classes)
    print(report)
    report_df = pd.DataFrame(report).transpose()

    # Save classification report to file
    with open(report_file, 'w') as f:
        f.write(report_df.to_markdown())

    # Generate confusion matrix
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    classes = test_loader.dataset.classes

    # Save confusion matrix to file
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(confusion, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    plt.savefig(confusion_file)

    return report, confusion



# --------------------Datasets intialization--------------------
datasets = create_data_loaders(data_dir,batch_size)
# visualizing few sample images of the dataset to confirm loading process
samples_visualization(datasets)


# --------------------Pre-Trained Model intialization--------------------
# Load the pre-trained vgg16 model
model = models.vgg16(pretrained=True)

# Freeze all the layers except the last one
for param in model.parameters():
    param.requires_grad = False

num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 4)


# --------------------Training Parameters intialization--------------------
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier[6].parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# Train the model
model = model.to(device)


# --------------------Model Training--------------------
trained_model, metrics = train_model(datasets, device , model, criterion, optimizer, exp_lr_scheduler,num_epochs,models_path)


# Evluating the model on the Testset which saved for metrics
# evaluate_model(trained_model, datasets[-1], device)
evaluate_model(model, datasets[-1], device, report_file, confusion_file)



# saving Metrics into csv format for Plotting
Save_metrics(metrics,file_name)



