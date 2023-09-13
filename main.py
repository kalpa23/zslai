import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from sklearn.model_selection import train_test_split
import numpy as np

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



class model:
    def generator(self,image_files,destination_folder,source_folder):
        datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Rescale pixel values to [0, 1]
        horizontal_flip=True,
        vertical_flip=True
        )
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # List image files in the source folder
        # Set the desired width and height for resizing
        width = height = 256
        desired_size = (width, height) 

        # Loop through the image files, resize, and save them in the destination folder
        i = 0
        for image_file in image_files:
            source_path = os.path.join(source_folder, image_file)
            image = Image.open(source_path)
            image = image.resize(desired_size, Image.ANTIALIAS)  # Resize using antialiasing for better quality
            
            # Save the resized image in the destination folder
            image_array = img_to_array(image)

            # Reshape for batch processing (batch size = 1)
            image_array = image_array.reshape((1,) + image_array.shape)

            
            for batch in datagen.flow(image_array, batch_size=1):
                augmented_image = batch[0]  # Extract the augmented image
                augmented_image_path = os.path.join(destination_folder, f'augmented_{i}.jpg')
                
                # Convert the augmented image back to a valid image format and save it
                augmented_image = (augmented_image * 255).astype('uint8') 
                augmented_image = array_to_img(augmented_image) 
                augmented_image.save(augmented_image_path)
                i += 1
                if i >= 1000:
                    break

    def load_vgg16():
        vgg16 = models.vgg16(pretrained=True)
        return vgg16
    
    def feature_extraction_through_vgg16(self,features_output_folder,training_data_folder):
        # Create the features output folder if it doesn't exist
        os.makedirs(features_output_folder, exist_ok=True)
        # Transformation to preprocess images
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

        # Iterate through the images in the folder
        for filename in os.listdir(training_data_folder):
            image_path = os.path.join(training_data_folder, filename)
            image = Image.open(image_path)
            image = transform(image)
            image = Variable(image.unsqueeze(0))
            features = vgg16(image)
            features = features.squeeze().detach().numpy()
            feature_output_path = os.path.join(features_output_folder, f"{os.path.splitext(filename)[0]}.npy")
            np.save(feature_output_path, features)

    def load_extracted_feature(self,features_folder):
        feature_files = [os.path.join(features_folder, file) for file in os.listdir(features_folder)]
        features = []
        for file in feature_files:
            if file.endswith(".npy"):
                feature = np.load(file)
                features.append(feature)
        features = np.array(features)
        return features
    
    def assigning_labels():
        labels = [1] * 1008  + [4] * 200
        labels = np.array(labels)
        return labels
    def logistic_reg(self,features, labels):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        accuracy = classifier.score(X_test, y_test)
        print("Accuracy:", accuracy)
        return accuracy



if __name__ == "__main__":
    destination_folder = r'C:\Users\user\Desktop\cv models\bottle\train\generated_resized'
    source_folder = r'C:\Users\user\Desktop\cv models\bottle\train\good'
    training_data_folder = r"C:\Users\user\Desktop\cv models\bottle\train\generated_resized"
    features_output_folder = r"C:\Users\user\Desktop\cv models\bottle\output"
    test_folder = r"C:\Users\user\Desktop\cv models\bottle\test"
    image_files = [f for f in os.listdir(source_folder) if f.endswith(('jpg', 'jpeg', 'png'))]

    model.generator(image_files,destination_folder,source_folder)
    vgg16 = model.load_vgg16()
    model.feature_extraction_through_vgg16(features_output_folder,training_data_folder)
    features = model.load_extracted_feature(features_output_folder)

    labels = model.assigning_labels()

    # applying Logistic regressor classifier function
    accuracy = model.logistic_reg(features,labels)




