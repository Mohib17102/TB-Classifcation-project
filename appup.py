import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

# Define class names (adjust according to your dataset)
class_names = ['COVID', 'Extrapulmonary TB', 'Miliary TB', 'Normal', 'Pneumonia']

# Function to load the model, preprocess the image, and make predictions
def predict_image_class(model_path, img, target_size=(224, 224)):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)  # Adjust for other models if needed

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0], prediction

# Streamlit app layout
def main():
    st.title("TB Image Classification App")
    
    # Model selection
    model_option = st.selectbox("Select a Model", ['VGG16', 'VGG19','ResNEt50','ResNet101','DenseNet201','InceptionV3'])  # Updated to only include VGG models
    
    # Set model file paths (You need to adjust the paths to where your models are stored)
    model_paths = {
        'VGG16': 'VGG16_best_model.keras',
        'VGG19': 'VGG19_best_model.keras',
        'ResNEt50': 'ResNet50_best1_model.keras',
        'ResNet101': 'ResNet101_best1_model.keras',
        'DenseNet201': 'DenseNet201_best1_model.keras',
        'InceptionV3': 'InceptionV3_best1_model.keras',
        
        

    }
    
    model_path = model_paths[model_option]
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("Predict"):
            # Resize image to the model input size
            img_resized = img.resize((224, 224))
            
            # Make predictions
            predicted_class, prediction = predict_image_class(model_path, img_resized)
            
            # Display the result
            st.write(f"Predicted Class: {class_names[predicted_class]}")
            
            # Optionally, show the prediction probabilities for each class
            st.write("Prediction Probabilities:")
            for i, class_name in enumerate(class_names):
                st.write(f"{class_name}: {prediction[0][i]:.2f}")

# Run the app
if __name__ == "__main__":
    main()



 
