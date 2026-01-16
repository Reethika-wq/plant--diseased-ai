import streamlit as st
import tensorflow as tf
import numpy as np
import re

# --- STEP 1: ADD SEARCH FUNCTION ---
def get_about_info(prediction_name, filename="DISEASE-GUIDE.md"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        sections = content.split("###")
        for section in sections:
            if prediction_name.lower() in section.lower():
                lines = section.strip().split('\n')
                return "\n".join(lines[1:])
        return "No additional information found for this variety."
    except Exception:
        return "Please ensure 'DISEASE-GUIDE.md' is in the same folder as this app."

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar

#app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

# import Image from pillow to open images
from PIL import Image
img = Image.open("Diseases.png")

# display image using streamlit
# width is used to set the width of an image
st.image(img)
app_mode = st.selectbox("Select a Page", ["HOME", "DISEASE RECOGNITION"])

#Main Page
# --- HOME PAGE ---
if app_mode == "HOME":
    # 1. Main Title with green color
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>AgroAI: Smart Disease Detection</h1>", unsafe_allow_html=True)
    
    # 2. Sub-headings
    st.markdown("<p style='text-align: center;'>Empowering Farmers with AI-Powered Plant Disease Recognition.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload plant images to detect diseases accurately and access actionable insights.</p>", unsafe_allow_html=True)

    st.write("---") # Add a divider line

    # 3. Features Section (3 Columns)
    st.markdown("## Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("feat-1.png", use_container_width=True)
        st.markdown("<p style='text-align: center;'><b>Disease Detection</b></p>", unsafe_allow_html=True)
        st.write("Identify plant diseases with AI.")

    with col2:
        st.image("feat-2.jpg", use_container_width=True)
        st.markdown("<p style='text-align: center;'><b>Actionable Insights</b></p>", unsafe_allow_html=True)
        st.write("Get disease details and remedies.")

    with col3:
        st.image("feat-3.png", use_container_width=True)
        st.markdown("<p style='text-align: center;'><b>Real-Time Results</b></p>", unsafe_allow_html=True)
        st.write("Receive instant predictions.")

    st.write("---")

    # 4. How It Works Section
    st.markdown("## How It Works")
    st.markdown("""
    1.  Navigate to the Disease Recognition page.
    2.  Upload an image of the affected plant leaf.
    3.  Get instant results along with disease information and treatment suggestions.
    """)
    
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Disease Recognition</h1>", unsafe_allow_html=True)
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        predicted_name = class_name[result_index]
        st.success(f"Model is Predicting it's a {predicted_name}")

        # 2. Get Info from Guide
        info_text = get_about_info(predicted_name)
        
        # 3. Create 'About' Box (UI)
        display_title = predicted_name.replace("___", " ").replace("_", " ")
        with st.expander(f"About {display_title}"):
            st.markdown(info_text)
