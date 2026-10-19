import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# 1. Browser Layout Configuration
st.set_page_config(page_title="AI Tomato Disease Assistant", layout="centered")

st.title("🌿 Tomato Leaf Disease Detection AI Assistant")
st.write("Class 12 CBSE AI Capstone Project Framework (Subject Code 417)")
st.write("---")

# 2. Cached Deep Learning Model Loader
@st.cache_resource
def load_ai_model():
    model = load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

try:
    model, class_names = load_ai_model()
    st.success("🤖 AI Engine Active: Tomato Diagnostic Weights Loaded Flawlessly!")
except Exception as e:
    st.error("⚠️ Error loading model matrix. Ensure 'keras_model.h5' and 'labels.txt' are inside the project folder.")

# 3. Interactive File Uploader
uploaded_file = st.file_uploader("Upload a Tomato Leaf Image Specimen...", type=["jpg", "png", "jpeg"])

# 4. Processing and Diagnosis Engine
if uploaded_file is not None:
    # Open and display image metrics to browser layout
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Specimen Matrix", use_container_width=True)
    
    st.write("🔬 *Running deep convolutional matrix scan...*")
    
    # Preprocess image to fit the 224x224 input constraint
    size = (224, 224)
    image_resized = image.resize(size)
    image_array = np.asarray(image_resized, dtype=np.float32)
    
    # Normalization array bounds between -1 and 1
    normalized_image_array = (image_array / 127.5) - 1
    tensor_input = np.expand_dims(normalized_image_array, axis=0)
    
    # Run Prediction Metrics
    prediction = model.predict(tensor_input, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[index]
    
    # 5. Output Analytical Results Dashboard & SDG 2 Cure Guide
    st.write("---")
    st.subheader("📊 Diagnostic Report Dashboard")
    st.metric(label="Detected Tomato Health State", value=f"{class_name}")
    st.write(f"**AI Confidence Accuracy Matrix:** {np.round(confidence_score * 100, 2)}%")
    
    st.write("---")
    st.subheader("🍅 Farmer Advisory & Treatment Protocol (SDG 2: Zero Hunger)")
    
    # Dynamic Advisory Logic based on model output class mapping
    if index == 0:
        st.success("✅ The leaf matrix is highly healthy. Ensure optimal sunlight exposure and consistent root-zone watering schedules.")
        
    elif index == 1:
        st.error("⚠️ Threat Detected: Bacterial Spot")
        st.markdown("""
        *   **Immediate Action:** Prune infected low-hanging branches immediately to eliminate soil splash spore contamination.
        *   **Treatment:** Apply an approved copper-based bactericide solution early in the morning.
        *   **Prevention:** Avoid overhead sprinkler irrigation systems entirely; water crops strictly at the soil layer.
        """)
        
    elif index == 2:
        st.error("⚠️ Threat Detected: Late Blight")
        st.markdown("""
        *   **Immediate Action:** Remove severely infected plants completely from the field area to isolate airborne spore dissemination.
        *   **Treatment:** Spray protectant fungicides across surrounding healthy tomato rows immediately.
        *   **Prevention:** Space out crop alignments during planting to maximize dry atmospheric airflow between foliage canvases.
        """)
        
    elif index == 3:
        st.error("⚠️ Threat Detected: Tomato Yellow Leaf Curl Virus")
        st.markdown("""
        *   **Immediate Action:** Actively monitor and eliminate whitefly pests, as they are the primary insect vectors distributing this pathogen.
        *   **Treatment:** Implement organic neem oils or insecticidal soap applications to disrupt vector feeding patterns.
        *   **Prevention:** Utilize physical row covers during early cultivation stages or deploy certified virus-resistant hybrid seed strains.
        """)

st.write("---")
st.caption("Developed using Python, Google Teachable Machine, and Orange Mining frameworks.")
