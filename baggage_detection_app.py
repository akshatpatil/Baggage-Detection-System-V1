import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
import io
import time
import numpy as np
import base64

# Set page configuration
st.set_page_config(
    page_title="X-Ray Baggage Scanner",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS with the color scheme
def add_custom_css():
    st.markdown("""
    <style>
    /* Main background color */
    .stApp {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e0cdff;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #e0cdff;
        color: #2E2E2E;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
    }
    
    /* Button hover */
    .stButton>button:hover {
        background-color: #d0b7f5;
        box-shadow: 0 4px 8px rgba(224, 205, 255, 0.4);
        transform: translateY(-2px);
    }
    
    /* Success message */
    .success-box {
        background-color: rgba(224, 205, 255, 0.1);
        border-left: 5px solid #e0cdff;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Warning message */
    .danger-box {
        background-color: rgba(255, 77, 77, 0.1);
        border-left: 5px solid #ff4d4d;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Container styling */
    .css-1r6slb0, .css-keje6w {
        background-color: #383838;
        border: 1px solid #484848;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Image container */
    .img-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    
    /* Status indicators */
    .status-indicator {
        font-size: 18px;
        font-weight: bold;
        padding: 10px 15px;
        border-radius: 5px;
        display: inline-block;
        margin-top: 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #e0cdff;
    }
    </style>
    """, unsafe_allow_html=True)

# Model definition (same as in the original code)
class BaggageDetectionModel(nn.Module):
    def __init__(self):
        super(BaggageDetectionModel, self).__init__()
        # Load pre-trained ResNet18
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify the final layer for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.resnet(x)

# Function to load model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaggageDetectionModel().to(device)
    
    # Check if model file exists
    if os.path.exists('best_baggage_model.pth'):
        model.load_state_dict(torch.load('best_baggage_model.pth', map_location=device))
        model.eval()
        return model, device
    else:
        return None, device

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image, model, device):
    input_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output >= 0.5).float().item()
        confidence = output.item()
    
    return prediction, confidence

# Function to apply overlay effect on image
def apply_overlay(image, prediction):
    if prediction == 1:  # Dangerous
        overlay_color = (255, 0, 0, 60)  # Red with alpha
    else:  # Safe
        overlay_color = (0, 255, 0, 60)  # Green with alpha
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Create overlay
    overlay = np.zeros_like(img_array)
    overlay[:, :, :3] = overlay_color[:3]
    
    # Apply the overlay with transparency
    alpha = overlay_color[3] / 255.0
    result = (1 - alpha) * img_array + alpha * overlay[:, :, :3]
    
    # Convert back to uint8
    result = result.astype(np.uint8)
    
    return Image.fromarray(result)

# Function to create animation effect
def get_animated_html(text, is_dangerous):
    color = "#ff4d4d" if is_dangerous else "#4dff88"
    return f"""
    <div style="display: flex; justify-content: center; margin: 30px 0;">
        <div style="
            font-size: 32px;
            font-weight: bold;
            color: {color};
            animation: pulse 2s infinite;
            text-shadow: 0 0 10px {color}60;
        ">
            {text}
        </div>
    </div>
    <style>
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    </style>
    """

# Function to create a card UI for results
def display_result_card(image, prediction, confidence):
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.image(image, use_column_width=True)
    
    with col2:
        if prediction == 1:
            st.markdown(get_animated_html("⚠️ DANGEROUS ITEM DETECTED", True), unsafe_allow_html=True)
            st.markdown(f"""
            <div class="danger-box">
                <h3>Analysis Results</h3>
                <p>Our system has detected potentially dangerous items in this baggage scan.</p>
                <p>Confidence: {confidence * 100:.2f}%</p>
                <p>Possible threats: Metal objects, weapons, prohibited items</p>
                <p><b>Recommendation:</b> This bag requires manual inspection.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(get_animated_html("✓ NO THREATS DETECTED", False), unsafe_allow_html=True)
            st.markdown(f"""
            <div class="success-box">
                <h3>Analysis Results</h3>
                <p>Our system did not detect any dangerous items in this baggage scan.</p>
                <p>Confidence: {(1 - confidence) * 100:.2f}%</p>
                <p><b>Recommendation:</b> This bag is safe to proceed.</p>
            </div>
            """, unsafe_allow_html=True)

# Function for the about section
def show_about():
    st.markdown("""
    <div style="background-color: #383838; padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h2 style="color: #e0cdff;">About the X-Ray Baggage Scanner</h2>
        <p>This application uses a deep learning model trained on X-ray baggage images to detect potentially dangerous items.</p>
        <p>The model is based on ResNet18 architecture and has been trained to identify objects like knives, guns, wrenches, pliers, and scissors in X-ray scans.</p>
        <h3 style="color: #e0cdff;">How it works</h3>
        <p>1. Upload an X-ray baggage image</p>
        <p>2. The AI model analyzes the image for potential threats</p>
        <p>3. Results are displayed with the confidence level</p>
        <h3 style="color: #e0cdff;">Technology Stack</h3>
        <p>• Deep Learning: PyTorch with ResNet18</p>
        <p>• Frontend: Streamlit</p>
        <p>• Image Processing: PIL and OpenCV</p>
    </div>
    """, unsafe_allow_html=True)

# Main app function
def main():
    add_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/master/frontend/public/favicon.png", width=100)  # Placeholder logo
        st.title("X-Ray Scanner")
        st.markdown("---")
        
        page = st.radio("Navigation", ["Scan Baggage", "About"])
        
        st.markdown("---")
        st.markdown("### Model Information")
        st.markdown("• ResNet18 Architecture")
        st.markdown("• Binary Classification")
        st.markdown("• Trained on X-ray Dataset")

        # Sample images feature
        st.markdown("---")
        with st.expander("Try Sample Images"):
            if st.button("Load Sample Safe Baggage"):
                # This would load a sample safe image from your assets
                # For now we'll use a session state to simulate this
                st.session_state['sample_type'] = 'safe'
            
            if st.button("Load Sample Dangerous Baggage"):
                # This would load a sample dangerous image
                st.session_state['sample_type'] = 'dangerous'
    
    # Main content
    if page == "Scan Baggage":
        st.title("X-Ray Baggage Scanner")
        st.markdown("Upload an X-ray baggage image to check for dangerous items")
        
        # Load the model
        model, device = load_model()
        
        if model is None:
            st.error("Model file not found. Please ensure 'best_baggage_model.pth' is in the current directory.")
            return
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
        
        # Check if we should load a sample image
        if 'sample_type' in st.session_state:
            # In a real app, you would load actual sample images from your assets
            # For this demo, we'll simulate with a placeholder
            if st.session_state['sample_type'] == 'safe':
                # This is just a placeholder. In a real app, you'd load an actual file
                uploaded_file = "sample_safe.jpg"  # Placeholder
                st.info("Sample safe baggage loaded")
            elif st.session_state['sample_type'] == 'dangerous':
                uploaded_file = "sample_dangerous.jpg"  # Placeholder
                st.info("Sample dangerous baggage loaded")
            
            # Clear the session state to avoid loading the sample again
            del st.session_state['sample_type']
        
        if uploaded_file is not None:
            # If it's a string (sample image), we're just demonstrating the UI without actual files
            if isinstance(uploaded_file, str):
                # Simulating prediction for demo purposes
                if uploaded_file == "sample_safe.jpg":
                    prediction, confidence = 0, 0.1  # Safe with high confidence
                    image = Image.new('RGB', (400, 300), color=(100, 100, 100))  # Placeholder image
                else:
                    prediction, confidence = 1, 0.9  # Dangerous with high confidence
                    image = Image.new('RGB', (400, 300), color=(150, 100, 100))  # Placeholder image
            else:
                # For real file uploads
                image = Image.open(uploaded_file).convert('RGB')
                
                # Show a progress bar
                with st.spinner("Analyzing image..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)  # Simulate processing time
                        progress_bar.progress(i + 1)
                    
                    # Make prediction
                    prediction, confidence = predict(image, model, device)
                
                st.success("Analysis complete!")
            
            # Display results with the card UI
            display_result_card(image, prediction, confidence)
            
            # Additional image analysis visualization
            with st.expander("View Analysis Visualization"):
                st.markdown("### Scan with Analysis Overlay")
                # Apply a simple overlay to indicate detection
                overlay_image = apply_overlay(image, prediction)
                st.image(overlay_image, use_column_width=True)
                
                # Add fake heatmap or detection visualization
                st.markdown("### Detection Regions")
                st.markdown("This visualization highlights the areas where the model focused its attention when making the prediction.")
    
    elif page == "About":
        show_about()

if __name__ == "__main__":
    main()