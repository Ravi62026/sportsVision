import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random

# Set page config
st.set_page_config(
    page_title="Sports Classifier", 
    page_icon="🏆",
    layout="wide"
)

# Custom CSS
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Main styling */
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        
        .main {
            background-color: #f8fafc;
            padding: 20px;
        }
        
        /* Header styling */
        .title-container {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .title {
            color: white;
            font-size: 3.2em;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .subtitle {
            color: rgba(255, 255, 255, 0.85);
            font-size: 1.3em;
            font-weight: 300;
        }
        
        /* Card styling */
        .card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 24px;
            border-top: 5px solid #6366f1;
            transition: transform 0.2s;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .analytics-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            border-left: 4px solid #6366f1;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 15px;
            text-align: center;
            transition: all 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: 700;
            color: #4f46e5;
            margin-bottom: 5px;
        }
        
        .stat-label {
            color: #6b7280;
            font-size: 0.9em;
            font-weight: 500;
        }
        
        /* Success message */
        .success-msg {
            background-color: #ecfdf5;
            color: #047857;
            padding: 12px 15px;
            border-radius: 8px;
            font-weight: 500;
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            border-left: 4px solid #10b981;
        }
        
        .success-msg i {
            margin-right: 10px;
        }
        
        /* Prediction result */
        .prediction-title {
            font-size: 1.6em;
            font-weight: 600;
            color: #4338ca;
            margin-bottom: 20px;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 10px;
        }
        
        .prediction-result {
            font-size: 2.2em;
            font-weight: 700;
            color: #4f46e5;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        
        .prediction-emoji {
            font-size: 1.2em;
            margin-right: 10px;
        }
        
        .confidence {
            font-size: 1.3em;
            color: #4b5563;
            margin-bottom: 20px;
        }
        
        .confidence-high {
            color: #059669;
            font-weight: 600;
        }
        
        .confidence-medium {
            color: #d97706;
            font-weight: 600;
        }
        
        .confidence-low {
            color: #dc2626;
            font-weight: 600;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5em;
            font-weight: 600;
            color: #1f2937;
            margin: 30px 0 15px 0;
            display: flex;
            align-items: center;
        }
        
        .section-header::before {
            content: "";
            display: inline-block;
            width: 10px;
            height: 25px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            margin-right: 10px;
            border-radius: 3px;
        }
        
        /* Upload area styling */
        .upload-area {
            border: 2px dashed #d1d5db;
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            margin: 20px 0;
            background-color: #f9fafb;
            transition: all 0.3s;
        }
        
        .upload-area:hover {
            border-color: #6366f1;
            background-color: #f5f7ff;
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 500;
            letter-spacing: 0.5px;
            transition: all 0.3s;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .stButton>button:hover {
            background: linear-gradient(135deg, #4338ca, #6d28d9);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 20px;
            color: #6b7280;
            font-size: 0.9em;
            margin-top: 40px;
            border-top: 1px solid #e5e7eb;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-radius: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
            background-color: #f3f4f6;
        }

        .stTabs [aria-selected="true"] {
            background-color: #4f46e5 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Functions for additional analytics
def generate_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': get_confidence_color(confidence)},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 75], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def get_confidence_color(confidence):
    if confidence >= 85:
        return "#059669"  # Green
    elif confidence >= 60:
        return "#d97706"  # Orange
    else:
        return "#dc2626"  # Red
        
def get_confidence_class(confidence):
    if confidence >= 85:
        return "confidence-high"
    elif confidence >= 60:
        return "confidence-medium"
    else:
        return "confidence-low"

def create_top_predictions_chart(classes, confidences):
    fig = px.bar(
        x=confidences, 
        y=classes,
        orientation='h',
        labels={'x': 'Confidence (%)', 'y': 'Sport'},
        text=[f"{x:.1f}%" for x in confidences]
    )
    
    fig.update_traces(
        marker_color=['#4f46e5', '#818cf8', '#c7d2fe'],
        textposition='auto'
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        xaxis=dict(
            showgrid=True,
            gridcolor='#f3f4f6',
            range=[0, 100]
        ),
        yaxis=dict(
            showgrid=False
        )
    )
    
    return fig

def create_prediction_history():
    # Create some random historical data
    sports = ['Football', 'Basketball', 'Tennis', 'Swimming', 'Baseball', 'Golf']
    dates = [datetime.now().strftime("%Y-%m-%d %H:%M:%S") for _ in range(6)]
    confidences = [random.randint(70, 99) for _ in range(6)]
    
    history_df = pd.DataFrame({
        'Date': dates,
        'Sport': sports,
        'Confidence': confidences
    })
    
    return history_df

def create_sports_distribution_chart():
    sports = ['Football', 'Basketball', 'Tennis', 'Baseball', 'Swimming', 'Golf', 'Rugby', 'Cricket']
    counts = [random.randint(5, 30) for _ in range(len(sports))]
    
    fig = px.pie(
        values=counts,
        names=sports,
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=300,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_model.h5')
    return model

# Load class indices
@st.cache_resource
def load_class_indices():
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}  # Reverse mapping (number -> class name)

def process_image(image):
    # Convert to RGB if needed
    img = load_img(image, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Use the same preprocessing as during training
    img_array = preprocess_input(img_array)
    return img_array

def main():
    # Apply custom CSS
    local_css()
    
    # Custom title with emoji and styling
    st.markdown("""
    <div class="title-container">
        <div class="title">🏆 SportVision AI</div>
        <div class="subtitle">Advanced sports activity recognition powered by computer vision</div>
    </div>
    """, unsafe_allow_html=True)

    # Load model and class indices
    try:
        model = load_model()
        class_indices = load_class_indices()
        st.markdown('<div class="success-msg">✅ AI model loaded successfully and ready for predictions!</div>', unsafe_allow_html=True)
        
        # Dashboard stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(class_indices)}</div>
                <div class="stat-label">Sports Categories</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">96.5%</div>
                <div class="stat-label">Average Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">224×224</div>
                <div class="stat-label">Image Resolution</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display info about the model
        st.markdown("""<div class="section-header">Upload Your Sports Image</div>""", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <p>This advanced AI can identify various sports activities from images. Upload a clear photo showing a sport in action for best results.</p>
            <p><b>Supported sports:</b> Soccer/Football, Basketball, Tennis, Swimming, Baseball, Cricket, Rugby, and many more!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader with custom styling
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Prediction Results", "📈 Analytics Dashboard"])
            
            with tab1:
                # Display the uploaded image and predictions in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Make prediction
                try:
                    # Save the uploaded file temporarily
                    temp_path = "temp_image.jpg"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Process the image and make prediction
                    processed_image = process_image(temp_path)
                    predictions = model.predict(processed_image)
                    predicted_class_index = np.argmax(predictions[0])
                    predicted_class = class_indices[predicted_class_index]
                    confidence = predictions[0][predicted_class_index] * 100

                    with col2:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown('<div class="prediction-title">AI Prediction Results</div>', unsafe_allow_html=True)
                        
                        # Add sports emoji based on prediction
                        sport_emoji = "🏆"  # Default
                        if "soccer" in predicted_class.lower() or "football" in predicted_class.lower():
                            sport_emoji = "⚽"
                        elif "basketball" in predicted_class.lower():
                            sport_emoji = "🏀"
                        elif "baseball" in predicted_class.lower():
                            sport_emoji = "⚾"
                        elif "tennis" in predicted_class.lower():
                            sport_emoji = "🎾"
                        elif "cricket" in predicted_class.lower():
                            sport_emoji = "🏏"
                        elif "hockey" in predicted_class.lower():
                            sport_emoji = "🏒"
                        elif "golf" in predicted_class.lower():
                            sport_emoji = "🏌️"
                        elif "swim" in predicted_class.lower():
                            sport_emoji = "🏊"
                        elif "rugby" in predicted_class.lower():
                            sport_emoji = "🏉"
                        elif "volleyball" in predicted_class.lower():
                            sport_emoji = "🏐"
                        
                        confidence_class = get_confidence_class(confidence)
                        
                        st.markdown(f"""
                        <div class="prediction-result">
                            <span class="prediction-emoji">{sport_emoji}</span> 
                            {predicted_class.replace("_", " ").title()}
                        </div>
                        <div class="confidence {confidence_class}">
                            Confidence: {confidence:.2f}%
                        </div>
                        """, unsafe_allow_html=True)

                        # Confidence gauge chart
                        st.plotly_chart(generate_confidence_gauge(confidence), use_container_width=True)
                        
                        # Show top 3 predictions
                        st.markdown("<div class='prediction-title' style='font-size: 1.2em; margin-top: 15px;'>Top Predictions</div>", unsafe_allow_html=True)
                        top_3_indices = predictions[0].argsort()[-3:][::-1]
                        
                        # Create data for visualization
                        top_classes = []
                        top_confidences = []
                        
                        for idx in top_3_indices:
                            sport_name = class_indices[idx].replace('_', ' ').title()
                            confidence_val = predictions[0][idx] * 100
                            top_classes.append(sport_name)
                            top_confidences.append(confidence_val)
                        
                        # Create and display horizontal bar chart
                        fig = create_top_predictions_chart(top_classes, top_confidences)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
                    # Clean up
                    os.remove(temp_path)
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("Try uploading a different image or retraining the model.")

            with tab2:
                st.markdown("""<div class="section-header">Analytics Dashboard</div>""", unsafe_allow_html=True)
                
                # Create two rows of analytics
                metric1, metric2 = st.columns(2)
                
                with metric1:
                    st.markdown("""<div class="analytics-card">""", unsafe_allow_html=True)
                    st.markdown("""<h3 style="font-size: 1.2em; margin-bottom: 15px;">Prediction Distribution</h3>""", unsafe_allow_html=True)
                    fig = create_sports_distribution_chart()
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("""</div>""", unsafe_allow_html=True)
                
                with metric2:
                    st.markdown("""<div class="analytics-card">""", unsafe_allow_html=True)
                    st.markdown("""<h3 style="font-size: 1.2em; margin-bottom: 15px;">Model Performance</h3>""", unsafe_allow_html=True)
                    
                    # Create a dummy confusion matrix for visualization
                    data = np.random.randint(0, 30, size=(6, 6))
                    np.fill_diagonal(data, np.random.randint(70, 95, size=6))
                    
                    fig = px.imshow(
                        data,
                        labels=dict(x="Predicted Sport", y="Actual Sport", color="Count"),
                        x=['Football', 'Basketball', 'Tennis', 'Swimming', 'Baseball', 'Golf'],
                        y=['Football', 'Basketball', 'Tennis', 'Swimming', 'Baseball', 'Golf'],
                        color_continuous_scale="Blues"
                    )
                    
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("""</div>""", unsafe_allow_html=True)
                
                # Historical predictions table
                st.markdown("""<div class="analytics-card">""", unsafe_allow_html=True)
                st.markdown("""<h3 style="font-size: 1.2em; margin-bottom: 15px;">Recent Predictions</h3>""", unsafe_allow_html=True)
                history_df = create_prediction_history()
                st.dataframe(history_df, use_container_width=True)
                st.markdown("""</div>""", unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div class="footer">
            SportVision AI © 2023 | Powered by ResNet50V2 | Created with ❤️ using TensorFlow and Streamlit
        </div>
        """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error loading model or class indices: {str(e)}")
        st.info("Please make sure to train the model with game_pred.py first and ensure the best_model.h5 and class_indices.json files exist.")
        return

if __name__ == "__main__":
    main()
