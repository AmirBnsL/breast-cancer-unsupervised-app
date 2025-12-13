import gradio as gr
import os
import sys
import numpy as np
import tempfile
from PIL import Image
import json
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from io import BytesIO
import base64

# Add parent directory to path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import prediction function
from src.ctranspath import predict_slide, CLASSES, visualize_preprocessing_pipeline

# Model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Load metadata only (lightweight)
with open(os.path.join(MODEL_DIR, 'meta.json'), 'r') as f:
    meta_7 = json.load(f)

with open(os.path.join(MODEL_DIR, '3_clas', '3_clas', 'meta_3.json'), 'r') as f:
    meta_3 = json.load(f)

CLASSES_3 = meta_3['classes_3']

# Global variables for lazy loading
_voting_clf = None
_xgb_clf = None
_voting_clf_3 = None

def _load_additional_models():
    """Lazy load additional models only when needed"""
    global _voting_clf, _xgb_clf, _voting_clf_3
    
    if _voting_clf is None:
        print("Loading ensemble models...")
        _voting_clf = joblib.load(os.path.join(MODEL_DIR, 'voting_clf.pkl'))
        _xgb_clf = joblib.load(os.path.join(MODEL_DIR, 'XGB.pkl'))
        _voting_clf_3 = joblib.load(os.path.join(MODEL_DIR, '3_clas', '3_clas', 'voting_clf_3.pkl'))
        print("Ensemble models loaded!")
    
    return _voting_clf, _xgb_clf, _voting_clf_3

# Class descriptions
CLASS_DESCRIPTIONS = {
    "0_N": "Normal tissue - No abnormalities detected",
    "1_PB": "Pathological Benign - Non-cancerous abnormal tissue",
    "2_UDH": "Usual Ductal Hyperplasia - Increased cell growth, benign",
    "3_FEA": "Flat Epithelial Atypia - Atypical cells, low-risk precursor",
    "4_ADH": "Atypical Ductal Hyperplasia - High-risk precursor lesion",
    "5_DCIS": "Ductal Carcinoma In Situ - Non-invasive cancer",
    "6_IC": "Invasive Carcinoma - Malignant, invasive cancer"
}

CLASS_3_DESCRIPTIONS = {
    "Benign": "Non-cancerous tissue (Normal, PB, UDH)",
    "HighRisk": "Pre-cancerous lesions requiring monitoring (FEA, ADH)",
    "Carcinoma": "Cancerous tissue (DCIS, Invasive Carcinoma)"
}

RISK_LEVELS = {
    "0_N": ("🟢 Low Risk", "#28a745"),
    "1_PB": ("🟢 Low Risk", "#28a745"),
    "2_UDH": ("🟡 Low-Moderate Risk", "#ffc107"),
    "3_FEA": ("🟠 Moderate Risk", "#fd7e14"),
    "4_ADH": ("🟠 High Risk", "#dc3545"),
    "5_DCIS": ("🔴 Very High Risk", "#c82333"),
    "6_IC": ("🔴 Critical - Malignant", "#bd2130")
}

def create_confidence_chart(patch_preds, classes):
    """Create a bar chart showing class distribution across patches"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts = np.bincount(patch_preds, minlength=len(classes))
    percentages = (counts / len(patch_preds)) * 100
    
    colors = sns.color_palette("husl", len(classes))
    bars = ax.bar(range(len(classes)), percentages, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Patches (%)', fontsize=12, fontweight='bold')
    ax.set_title('Patch-Level Class Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace('_', '\n') for c in classes], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        if pct > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_risk_gauge(risk_level, color):
    """Create a visual risk indicator"""
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('off')
    
    # Create risk scale
    risk_colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#bd2130']
    risk_labels = ['Low', 'Low-Moderate', 'Moderate', 'High', 'Critical']
    
    for i, (rc, rl) in enumerate(zip(risk_colors, risk_labels)):
        ax.barh(0, 1, left=i, height=0.5, color=rc, alpha=0.7, edgecolor='black', linewidth=2)
        ax.text(i+0.5, 0, rl, ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Highlight current risk
    current_idx = risk_labels.index(risk_level.split()[1])
    ax.barh(0, 1, left=current_idx, height=0.5, color=color, alpha=1, edgecolor='red', linewidth=4)
    
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title('Risk Assessment Scale', fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    return fig

def classify_image(input_image, model_type="7-class"):
    """
    Classify breast cancer type from uploaded histopathology image.
    
    Args:
        input_image: PIL Image or numpy array
        model_type: "7-class" or "3-class" classification
    
    Returns:
        Tuple of (result_text, confidence_chart, risk_gauge, statistics)
    """
    if input_image is None:
        return ("❌ No image uploaded. Please drag and drop or upload a histopathology image.", 
                None, None, "")
    
    print(f"\n{'='*50}")
    print(f"Starting classification - Mode: {model_type}")
    print(f"{'='*50}")
    
    # Return initial processing message
    initial_msg = "## 🔄 Processing...\n\nAnalyzing your image. This may take 5-15 seconds...\n\n"
    initial_msg += "*First run may take longer (30-60s) to download models.*"
    
    try:
        # Convert to PIL Image if needed and save temporarily
        if isinstance(input_image, np.ndarray):
            img = Image.fromarray(input_image.astype('uint8'))
        else:
            img = input_image
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            img.save(tmp_file.name)
            temp_path = tmp_file.name
        
        # Run prediction with progress feedback
        print("Running CTransPath prediction...")
        idx, label, patch_preds = predict_slide(temp_path, max_patches=100, batch_size=16)
        print(f"Prediction complete! Detected: {label}")
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Calculate statistics
        total_patches = len(patch_preds)
        majority_count = np.sum(patch_preds == idx)
        confidence = (majority_count / total_patches) * 100
        
        # Map to 3-class if needed
        if model_type == "3-class":
            # Map 7-class to 3-class
            mapping_7_to_3 = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
            patch_preds_3 = np.array([mapping_7_to_3[p] for p in patch_preds])
            counts_3 = np.bincount(patch_preds_3, minlength=3)
            idx_3 = counts_3.argmax()
            label_3 = CLASSES_3[idx_3]
            confidence_3 = (counts_3[idx_3] / total_patches) * 100
            
            # Create visualizations
            chart = create_confidence_chart(patch_preds_3, CLASSES_3)
            
            # Format result
            result = f"# 🔬 Diagnosis: **{label_3}**\n\n"
            result += f"## 📊 Confidence Score: **{confidence_3:.1f}%**\n"
            result += f"*Based on {total_patches} analyzed patches*\n\n"
            result += f"---\n\n"
            result += f"### 📋 Description:\n{CLASS_3_DESCRIPTIONS[label_3]}\n\n"
            result += f"### 🎯 Detailed Breakdown:\n"
            for i, cls in enumerate(CLASSES_3):
                pct = (counts_3[i] / total_patches) * 100
                result += f"- **{cls}**: {pct:.1f}% ({counts_3[i]} patches)\n"
            
            risk_gauge = None
            
        else:  # 7-class
            # Get risk level
            risk_level, risk_color = RISK_LEVELS[label]
            
            # Create visualizations
            chart = create_confidence_chart(patch_preds, CLASSES)
            risk_gauge = create_risk_gauge(risk_level, risk_color)
            
            # Format result
            result = f"# 🔬 Diagnosis: **{label}**\n\n"
            result += f"## 📊 Confidence Score: **{confidence:.1f}%**\n"
            result += f"*Based on {total_patches} analyzed patches*\n\n"
            result += f"## {risk_level}\n\n"
            result += f"---\n\n"
            result += f"### 📋 Description:\n{CLASS_DESCRIPTIONS[label]}\n\n"
            result += f"### 🎯 Detailed Statistics:\n"
            result += f"- **Primary Diagnosis**: {label}\n"
            result += f"- **Patches Analyzed**: {total_patches}\n"
            result += f"- **Consensus Patches**: {majority_count}\n"
            result += f"- **Agreement Rate**: {confidence:.1f}%\n\n"
        
        # Additional statistics
        stats = f"### 📈 Patch Analysis Summary\n\n"
        stats += f"| Metric | Value |\n|--------|-------|\n"
        stats += f"| Total Patches Extracted | {total_patches} |\n"
        stats += f"| Primary Class Agreement | {confidence:.1f}% |\n"
        stats += f"| Image Resolution | {img.size[0]} × {img.size[1]} px |\n"
        stats += f"| Model Type | CTransPath + Random Forest |\n"
        
        return result, chart, risk_gauge, stats
        
    except Exception as e:
        error_msg = f"# ❌ Error During Classification\n\n"
        error_msg += f"**Error Details:**\n```\n{str(e)}\n```\n\n"
        error_msg += "**Troubleshooting:**\n"
        error_msg += "- Ensure the image is a valid histopathology slide\n"
        error_msg += "- Check that the image contains tissue (not blank/black)\n"
        error_msg += "- Try uploading a different image format (PNG, JPG)\n"
        return error_msg, None, None, ""

def show_preprocessing_pipeline(input_image):
    """
    Show the preprocessing pipeline steps for the uploaded image.
    Each step shows: Title + Output Image
    """
    if input_image is None:
        return [], "❌ No image uploaded. Please upload an image first."
    
    try:
        # Convert to PIL Image and save temporarily
        if isinstance(input_image, np.ndarray):
            img = Image.fromarray(input_image.astype('uint8'))
        else:
            img = input_image
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            img.save(tmp_file.name)
            temp_path = tmp_file.name
        
        # Get preprocessing steps
        print("Generating preprocessing visualization...")
        steps = visualize_preprocessing_pipeline(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        # Create gallery images with titles
        gallery_images = []
        output_md = "# 🔬 Preprocessing Pipeline\n\n"
        output_md += "Below you can see each preprocessing step and how your image looks after that step:\n\n"
        output_md += "---\n\n"
        
        for idx, (title, img_array) in enumerate(steps, 1):
            # Clean title (remove numbering if present)
            clean_title = title.split('.', 1)[1].strip() if '.' in title else title
            output_md += f"## Step {idx}: {clean_title}\n\n"
            
            # Add image to gallery (convert numpy array to PIL Image)
            pil_img = Image.fromarray(img_array.astype('uint8'))
            gallery_images.append((pil_img, f"Step {idx}: {clean_title}"))
        
        output_md += "\n### 📊 Summary:\n"
        output_md += f"- **Total Steps:** {len(steps)}\n"
        output_md += "- **Patch Size:** 224×224 pixels\n"
        output_md += "- **Normalization:** ImageNet mean/std\n"
        output_md += "- **Tissue Threshold:** 20% minimum tissue content\n"
        
        return gallery_images, output_md
        
    except Exception as e:
        error_msg = f"## ❌ Error Generating Pipeline\n\n**Error:** {str(e)}"
        return [], error_msg

# Custom CSS for Modern Dark Medical UI
custom_css = """
/* Modern Dark Theme Color System */
:root {
    --primary-cyan: #00d9ff;
    --primary-teal: #00acc1;
    --dark-bg: #0a0e1a;
    --dark-surface: #131720;
    --dark-elevated: #1a1f2e;
    --dark-hover: #2d3748;
    --text-primary: #e8eaed;
    --text-secondary: #a0aec0;
    --text-muted: #718096;
    --border-subtle: #2d3748;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --glow-cyan: rgba(0, 217, 255, 0.3);
    --shadow-cyan: 0 0 20px rgba(0, 217, 255, 0.2);
}

/* Global Container Styling */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
    background: var(--dark-bg) !important;
    color: var(--text-primary) !important;
    background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 50%, #FFFFFF 100%) !important;
    min-height: 100vh;
}

/* Main Title Header with Ribbon Motif */
#main-title {
    text-align: center;
    background: linear-gradient(135deg, var(--pink-primary) 0%, var(--pink-dark) 100%);
    padding: 40px 30px;
    border-radius: 20px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(233, 30, 99, 0.3);
    border: 3px solid var(--pink-light);
    position: relative;
    overflow: hidden;
}

#main-title::before {
    content: "🎗️";
    position: absolute;
    top: 10px;
    right: 20px;
    font-size: 60px;
    opacity: 0.2;
}

#main-title h1 {
    font-weight: 700;
    font-size: 2.5em;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

#main-title p {
    font-size: 1.1em;
    font-weight: 400;
    margin-top: 10px;
    opacity: 0.95;
}

/* Upload Box Styling - Glass Morphism Design */
#upload-box {
    border: 2px dashed var(--primary-cyan);
    border-radius: 16px;
    padding: 48px 24px;
    background: linear-gradient(135deg, rgba(0, 217, 255, 0.05) 0%, rgba(0, 172, 193, 0.05) 100%);
    backdrop-filter: blur(10px);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    position: relative;
    overflow: hidden;
}

#upload-box::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, var(--glow-cyan) 0%, transparent 70%);
    opacity: 0;
    transition: opacity 0.4s ease;
    pointer-events: none;
}

#upload-box:hover {
    border-color: var(--primary-teal);
    background: linear-gradient(135deg, rgba(0, 217, 255, 0.08) 0%, rgba(0, 172, 193, 0.08) 100%);
    transform: translateY(-4px) scale(1.01);
    box-shadow: var(--shadow-cyan), 0 12px 32px rgba(0, 0, 0, 0.4);
}

#upload-box:hover::before {
    opacity: 1;
}

/* Result Box Styling - Elevated Card Design */
#result-box {
    background: var(--dark-elevated);
    border-radius: 16px;
    padding: 32px;
    border-left: 4px solid var(--primary-cyan);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    margin: 16px 0;
    position: relative;
    overflow: hidden;
}

#result-box::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 100px;
    height: 100px;
    background: radial-gradient(circle, var(--glow-cyan) 0%, transparent 70%);
    opacity: 0.3;
    pointer-events: none;
}

/* Typography System */
.markdown-text, .markdown-text *, 
#result-box, #result-box *,
.output-markdown, .output-markdown *,
.prose, .prose * {
    color: var(--text-primary) !important;
    line-height: 1.7;
}

.markdown-text h1, .markdown-text h2 {
    color: var(--primary-cyan) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    margin-bottom: 16px;
}

.markdown-text h3, .markdown-text h4 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    margin-bottom: 12px;
}

#result-box h1, #result-box h2, #result-box h3 {
    color: var(--pink-dark) !important;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 10px;
}

#result-box strong {
    color: var(--primary-cyan) !important;
    font-weight: 700;
}

/* Button System - Modern Neumorphism */
.primary-button, button[variant="primary"] {
    background: linear-gradient(135deg, var(--primary-cyan) 0%, var(--primary-teal) 100%) !important;
    color: var(--dark-bg) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 40px !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: var(--shadow-cyan), 0 4px 16px rgba(0, 0, 0, 0.3) !important;
    position: relative !important;
    overflow: hidden !important;
}

.primary-button::before, button[variant="primary"]::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.primary-button:hover, button[variant="primary"]:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 0 30px var(--glow-cyan), 0 8px 24px rgba(0, 0, 0, 0.4) !important;
}

.primary-button:hover::before, button[variant="primary"]:hover::before {
    width: 300px;
    height: 300px;
}

.primary-button:active, button[variant="primary"]:active {
    transform: translateY(-1px) scale(0.98) !important;
}

/* Button Styling - Secondary Action */
.secondary-button, button[variant="secondary"] {
    background: var(--dark-elevated) !important;
    color: var(--primary-cyan) !important;
    border: 2px solid var(--primary-cyan) !important;
    border-radius: 12px !important;
    padding: 14px 36px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}

.secondary-button:hover, button[variant="secondary"]:hover {
    background: rgba(0, 217, 255, 0.1) !important;
    border-color: var(--primary-teal) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-cyan), 0 6px 20px rgba(0, 0, 0, 0.3) !important;
}

/* Radio Button & Input Styling */
.radio-group, .radio-item {
    accent-color: var(--primary-cyan) !important;
}

input[type="radio"] {
    appearance: none;
    width: 20px;
    height: 20px;
    border: 2px solid var(--border-subtle);
    border-radius: 50%;
    position: relative;
    cursor: pointer;
    transition: all 0.3s ease;
}

input[type="radio"]:checked {
    background-color: var(--primary-cyan) !important;
    border-color: var(--primary-cyan) !important;
    box-shadow: 0 0 12px var(--glow-cyan);
}

input[type="radio"]:checked::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 8px;
    height: 8px;
    background: var(--dark-bg);
    border-radius: 50%;
    transform: translate(-50%, -50%);
}

/* Card/Panel Styling - Elevated Surfaces */
.panel, .block, .form {
    background: var(--dark-surface) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--border-subtle) !important;
    transition: all 0.3s ease !important;
}

.panel:hover, .block:hover {
    border-color: var(--primary-cyan);
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.4), var(--shadow-cyan);
}

/* Accordion Styling - Collapsible Sections */
.accordion {
    background: var(--dark-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
    margin: 20px 0 !important;
    overflow: hidden !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}

.accordion summary {
    background: var(--dark-elevated) !important;
    color: var(--primary-cyan) !important;
    font-weight: 700 !important;
    padding: 18px 24px !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border-left: 4px solid transparent !important;
    display: flex !important;
    align-items: center !important;
}

.accordion summary:hover {
    background: var(--dark-hover) !important;
    border-left-color: var(--primary-cyan) !important;
    padding-left: 28px !important;
}

.accordion[open] summary {
    border-bottom: 1px solid var(--border-subtle) !important;
    margin-bottom: 16px !important;
}

/* Gallery Styling - Image Grid */
.gallery {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    background: var(--dark-surface) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
}

.gallery img {
    border-radius: 12px !important;
    border: 1px solid var(--border-subtle) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}

.gallery img:hover {
    transform: scale(1.08) translateY(-4px) !important;
    border-color: var(--primary-cyan) !important;
    box-shadow: var(--shadow-cyan), 0 8px 24px rgba(0, 0, 0, 0.5) !important;
    z-index: 10 !important;
}

/* Plot/Chart Styling - Data Visualization */
.plot-container {
    background: var(--dark-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s ease !important;
}

.plot-container:hover {
    border-color: var(--primary-cyan);
    box-shadow: var(--shadow-cyan), 0 6px 24px rgba(0, 0, 0, 0.4);
}

/* Info/About Section Styling */
.markdown-text h3 {
    color: var(--primary-cyan) !important;
    font-weight: 700 !important;
    margin-top: 24px !important;
    margin-bottom: 12px !important;
    text-shadow: 0 0 10px var(--glow-cyan);
}

.markdown-text ul {
    list-style: none !important;
    padding-left: 0 !important;
}

.markdown-text ul li {
    margin: 10px 0 !important;
    padding-left: 28px !important;
    position: relative !important;
}

.markdown-text ul li::before {
    content: '▸' !important;
    color: var(--primary-cyan) !important;
    position: absolute !important;
    left: 8px !important;
    font-size: 1.2em !important;
}

/* Info Boxes - Status Messages */
.info-box {
    background: rgba(0, 217, 255, 0.08) !important;
    border-left: 4px solid var(--primary-cyan) !important;
    padding: 16px 24px !important;
    border-radius: 12px !important;
    margin: 16px 0 !important;
    color: var(--text-primary) !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2) !important;
    backdrop-filter: blur(10px) !important;
}

/* Responsive Typography System */
h1 {
    font-weight: 800 !important;
    color: var(--primary-cyan) !important;
    font-size: 2.5em !important;
    letter-spacing: -0.03em !important;
    text-shadow: 0 0 20px var(--glow-cyan);
}

h2 {
    font-weight: 700 !important;
    color: var(--primary-cyan) !important;
    font-size: 2em !important;
    letter-spacing: -0.02em !important;
}

h3, h4 {
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

h5, h6 {
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
}

p, li, span {
    line-height: 1.7 !important;
    color: var(--text-secondary) !important;
}

strong, b {
    color: var(--primary-cyan) !important;
    font-weight: 700 !important;
}

code {
    background: rgba(0, 217, 255, 0.1) !important;
    color: var(--primary-cyan) !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
    font-family: 'Monaco', 'Menlo', 'Courier New', monospace !important;
}

/* Loading/Processing Animation */
.loading {
    border: 3px solid var(--pink-lighter);
    border-top: 3px solid var(--pink-primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Custom Scrollbar - Minimal Design */
::-webkit-scrollbar {
    width: 12px;
}

::-webkit-scrollbar-track {
    background: var(--dark-surface);
    border-radius: 6px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--primary-cyan) 0%, var(--primary-teal) 100%);
    border-radius: 6px;
    border: 2px solid var(--dark-surface);
    transition: all 0.3s ease;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-cyan);
    box-shadow: 0 0 10px var(--glow-cyan);
}

/* Accessibility - Enhanced Focus States */
*:focus {
    outline: 2px solid var(--primary-cyan) !important;
    outline-offset: 3px !important;
    box-shadow: 0 0 0 4px rgba(0, 217, 255, 0.2) !important;
}

*:focus:not(:focus-visible) {
    outline: none !important;
    box-shadow: none !important;
}

/* Animations & Transitions */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.6;
    }
}

@keyframes shimmer {
    0% {
        background-position: -1000px 0;
    }
    100% {
        background-position: 1000px 0;
    }
}

.animate-in {
    animation: fadeIn 0.5s ease-out;
}

/* Loading States */
.loading-skeleton {
    background: linear-gradient(90deg, var(--dark-surface) 0%, var(--dark-hover) 50%, var(--dark-surface) 100%);
    background-size: 1000px 100%;
    animation: shimmer 2s infinite;
}

/* Status Badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.badge-success {
    background: rgba(16, 185, 129, 0.2);
    color: var(--success);
    border: 1px solid var(--success);
}

.badge-warning {
    background: rgba(245, 158, 11, 0.2);
    color: var(--warning);
    border: 1px solid var(--warning);
}

.badge-danger {
    background: rgba(239, 68, 68, 0.2);
    color: var(--danger);
    border: 1px solid var(--danger);
}

/* Responsive Design */
@media (max-width: 768px) {
    h1 {
        font-size: 2em !important;
    }
    
    .panel, .block {
        padding: 16px !important;
    }
    
    .primary-button, button[variant="primary"] {
        padding: 14px 28px !important;
        font-size: 1em !important;
    }
}
"""

# Create Gradio interface with Blocks for more control
with gr.Blocks(title="Breast Cancer AI Classifier") as demo:
    
    # Header
    gr.HTML("""
        <div style="background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 100%); padding: 30px; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(0, 217, 255, 0.2); border: 1px solid #00d9ff;">
            <h1 style="color: #00d9ff; margin: 0; font-size: 2.5em; text-align: center; text-shadow: 0 0 20px rgba(0, 217, 255, 0.5);">
                🎗️ Breast Cancer Histopathology Classifier
            </h1>
            <p style="color: #e8eaed; font-size: 1.2em; margin-top: 15px; text-align: center; font-weight: 500;">
                AI-Powered Early Detection & Analysis for Better Health Outcomes
            </p>
            <p style="color: #a0aec0; font-size: 1em; margin-top: 10px; text-align: center;">
                Supporting Awareness, Research & Timely Diagnosis
            </p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📤 Upload Image")
            input_image = gr.Image(
                label="Drag & Drop Histopathology Image Here",
                type="pil",
                elem_id="upload-box"
            )
            
            model_selector = gr.Radio(
                choices=["7-class", "3-class"],
                value="7-class",
                label="🎯 Classification Mode",
                info="Choose detailed (7-class) or simplified (3-class) analysis"
            )
            
            classify_btn = gr.Button(
                "🚀 Analyze Image",
                variant="primary",
                size="lg"
            )
            
            preprocess_btn = gr.Button(
                "🔍 Show Preprocessing Pipeline",
                variant="secondary",
                size="lg"
            )
    
            gr.Markdown("""
                ⏱️ **Processing Time:**
                - First run: ~30-60 sec (model download)
                - Subsequent runs: ~5-15 sec
                - Analyzes up to 100 tissue patches
            """)
            
            gr.Markdown("""
                ### 📚 Classification Categories
                
                **7-Class Model:**
                - 0_N: Normal
                - 1_PB: Pathological Benign
                - 2_UDH: Usual Ductal Hyperplasia
                - 3_FEA: Flat Epithelial Atypia
                - 4_ADH: Atypical Ductal Hyperplasia
                - 5_DCIS: Ductal Carcinoma In Situ
                - 6_IC: Invasive Carcinoma
                
                **3-Class Model:**
                - Benign (Safe)
                - High-Risk (Monitor)
                - Carcinoma (Malignant)
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("## 🎯 Analysis Results")
            result_output = gr.Markdown(
                elem_id="result-box",
                value="*Awaiting image upload...*"
            )
            
            with gr.Row():
                confidence_plot = gr.Plot(label="📊 Patch Distribution Analysis")
                risk_plot = gr.Plot(label="⚠️ Risk Assessment")
            
            stats_output = gr.Markdown()
    
    # Preprocessing Pipeline Section
    with gr.Accordion("🔬 Preprocessing Pipeline Visualization", open=False):
        gr.Markdown("""
            Click the **"Show Preprocessing Pipeline"** button above to see how your image is processed step-by-step.
            This will show you each transformation applied before classification.
        """)
        
        preprocess_gallery = gr.Gallery(
            label="Preprocessing Steps",
            show_label=True,
            columns=3,
            rows=2,
            height="auto",
            object_fit="contain"
        )
        
        preprocess_summary = gr.Markdown()
    
    # Set up event handlers
    classify_btn.click(
        fn=classify_image,
        inputs=[input_image, model_selector],
        outputs=[result_output, confidence_plot, risk_plot, stats_output]
    )
    
    preprocess_btn.click(
        fn=show_preprocessing_pipeline,
        inputs=[input_image],
        outputs=[preprocess_gallery, preprocess_summary]
    )

if __name__ == "__main__":
    # Create dark mode theme with cyan/teal accents
    custom_theme = gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="teal",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
    ).set(
        # Dark background colors
        body_background_fill="#0a0e1a",
        body_background_fill_dark="#0a0e1a",
        background_fill_primary="#131720",
        background_fill_primary_dark="#131720",
        background_fill_secondary="#1a1f2e",
        background_fill_secondary_dark="#1a1f2e",
        
        # Text colors
        body_text_color="#e8eaed",
        body_text_color_dark="#e8eaed",
        block_title_text_color="#00d9ff",
        block_title_text_color_dark="#00d9ff",
        block_label_text_color="#a0aec0",
        block_label_text_color_dark="#a0aec0",
        
        # Button colors
        button_primary_background_fill="#00d9ff",
        button_primary_background_fill_dark="#00d9ff",
        button_primary_background_fill_hover="#00b8d4",
        button_primary_background_fill_hover_dark="#00b8d4",
        button_primary_text_color="#0a0e1a",
        button_primary_text_color_dark="#0a0e1a",
        
        button_secondary_background_fill="#1a1f2e",
        button_secondary_background_fill_dark="#1a1f2e",
        button_secondary_background_fill_hover="#2d3748",
        button_secondary_background_fill_hover_dark="#2d3748",
        button_secondary_text_color="#00d9ff",
        button_secondary_text_color_dark="#00d9ff",
        
        # Border colors
        block_border_color="#2d3748",
        block_border_color_dark="#2d3748",
        border_color_primary="#00d9ff",
        border_color_primary_dark="#00d9ff",
        
        # Input colors
        input_background_fill="#1a1f2e",
        input_background_fill_dark="#1a1f2e",
        input_border_color="#2d3748",
        input_border_color_dark="#2d3748",
        
        # Shadow
        shadow_drop="0 4px 6px rgba(0, 217, 255, 0.1)",
        shadow_drop_lg="0 10px 15px rgba(0, 217, 255, 0.2)",
    )
    
    demo.launch(
        share=False, 
        server_name="127.0.0.1", 
        server_port=7864,
        css=custom_css,
        theme=custom_theme
    )
