import gradio as gr



def classify_image(input_image):
    if input_image is None:
        return "No image uploaded"
    
        
    return "Predicted Cancer Type"

# 3. UI SETUP
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(label="Upload Image"),
    
    # CHANGE: Use Textbox instead of Label since we have a string, not probabilities
    outputs=gr.Textbox(label="Result"), 
    
    title="Breast Cancer Type Classifier",
    description="Upload an image to see the specific cancer subtype."
)

iface.launch()
