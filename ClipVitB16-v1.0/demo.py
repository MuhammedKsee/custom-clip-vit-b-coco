from utils import load_model, predict
from config import HParams

if __name__ == "__main__":
    # 1. Load the Model
    model = load_model()
    
    if model:
        # 2. Define Test Prompts (Classes)
        # You can change these to whatever you want to test
        test_labels = [
            "a photo of a cat", 
            "a photo of a dog", 
            "a photo of a sports car",
            "a photo of a pizza",
            "a photo of a laptop computer"
        ]
        
        # 3. Run Prediction
        # Ensure that 'images/test_image.jpg' exists or change the path in config.py
        print(f"🚀 Running inference on: {HParams.TEST_IMAGE}")
        predict(model, HParams.TEST_IMAGE, test_labels)