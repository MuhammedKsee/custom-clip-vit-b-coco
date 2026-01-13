from utils import load_model, predict
from config import HParams

if __name__ == "__main__":
    # 1. Load the Model
    model = load_model()
    
    if model:
        # 2. Define Test Prompts (Classes)
        # You can change these to whatever you want to test
        test_labels = [
            "a photo of a cat on the grass", 
            "a photo of a dog on the grass", 
            "a photo of a sports car on the road",
            "a photo of a octopus in the water",
            "a photo of a laptop computer on the table"
        ]
        
        # 3. Run Prediction
        # Ensure that 'images/test_image.jpg' exists or change the path in config.py
        print(f"🚀 Running inference on: {HParams.TEST_IMAGE}")
        predict(model, HParams.TEST_IMAGE, test_labels)
