from utils import load_model, predict
from config import HParams

if __name__ == "__main__":
    # 1. Modeli Yükle
    model = load_model()
    
    if model:
        # 2. Test Seçeneklerini Belirle
        test_labels = [
            "a photo of a cat", 
            "a photo of a dog", 
            "a photo of a sports car",
            "a photo of a pizza",
            "a photo of a computer"
        ]
        
        # 3. Tahmin Yap
        # Not: 'images/test_image.jpg' yolunda bir resim olduğundan emin olun
        predict(model, HParams.TEST_IMAGE, test_labels)