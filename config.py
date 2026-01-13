import torch

class HParams:
    # Model Mimari Ayarları (Model ağırlıkları ile eşleşmesi için AYNEN kalmalı)
    IMAGE_SIZE = 224
    MAX_TOKENS = 77
    VISION_WIDTH = 768
    TEXT_WIDTH = 512
    EMBED_DIM = 512
    VISION_LAYERS = 12
    TEXT_LAYERS = 12
    VISION_HEADS = 12
    TEXT_HEADS = 8
    FFN_MULT = 4
    
    # Test Ayarları
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "best_model.pt"      # Hugging Face'den indirilen model
    TEST_IMAGE = "images/test_image.jpg" # Test edilecek örnek resim