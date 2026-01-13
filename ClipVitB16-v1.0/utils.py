import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer
from config import HParams
from model import CLIP

# Load Tokenizer (Must match the training tokenizer)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

# Define Transform (Resize + CenterCrop + Normalize)
test_transform = transforms.Compose([
    transforms.Resize(HParams.IMAGE_SIZE),
    transforms.CenterCrop(HParams.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
])

def load_model(model_path=HParams.MODEL_PATH, device=HParams.DEVICE):
    print(f"⚙️  Device: {device}")
    model = CLIP(len(tokenizer)).to(device)
    
    if os.path.exists(model_path):
        print(f"📂 Loading model weights from: {model_path}")
        ckpt = torch.load(model_path, map_location=device)
        
        # Clean '_orig_mod.' prefix if the model was compiled during training
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        try:
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            
            # OPTIMIZATION: Enable FP16 (Half Precision) if on CUDA
            if device == "cuda":
                model = model.half() 
            
            print("✅ Model loaded successfully!")
            return model
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return None
    else:
        print(f"❌ Model file not found: {model_path}")
        print("   -> Please download 'best_model.pt' from Hugging Face and place it in the root directory.")
        return None

def predict(model, image_path, text_options):
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return

    # 1. Prepare Image
    try:
        img_pil = Image.open(image_path).convert("RGB")
    except:
        print("❌ Failed to open image file.")
        return
        
    img_tensor = test_transform(img_pil).unsqueeze(0).to(HParams.DEVICE)
    
    # Convert to FP16 if using CUDA
    if HParams.DEVICE == "cuda":
        img_tensor = img_tensor.half()
    
    # 2. Prepare Text
    text_inputs = tokenizer(
        text_options, 
        padding="max_length", 
        max_length=HParams.MAX_TOKENS, 
        truncation=True, 
        return_tensors="pt"
    ).to(HParams.DEVICE)
    
    # 3. Inference
    with torch.no_grad():
        img_features = model.visual(img_tensor)
        text_features = model.text(text_inputs["input_ids"], text_inputs["attention_mask"])
        
        # Normalization
        img_features /= img_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate Similarity
        similarity = (100.0 * img_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(len(text_options))

    # 4. Visualize Results
    plt.figure(figsize=(12, 6))
    
    # Show Image
    plt.subplot(1, 2, 1)
    plt.imshow(img_pil)
    plt.axis("off")
    plt.title("Input Image")
    
    # Show Chart
    plt.subplot(1, 2, 2)
    scores = values.cpu().float().numpy() * 100
    labels = [text_options[idx] for idx in indices.cpu().numpy()]
    
    # Color logic: Green for >50%, Blue for others
    colors = ['#4CAF50' if s > 50 else '#2196F3' for s in scores]
    plt.barh(range(len(labels)), scores, color=colors)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Confidence Score (%)')
    plt.xlim(0, 100)
    plt.gca().invert_yaxis() # Display highest score at top
    
    # Add labels to bars
    for i, v in enumerate(scores):
        plt.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()