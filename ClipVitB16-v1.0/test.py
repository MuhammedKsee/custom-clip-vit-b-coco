import os
import math
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import CLIPTokenizer

# ============================================================
# 1. AYARLAR (Eğitim kodundaki HParams'ın Test Versiyonu)
# ============================================================
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
    MODEL_PATH = "best_model.pt"      # Model dosyanın adı (yanında olmalı)
    TEST_IMAGE = "images/image8.png"    # Test edilecek resim (yanında olmalı)

# ============================================================
# 2. MODEL MİMARİSİ (Training.py'dan Birebir Alındı)
# ============================================================

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Patch Embedding
        self.conv1 = nn.Conv2d(3, HParams.VISION_WIDTH, kernel_size=16, stride=16, bias=False)
        
        scale = HParams.VISION_WIDTH ** -0.5
        self.cls = nn.Parameter(scale * torch.randn(HParams.VISION_WIDTH))
        self.pos = nn.Parameter(scale * torch.randn((HParams.IMAGE_SIZE // 16) ** 2 + 1, HParams.VISION_WIDTH))
        self.ln_pre = nn.LayerNorm(HParams.VISION_WIDTH)

        # Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=HParams.VISION_WIDTH,
            nhead=HParams.VISION_HEADS,
            dim_feedforward=HParams.VISION_WIDTH * HParams.FFN_MULT,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, HParams.VISION_LAYERS)

        self.ln_post = nn.LayerNorm(HParams.VISION_WIDTH)
        self.proj = nn.Parameter(scale * torch.randn(HParams.VISION_WIDTH, HParams.EMBED_DIM))

    def forward(self, x):
        x = self.conv1(x)  # [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, L, C]
        
        cls = self.cls.expand(x.size(0), 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos
        
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x[:, 0])  # CLS token

        return x @ self.proj

class TextTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token = nn.Embedding(vocab_size, HParams.TEXT_WIDTH)
        self.pos = nn.Parameter(torch.randn(HParams.MAX_TOKENS, HParams.TEXT_WIDTH))

        layer = nn.TransformerEncoderLayer(
            d_model=HParams.TEXT_WIDTH,
            nhead=HParams.TEXT_HEADS,
            dim_feedforward=HParams.TEXT_WIDTH * HParams.FFN_MULT,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, HParams.TEXT_LAYERS)
        self.ln = nn.LayerNorm(HParams.TEXT_WIDTH)
        self.proj = nn.Parameter(torch.randn(HParams.TEXT_WIDTH, HParams.EMBED_DIM))

    def forward(self, input_ids, attention_mask):
        x = self.token(input_ids)
        x = x + self.pos[:x.size(1)]
        
        # Maskeleme
        mask = attention_mask == 0 
        
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.ln(x)

        # EOT token embeddingini al
        eot_indices = attention_mask.sum(dim=1) - 1
        x = x[torch.arange(x.size(0)), eot_indices]
        return x @ self.proj

class CLIP(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.visual = VisionTransformer()
        self.text = TextTransformer(vocab_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, img, text_ids, text_mask):
        # NOT: Test sırasında genellikle bu forward yerine
        # encode_image ve encode_text ayrı ayrı çağrılır ama
        # sınıf bütünlüğü bozulmasın diye burası kalsın.
        img_features = self.visual(img)
        text_features = self.text(text_ids, text_mask)

        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * img_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

# ============================================================
# 3. TEST FONKSİYONLARI
# ============================================================

# Tokenizer (Eğitimde kullandığının aynısı olmalı)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

# Test Transform (Resize + CenterCrop)
test_transform = transforms.Compose([
    transforms.Resize(HParams.IMAGE_SIZE),
    transforms.CenterCrop(HParams.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
])

def load_model():
    print(f"⚙️ Çalışma Ortamı: {HParams.DEVICE}")
    model = CLIP(len(tokenizer)).to(HParams.DEVICE)
    
    if os.path.exists(HParams.MODEL_PATH):
        print(f"📂 Model yükleniyor: {HParams.MODEL_PATH}")
        ckpt = torch.load(HParams.MODEL_PATH, map_location=HParams.DEVICE)
        
        # '_orig_mod.' temizliği (Compile kullanıldıysa oluşur)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        try:
            model.load_state_dict(new_state_dict, strict=False)
            model.eval() # Test modu
            print("✅ Model başarıyla yüklendi!")
            return model
        except Exception as e:
            print(f"❌ Ağırlık yükleme hatası: {e}")
            return None
    else:
        print(f"❌ Model dosyası bulunamadı: {HParams.MODEL_PATH}")
        return None

def predict(model, image_path, text_options):
    if not os.path.exists(image_path):
        print(f"❌ Resim bulunamadı: {image_path}")
        return

    # 1. Resim Hazırla
    try:
        img_pil = Image.open(image_path).convert("RGB")
    except:
        print("❌ Resim açılamadı.")
        return
        
    img_tensor = test_transform(img_pil).unsqueeze(0).to(HParams.DEVICE)
    
    # 2. Metin Hazırla
    text_inputs = tokenizer(
        text_options, 
        padding="max_length", 
        max_length=HParams.MAX_TOKENS, 
        truncation=True, 
        return_tensors="pt"
    ).to(HParams.DEVICE)
    
    # 3. Tahmin
    with torch.no_grad():
        # Görsel Özellikleri
        img_features = model.visual(img_tensor)
        # Metin Özellikleri
        text_features = model.text(text_inputs["input_ids"], text_inputs["attention_mask"])
        
        # Normalize
        img_features /= img_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Benzerlik
        similarity = (100.0 * img_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(len(text_options))

    # 4. Sonuçları Göster
    plt.figure(figsize=(12, 6))
    
    # Resim
    plt.subplot(1, 2, 1)
    plt.imshow(img_pil)
    plt.axis("off")
    plt.title("Test Edilen Resim")
    
    # Grafik
    plt.subplot(1, 2, 2)
    scores = values.cpu().numpy() * 100
    labels = [text_options[idx] for idx in indices.cpu().numpy()]
    
    # Renkli Çubuklar
    colors = ['#4CAF50' if s > 50 else '#2196F3' for s in scores]
    plt.barh(range(len(labels)), scores, color=colors)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Güven Skoru (%)')
    plt.xlim(0, 100)
    plt.gca().invert_yaxis()
    
    # Değerleri yaz
    for i, v in enumerate(scores):
        plt.text(v + 1, i, f"%{v:.1f}", va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

# ============================================================
# 4. ÇALIŞTIR
# ============================================================
if __name__ == "__main__":
    # Modeli Yükle
    model = load_model()
    
    if model:
        # Test Metinleri (İstediğin gibi değiştir)
        secenekler = [
            "a photo of a cat on the grass", 
            "a photo of a octopus in the water", 
            "a photo of a dog on the grass", 
            "a photo of a car on the road", 
            
        ]
        
        # Tahmin Yap
        predict(model, HParams.TEST_IMAGE, secenekler)