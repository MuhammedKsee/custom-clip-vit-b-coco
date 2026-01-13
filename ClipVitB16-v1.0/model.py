import torch
import torch.nn as nn
import math
from config import HParams

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Patch Embedding
        self.conv1 = nn.Conv2d(3, HParams.VISION_WIDTH, kernel_size=16, stride=16, bias=False)
        
        scale = HParams.VISION_WIDTH ** -0.5
        self.cls = nn.Parameter(scale * torch.randn(HParams.VISION_WIDTH))
        self.pos = nn.Parameter(scale * torch.randn((HParams.IMAGE_SIZE // 16) ** 2 + 1, HParams.VISION_WIDTH))
        self.ln_pre = nn.LayerNorm(HParams.VISION_WIDTH)

        # Transformer Encoder
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
        x = self.ln_post(x[:, 0])  # Take CLS token

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
        
        # Masking (ignore padding)
        mask = attention_mask == 0 
        
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.ln(x)

        # Get embedding from EOT (End of Text) token
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
        # NOTE: During inference, encode_image and encode_text are usually called separately.
        # However, keeping the forward method ensures structural integrity.
        img_features = self.visual(img)
        text_features = self.text(text_ids, text_mask)

        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * img_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text