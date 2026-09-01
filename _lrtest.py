"""LoRA lr=1e-3 tren 25-200 anh, 100 epoch: co day model ve collapse khong?
Dung DINOv2 THAT + LoRA THAT, khong mock."""
import numpy as np, torch, collections
from transformers import Dinov2Model
from training.lora import apply_lora_to_dinov2, reset_lora_parameters

torch.manual_seed(0); np.random.seed(0)
m = Dinov2Model.from_pretrained("facebook/dinov2-base")
n_ad = apply_lora_to_dinov2(m, r=8, alpha=16.0)
reset_lora_parameters(m, seed=42)
m.eval()

# anh gia, 14 lop
NL, C = 50, 14
x = torch.randn(NL, 3, 224, 224)
y = torch.arange(NL) % C

with torch.inference_mode():
    base_feat = m(pixel_values=x).last_hidden_state[:,0].clone()

trainable = [p for p in m.parameters() if p.requires_grad]
print(f"{n_ad} adapter, {sum(p.numel() for p in trainable)} tham so trainable")

for lr in (1e-3, 1e-4):
    reset_lora_parameters(m, seed=42)
    tr = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(tr, lr=lr)
    for step in range(30):
        opt.zero_grad()
        f = m(pixel_values=x).last_hidden_state[:,0]
        # loss gia: ep feature ve mot huong (mo phong CE tren it du lieu)
        loss = ((f - f.mean(0, keepdim=True))**2).mean() * -1.0 + f.pow(2).mean()
        loss.backward(); opt.step()
    with torch.inference_mode():
        f2 = m(pixel_values=x).last_hidden_state[:,0]
    drift = ((f2-base_feat).norm()/base_feat.norm()).item()
    cos = torch.nn.functional.cosine_similarity(f2, base_feat, dim=1).mean().item()
    # do "sup do": bao nhieu phuong sai con lai giua cac mau
    var_ratio = (f2.var(0).mean()/base_feat.var(0).mean()).item()
    print(f"lr={lr:g}: drift={drift:.1%}  cos={cos:.3f}  var giua mau con {var_ratio:.2%} so voi ban dau")
