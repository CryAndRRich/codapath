import os
from typing import List, Tuple, Dict
from tqdm import tqdm

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torch.amp import autocast

import open_clip
from transformers import CLIPVisionModel, AutoTokenizer, AutoModel, CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model

from load_data import ActiveLearningDataset
from set_up import clear_memory
from sampling import get_sampler
from contrastive import train_contrastive

class CODAModel(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 r: int, 
                 lora_alpha: int) -> None: 
        super(CODAModel, self).__init__()
        
        backbone_plip = CLIPVisionModel.from_pretrained("vinid/plip")
        biomed_full, _, _ = open_clip.create_model_and_transforms("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        backbone_biomed = biomed_full.visual 

        config_plip = LoraConfig(
            r=r, 
            lora_alpha=lora_alpha, 
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.1, 
            bias="none"
        )
        config_biomed = LoraConfig(
            r=r, 
            lora_alpha=lora_alpha, 
            target_modules=["qkv", "in_proj", "out_proj"], 
            lora_dropout=0.1, 
            bias="none"
        )

        self.backbone_plip = get_peft_model(backbone_plip, config_plip)
        self.backbone_biomed = get_peft_model(backbone_biomed, config_biomed)

        dummy_x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feat1 = self.backbone_plip(pixel_values=dummy_x).pooler_output
            feat2 = self.backbone_biomed(dummy_x)
            if isinstance(feat2, tuple): 
                feat2 = feat2[0]
                
        fused_dim = feat1.shape[1] + feat2.shape[1]

        self.projection_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.classification_head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        plip_outputs = self.backbone_plip(pixel_values=x)
        f_plip = plip_outputs.pooler_output

        f_biomed = self.backbone_biomed(x)
        if isinstance(f_biomed, tuple): 
            f_biomed = f_biomed[0]
            
        f_concat = torch.cat((f_plip, f_biomed), dim=1) 
        f_proj = self.projection_head(f_concat)
        logits = self.classification_head(f_proj)
        
        return f_concat, f_proj, logits

@torch.inference_mode() 
def extract_image_embeddings(dataloader: DataLoader, 
                             model: nn.Module, 
                             device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()

    list_embeddings_proj, list_probs, list_embeddings_concat = [], [], []

    for images, _ in tqdm(dataloader, desc="Extracting Features", leave=False):
        images = images.to(device, non_blocking=True)
        with autocast(device_type="cuda"):
            f_concat, f_proj, logits = model(images)
            probs = F.softmax(logits, dim=1)
        
        list_embeddings_concat.append(f_concat.cpu().numpy().astype(np.float32))
        list_embeddings_proj.append(f_proj.cpu().numpy().astype(np.float32))
        list_probs.append(probs.cpu().numpy().astype(np.float32))

    embeddings_concat = np.vstack(list_embeddings_concat)
    embeddings_proj = np.vstack(list_embeddings_proj)
    probs = np.vstack(list_probs)
    
    del list_embeddings_concat, list_embeddings_proj, list_probs
    clear_memory()
    
    return embeddings_concat, embeddings_proj, probs

def extract_text_embeddings(class_descriptions: Dict[str, str], 
                            prompt_templates: List[str], 
                            class_names: List[str], 
                            device: torch.device) -> np.ndarray:
    detailed_descriptions = [class_descriptions.get(cls, cls) for cls in class_names]
    list_prompts = []
    for desc in detailed_descriptions:
        for template in prompt_templates:
            list_prompts.append(template.format(desc))
    
    tokenizer_bio = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    encoder_bio = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext").to(device).eval()
    
    tokenizer_plip = CLIPTokenizer.from_pretrained("vinid/plip")
    encoder_plip = CLIPTextModel.from_pretrained("vinid/plip").to(device).eval()
    
    tokens_bio = tokenizer_bio(
        list_prompts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    ).to(device)
    with torch.no_grad(): 
        emb_bio = encoder_bio(**tokens_bio).pooler_output.cpu().numpy()
    
    tokens_plip = tokenizer_plip(
        list_prompts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    ).to(device)
    with torch.no_grad(): 
        emb_plip = encoder_plip(**tokens_plip).pooler_output.cpu().numpy()
    
    list_text_embeddings = np.concatenate((emb_plip, emb_bio), axis=1)
    
    text_embeddings = list_text_embeddings.reshape(
        len(class_names), len(prompt_templates), -1
    ).mean(axis=1)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    del encoder_bio, encoder_plip, tokens_bio, tokens_plip, emb_bio, emb_plip, list_text_embeddings
    clear_memory()

    return text_embeddings

from evaluate import evaluate_model, visualize_tsne

def save_model(model: nn.Module, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    trainable_state_dict = {
        name: param.cpu() for name, param in model.named_parameters() if param.requires_grad
    }
    
    torch.save(trainable_state_dict, save_path)
    print(f"Saved model parameters to: {save_path}")

def load_model(model: nn.Module, load_path: str) -> nn.Module:
    state_dict = torch.load(load_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model parameters from: {load_path}")
    return model

def train_model(model: nn.Module, 
                sampler_name: str,
                train_dataset: Dataset, 
                oracle_labels: np.ndarray, 
                test_loader: DataLoader, 
                test_labels: np.ndarray, 
                class_names: List[str],
                text_embeddings: np.ndarray, 
                color_map: Dict[str, str],
                cumulative_budget: List[int], 
                num_epochs: int, 
                learn_rate: float,
                alpha: float, 
                r: int, 
                device: torch.device, 
                seed_worker_fn: callable, 
                g_seed: torch.Generator, 
                seed: int,
                save_dir: str,
                verbose: bool = False) -> None:
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=0,
        worker_init_fn=seed_worker_fn, 
        generator=g_seed
    )
    embeddings_concat, _, _ = extract_image_embeddings(train_loader, model, device=device)
    
    if verbose:
        visualize_tsne(
            embeddings_concat, 
            oracle_labels, 
            class_names, 
            title="Initial", 
            color_map=color_map, 
            seed=seed
        )

    queries_idx = get_sampler(
        name=sampler_name,
        image_embeddings=embeddings_concat, 
        text_embeddings=text_embeddings, 
        max_budget=max(cumulative_budget), 
        alpha=alpha, 
        device=device,
    )
    
    del embeddings_concat
    clear_memory()

    for budget in cumulative_budget:
        queried_labels = oracle_labels[queries_idx[:budget]]
        train_subset = Subset(train_dataset, queries_idx[:budget])
        al_dataset = ActiveLearningDataset(train_subset, queried_labels)
        al_loader = DataLoader(
            al_dataset, 
            batch_size=min(32, budget), 
            shuffle=True, 
            worker_init_fn=seed_worker_fn, 
            generator=g_seed
        )
        
        print(f"\nTraining with {budget} samples...")
        num_classes_total = len(class_names)
        del model 
        clear_memory() 
        
        model = CODAModel(
            num_classes=num_classes_total, 
            r=r, 
            lora_alpha=r * 2
        ).to(device)
        
        model = train_contrastive(
            model=model, 
            labeled_loader=al_loader, 
            num_epochs=num_epochs, 
            learn_rate=learn_rate, 
            device=device,
            verbose=verbose
        )
        
        if verbose:
            evaluate_model(model, test_loader, test_labels, device)
        
            embeddings_concat, _, _ = extract_image_embeddings(train_loader, model, device=device)
            visualize_tsne(
                embeddings_concat, 
                oracle_labels, 
                class_names, 
                title=f"After {budget} samples", 
                color_map=color_map, 
                seed=seed
            )

            del embeddings_concat

        save_path = f"{save_dir}/{sampler_name}_budget_{budget}.pth"
        save_model(model, save_path)
        
        del train_subset, al_dataset, al_loader, model
        clear_memory()