"""CODAPath -- this project's earlier method, kept as a baseline.

Scores the pool against dual-VLM (PLIP + BiomedCLIP) text prototypes built from
per-class descriptions, then greedily maximises
`(1 - alpha) * coverage + alpha * uncertainty` with
`uncertainty = (1 - margin)(1 + JSD)` at tau = 0.05.

Unlike every other sampler here it selects in the VLM embedding space rather
than DINOv2, so `main.py` extracts VLM image features for it. Evaluation still
uses the shared DINOv2 probe, so the comparison stays fair.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPTokenizer, CLIPTextModel, AutoTokenizer, AutoModel

from utils.runtime import clear_memory
from ..registry import register_sampler


class DualVLMExtractor(nn.Module):
    def __init__(self,
                 plip_model: str = "vinid/plip",
                 biomedclip_model: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                 ) -> None:
        super().__init__()
        import open_clip

        backbone_plip = CLIPVisionModel.from_pretrained(plip_model)
        biomed_full, _, _ = open_clip.create_model_and_transforms(biomedclip_model)
        self.backbone_plip = backbone_plip
        self.backbone_biomed = biomed_full.visual

        for p in self.backbone_plip.parameters():
            p.requires_grad = False
        for p in self.backbone_biomed.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_plip = self.backbone_plip(pixel_values=x).pooler_output
        f_biomed = self.backbone_biomed(x)
        if isinstance(f_biomed, tuple):
            f_biomed = f_biomed[0]
        return torch.cat((f_plip, f_biomed), dim=1)


def extract_text_features(class_descriptions: Dict[str, str],
                           prompt_templates: List[str],
                           class_names: List[str],
                           device: torch.device,
                           plip_model: str = "vinid/plip",
                           biomedbert_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
                           ) -> np.ndarray:
    detailed_descriptions = [class_descriptions.get(cls, cls) for cls in class_names]
    list_prompts = [
        template.format(desc)
        for desc in detailed_descriptions
        for template in prompt_templates
    ]

    tokenizer_bio = AutoTokenizer.from_pretrained(biomedbert_model)
    encoder_bio = AutoModel.from_pretrained(biomedbert_model).to(device).eval()

    tokenizer_plip = CLIPTokenizer.from_pretrained(plip_model)
    encoder_plip = CLIPTextModel.from_pretrained(plip_model).to(device).eval()

    tokens_bio = tokenizer_bio(
        list_prompts, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        emb_bio = encoder_bio(**tokens_bio).pooler_output.cpu().numpy()

    tokens_plip = tokenizer_plip(
        list_prompts, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        emb_plip = encoder_plip(**tokens_plip).pooler_output.cpu().numpy()

    combined = np.concatenate((emb_plip, emb_bio), axis=1)
    text_embeddings = combined.reshape(
        len(class_names), len(prompt_templates), -1
    ).mean(axis=1).astype(np.float32)
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    del encoder_bio, encoder_plip, tokens_bio, tokens_plip, emb_bio, emb_plip, combined
    clear_memory()

    return text_embeddings


@register_sampler("codapath")
def coda_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    text_embeddings = kwargs["text_embeddings"]
    max_budget = kwargs["max_budget"]
    alpha = kwargs["alpha"]
    device = kwargs["device"]
    chunk_size = kwargs["chunk_size"]

    num_samples = image_embeddings.shape[0]

    img_tensor = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    txt_tensor = torch.tensor(text_embeddings, device=device, dtype=torch.float32)

    img_norm = F.normalize(img_tensor, p=2, dim=1)
    txt_norm = F.normalize(txt_tensor, p=2, dim=1)

    # --- Uncertainty: (1 - margin) * (1 + JSD) ---
    sim_matrix = torch.matmul(img_norm, txt_norm.T)
    probs = F.softmax(sim_matrix / 0.05, dim=1)

    sorted_probs, _ = torch.sort(probs, dim=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    predicted_classes = torch.argmax(probs, dim=1)
    one_hot_preds = torch.zeros_like(probs)
    one_hot_preds.scatter_(1, predicted_classes.unsqueeze(1), 1.0)

    M = 0.5 * (probs + one_hot_preds)
    kl_p_m = torch.sum(probs * torch.log(probs / M + 1e-10), dim=1)
    kl_q_m = torch.sum(one_hot_preds * torch.log(one_hot_preds / M + 1e-10), dim=1)
    jsd_scores = 0.5 * kl_p_m + 0.5 * kl_q_m

    uncertainty_weights = (1.0 - margins) * (1.0 + jsd_scores)
    if uncertainty_weights.max() > 0:
        uncertainty_weights = uncertainty_weights / uncertainty_weights.max()

    selected_indices = []
    unlabeled_tensor = img_norm
    uncertainty_tensor = uncertainty_weights

    max_sim_to_S = torch.full((num_samples, 1), -float("inf"), device=device)

    for _ in tqdm(range(max_budget), desc="CODAPath Selection"):
        global_max_coverage = 0.0

        for chunk_start in range(0, num_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_samples)
            chunk_tensor = unlabeled_tensor[chunk_start:chunk_end]

            sim_chunk = torch.matmul(unlabeled_tensor, chunk_tensor.T)
            gain_chunk = torch.clamp(sim_chunk - max_sim_to_S, min=0.0)
            coverage_gain_chunk = torch.sum(gain_chunk, dim=0)

            local_max = torch.max(coverage_gain_chunk).item()
            if local_max > global_max_coverage:
                global_max_coverage = local_max

            del chunk_tensor, sim_chunk, gain_chunk, coverage_gain_chunk
            clear_memory()

        best_candidate_global = -1
        best_objective_score = -float("inf")
        best_sim_column = None

        for chunk_start in range(0, num_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_samples)
            chunk_tensor = unlabeled_tensor[chunk_start:chunk_end]
            chunk_uncert = uncertainty_tensor[chunk_start:chunk_end]

            sim_chunk = torch.matmul(unlabeled_tensor, chunk_tensor.T)
            gain_chunk = torch.clamp(sim_chunk - max_sim_to_S, min=0.0)
            coverage_gain_chunk = torch.sum(gain_chunk, dim=0)

            if global_max_coverage > 0:
                normalized_coverage = coverage_gain_chunk / global_max_coverage
            else:
                normalized_coverage = torch.zeros_like(coverage_gain_chunk)

            objective_scores = (1 - alpha) * normalized_coverage + alpha * chunk_uncert

            for selected_idx in selected_indices:
                if chunk_start <= selected_idx < chunk_end:
                    local_idx = selected_idx - chunk_start
                    objective_scores[local_idx] = -float("inf")

            local_best_idx = torch.argmax(objective_scores).item()
            local_best_score = objective_scores[local_best_idx].item()

            if local_best_score > best_objective_score or best_sim_column is None:
                best_objective_score = local_best_score
                best_candidate_global = chunk_start + local_best_idx
                best_sim_column = sim_chunk[:, local_best_idx].reshape(-1, 1).clone()

            del chunk_tensor, chunk_uncert, sim_chunk, gain_chunk, coverage_gain_chunk, objective_scores
            clear_memory()

        if best_sim_column is not None:
            selected_indices.append(best_candidate_global)
            max_sim_to_S = torch.maximum(max_sim_to_S, best_sim_column)
            del best_sim_column
            clear_memory()
        else:
            print("Cannot find a valid candidate, stopping early")
            break

    del unlabeled_tensor, uncertainty_tensor, max_sim_to_S
    clear_memory()

    return selected_indices
