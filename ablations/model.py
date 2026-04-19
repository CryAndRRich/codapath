import os
from contextlib import nullcontext
from typing import Dict, List, Tuple, Union

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from ablations import VALID_ABLATIONS
from ablations.contrastive import train_contrastive
from evaluate import visualize_tsne
from load_data import ActiveLearningDataset
from sampling import get_sampler
from set_up import clear_memory, set_seed


def _validate_ablation(ablation_approach: str) -> None:
    if ablation_approach not in VALID_ABLATIONS:
        raise ValueError(
            f"Unsupported ablation '{ablation_approach}'. Expected one of: {sorted(VALID_ABLATIONS)}"
        )


def _validate_sampler_embedding_shapes(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    sampler_name: str,
    ablation_approach: str,
) -> None:
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError(
            "Sampler embeddings must be rank-2 arrays. "
            f"Got image_embeddings.ndim={image_embeddings.ndim}, "
            f"text_embeddings.ndim={text_embeddings.ndim}."
        )

    if sampler_name == "codapath" and image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            "CODAPath ablation embedding mismatch. "
            f"ablation={ablation_approach}, "
            f"image_embeddings.shape={tuple(image_embeddings.shape)}, "
            f"text_embeddings.shape={tuple(text_embeddings.shape)}. "
            "Both must share the same feature width because CODAPath computes image-text similarity."
        )


def _normalize_numpy_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def _extract_plip_text_features(
    plip_model: CLIPModel,
    tokenized_inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    text_features = plip_model.get_text_features(**tokenized_inputs)
    if isinstance(text_features, torch.Tensor):
        return text_features

    text_outputs = plip_model.text_model(**tokenized_inputs)
    pooled_output = text_outputs.pooler_output
    if hasattr(plip_model, "text_projection") and plip_model.text_projection is not None:
        pooled_output = plip_model.text_projection(pooled_output)
    return pooled_output


def _extract_plip_image_features(
    plip_model: CLIPModel,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    image_features = plip_model.get_image_features(pixel_values=pixel_values)
    if isinstance(image_features, torch.Tensor):
        return image_features

    vision_outputs = plip_model.vision_model(pixel_values=pixel_values)
    pooled_output = vision_outputs.pooler_output
    if hasattr(plip_model, "visual_projection") and plip_model.visual_projection is not None:
        pooled_output = plip_model.visual_projection(pooled_output)
    return pooled_output


class AblationCODAModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        r: int,
        lora_alpha: int,
        ablation_approach: str,
    ) -> None:
        super().__init__()
        _validate_ablation(ablation_approach)

        self.ablation_approach = ablation_approach
        self.use_plip = ablation_approach != "biomedclip_only"
        self.use_biomedclip = ablation_approach != "plip_only"
        self.enable_lora = ablation_approach != "no_lora_cls_only"
        self.freeze_backbones = ablation_approach == "no_lora_cls_only"

        if self.use_plip:
            plip_backbone = CLIPVisionModel.from_pretrained("vinid/plip")
            if self.enable_lora:
                config_plip = LoraConfig(
                    r=r,
                    lora_alpha=lora_alpha,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.1,
                    bias="none",
                )
                self.backbone_plip = get_peft_model(plip_backbone, config_plip)
            else:
                self.backbone_plip = plip_backbone
                # Freeze the pretrained tower so the task heads are trained on fixed CLS features.
                for param in self.backbone_plip.parameters():
                    param.requires_grad = False
        else:
            self.backbone_plip = None

        if self.use_biomedclip:
            biomed_full, _, _ = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            biomed_backbone = biomed_full.visual
            if self.enable_lora:
                config_biomed = LoraConfig(
                    r=r,
                    lora_alpha=lora_alpha,
                    target_modules=["qkv", "in_proj", "out_proj"],
                    lora_dropout=0.1,
                    bias="none",
                )
                self.backbone_biomed = get_peft_model(biomed_backbone, config_biomed)
            else:
                self.backbone_biomed = biomed_backbone
                # Freeze the pretrained tower so the task heads are trained on fixed CLS features.
                for param in self.backbone_biomed.parameters():
                    param.requires_grad = False
        else:
            self.backbone_biomed = None

        fused_dim = self._infer_feature_dim()

        self.projection_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.classification_head = nn.Linear(256, num_classes)

    def _infer_feature_dim(self) -> int:
        dummy_x = torch.randn(1, 3, 224, 224)
        features = []

        with torch.no_grad():
            if self.backbone_plip is not None:
                plip_outputs = self.backbone_plip(pixel_values=dummy_x)
                features.append(plip_outputs.pooler_output)

            if self.backbone_biomed is not None:
                biomed_outputs = self.backbone_biomed(dummy_x)
                if isinstance(biomed_outputs, tuple):
                    biomed_outputs = biomed_outputs[0]
                features.append(biomed_outputs)

        if not features:
            raise RuntimeError("At least one backbone must be enabled for ablations.")

        return int(sum(feature.shape[1] for feature in features))

    def train(self, mode: bool = True) -> "AblationCODAModel":
        super().train(mode)
        if self.freeze_backbones:
            if self.backbone_plip is not None:
                self.backbone_plip.eval()
            if self.backbone_biomed is not None:
                self.backbone_biomed.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = []

        if self.backbone_plip is not None:
            plip_outputs = self.backbone_plip(pixel_values=x)
            features.append(plip_outputs.pooler_output)

        if self.backbone_biomed is not None:
            biomed_outputs = self.backbone_biomed(x)
            if isinstance(biomed_outputs, tuple):
                biomed_outputs = biomed_outputs[0]
            features.append(biomed_outputs)

        f_concat = features[0] if len(features) == 1 else torch.cat(features, dim=1)
        f_proj = self.projection_head(f_concat)
        logits = self.classification_head(f_proj)
        return f_concat, f_proj, logits


@torch.inference_mode()
def extract_image_embeddings(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()

    list_embeddings_proj, list_probs, list_embeddings_concat = [], [], []
    amp_context = autocast(device_type=device.type) if device.type == "cuda" else nullcontext()

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        with amp_context:
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


def extract_text_embeddings(
    class_descriptions: Dict[str, str],
    prompt_templates: List[str],
    class_names: List[str],
    device: torch.device,
    ablation_approach: str,
) -> np.ndarray:
    _validate_ablation(ablation_approach)
    use_plip = ablation_approach != "biomedclip_only"
    use_biomedclip = ablation_approach != "plip_only"

    detailed_descriptions = [class_descriptions.get(cls, cls) for cls in class_names]
    prompts = [template.format(desc) for desc in detailed_descriptions for template in prompt_templates]
    embeddings_per_tower = []

    if use_plip:
        plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
        plip_model = CLIPModel.from_pretrained("vinid/plip").to(device).eval()
        tokens_plip = plip_processor.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            emb_plip = _extract_plip_text_features(plip_model, tokens_plip).cpu().numpy()
        embeddings_per_tower.append(emb_plip)
        del plip_model, plip_processor, tokens_plip, emb_plip
        clear_memory()

    if use_biomedclip:
        biomed_model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        biomed_model = biomed_model.to(device).eval()
        biomed_tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        tokens_bio = biomed_tokenizer(prompts).to(device)
        with torch.no_grad():
            emb_bio = biomed_model.encode_text(tokens_bio).cpu().numpy()
        embeddings_per_tower.append(emb_bio)
        del biomed_model, biomed_tokenizer, tokens_bio, emb_bio
        clear_memory()

    if not embeddings_per_tower:
        raise RuntimeError("No text encoder was enabled for the selected ablation.")

    merged_embeddings = (
        embeddings_per_tower[0]
        if len(embeddings_per_tower) == 1
        else np.concatenate(embeddings_per_tower, axis=1)
    )
    text_embeddings = merged_embeddings.reshape(len(class_names), len(prompt_templates), -1).mean(axis=1)
    text_embeddings = _normalize_numpy_embeddings(text_embeddings)
    return text_embeddings


@torch.inference_mode()
def extract_sampler_image_embeddings(
    dataloader: DataLoader,
    device: torch.device,
    ablation_approach: str,
) -> np.ndarray:
    _validate_ablation(ablation_approach)
    use_plip = ablation_approach != "biomedclip_only"
    use_biomedclip = ablation_approach != "plip_only"

    embeddings_per_batch = []

    plip_model = None
    plip_processor = None
    if use_plip:
        plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
        plip_model = CLIPModel.from_pretrained("vinid/plip").to(device).eval()

    biomed_model = None
    if use_biomedclip:
        biomed_model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        biomed_model = biomed_model.to(device).eval()

    amp_context = autocast(device_type=device.type) if device.type == "cuda" else nullcontext()

    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        towers = []

        with amp_context:
            if plip_model is not None:
                plip_features = _extract_plip_image_features(plip_model, images)
                towers.append(plip_features)

            if biomed_model is not None:
                biomed_features = biomed_model.encode_image(images)
                towers.append(biomed_features)

        merged = towers[0] if len(towers) == 1 else torch.cat(towers, dim=1)
        embeddings_per_batch.append(merged.cpu().numpy().astype(np.float32))

    if plip_model is not None:
        del plip_model, plip_processor
    if biomed_model is not None:
        del biomed_model
    clear_memory()

    return _normalize_numpy_embeddings(np.vstack(embeddings_per_batch))


def inspect_sampler_alignment(
    dataloader: DataLoader,
    class_descriptions: Dict[str, str],
    prompt_templates: List[str],
    class_names: List[str],
    device: torch.device,
    ablation_approach: str,
    sampler_name: str = "codapath",
) -> Dict[str, Union[str, Tuple[int, ...], bool]]:
    image_embeddings = extract_sampler_image_embeddings(
        dataloader=dataloader,
        device=device,
        ablation_approach=ablation_approach,
    )
    text_embeddings = extract_text_embeddings(
        class_descriptions=class_descriptions,
        prompt_templates=prompt_templates,
        class_names=class_names,
        device=device,
        ablation_approach=ablation_approach,
    )

    is_compatible = image_embeddings.shape[1] == text_embeddings.shape[1]
    return {
        "ablation_approach": ablation_approach,
        "sampler_name": sampler_name,
        "image_embeddings_shape": tuple(image_embeddings.shape),
        "text_embeddings_shape": tuple(text_embeddings.shape),
        "feature_width_match": is_compatible,
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    test_labels: np.ndarray,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    _, _, probs = extract_image_embeddings(test_loader, model, device=device)
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(test_labels, preds)
    pre = precision_score(test_labels, preds, average="macro", zero_division=0)
    rec = recall_score(test_labels, preds, average="macro", zero_division=0)
    f1 = f1_score(test_labels, preds, average="macro", zero_division=0)

    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {pre * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"Macro F1 : {f1 * 100:.2f}%")

    return acc, pre, rec, f1


def save_model(model: nn.Module, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    unwrapped_model = getattr(model, "_orig_mod", model)
    full_state_dict = unwrapped_model.state_dict()
    target_keys = ["lora", "projection_head", "classification_head"]
    trainable_state_dict = {
        key: value.cpu()
        for key, value in full_state_dict.items()
        if any(fragment in key for fragment in target_keys)
    }
    torch.save(trainable_state_dict, save_path)
    print(f"Saved {len(trainable_state_dict)} components to: {save_path}")


def load_model(model: nn.Module, load_path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded model parameters from: {load_path}")
    return model.to(device).eval()


def train_model(
    model: nn.Module,
    sampler_name: str,
    train_dataset: Dataset,
    oracle_labels: np.ndarray,
    test_loader: DataLoader,
    test_labels: np.ndarray,
    class_names: List[str],
    text_embeddings: np.ndarray,
    color_map: Dict[str, str],
    cumulative_budget: Union[List[int], str],
    num_epochs: int,
    learn_rate: float,
    alpha: float,
    r: int,
    device: torch.device,
    seed_worker_fn: callable,
    g_seed: torch.Generator,
    seed: int,
    save_dir: str,
    ablation_approach: str,
    verbose: bool = False,
) -> None:
    _validate_ablation(ablation_approach)
    unsupported_samplers = {"entropy", "margin", "badge"}
    if sampler_name in unsupported_samplers:
        raise ValueError(
            f"Sampler '{sampler_name}' is not supported in ablations because its selection "
            "logic instantiates the baseline model internally. Use codapath, random, coreset, "
            "typiclust, or activeft instead."
        )

    use_center_loss = ablation_approach != "no_contrastive_loss"

    if isinstance(cumulative_budget, str):
        al_dataset = ActiveLearningDataset(train_dataset, oracle_labels)
        al_loader = DataLoader(
            al_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker_fn,
            generator=g_seed,
        )

        fresh_model = AblationCODAModel(
            num_classes=len(class_names),
            r=r,
            lora_alpha=r * 2,
            ablation_approach=ablation_approach,
        ).to(device)
        fresh_model = train_contrastive(
            model=fresh_model,
            labeled_loader=al_loader,
            num_epochs=num_epochs,
            learn_rate=learn_rate,
            device=device,
            use_center_loss=use_center_loss,
            verbose=verbose,
        )

        if verbose:
            evaluate_model(fresh_model, test_loader, test_labels, device)

        save_path = os.path.join(save_dir, f"{sampler_name}_{ablation_approach}_{cumulative_budget}.pth")
        save_model(fresh_model, save_path)
        del al_dataset, al_loader, fresh_model
        clear_memory()
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker_fn,
        generator=g_seed,
    )

    embeddings_concat, embeddings_proj, _ = extract_image_embeddings(train_loader, model, device=device)
    sampler_image_embeddings = extract_sampler_image_embeddings(
        dataloader=train_loader,
        device=device,
        ablation_approach=ablation_approach,
    )
    _validate_sampler_embedding_shapes(
        image_embeddings=sampler_image_embeddings,
        text_embeddings=text_embeddings,
        sampler_name=sampler_name,
        ablation_approach=ablation_approach,
    )
    master_selected_indices = get_sampler(
        name=sampler_name,
        image_embeddings=sampler_image_embeddings,
        text_embeddings=text_embeddings,
        max_budget=max(cumulative_budget),
        alpha=alpha,
        device=device,
        chunk_size=10000,
    )

    for budget in cumulative_budget:
        seed_worker_fn = set_seed(seed)
        g_seed = torch.Generator()
        g_seed.manual_seed(seed)
        selected_indices = master_selected_indices[:budget]

        if verbose and budget == max(cumulative_budget):
            visualize_tsne(
                embeddings_proj=embeddings_proj,
                true_labels=oracle_labels,
                class_names=class_names,
                title=f"{ablation_approach} initial",
                color_map=color_map,
                seed=seed,
                selected_indices=selected_indices,
            )
            del embeddings_proj

        selected_labels = oracle_labels[selected_indices]
        train_subset = Subset(train_dataset, selected_indices)
        al_dataset = ActiveLearningDataset(train_subset, selected_labels)
        al_loader = DataLoader(
            al_dataset,
            batch_size=min(32, budget),
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker_fn,
            generator=g_seed,
        )

        fresh_model = AblationCODAModel(
            num_classes=len(class_names),
            r=r,
            lora_alpha=r * 2,
            ablation_approach=ablation_approach,
        ).to(device)
        fresh_model = train_contrastive(
            model=fresh_model,
            labeled_loader=al_loader,
            num_epochs=num_epochs,
            learn_rate=learn_rate,
            device=device,
            use_center_loss=use_center_loss,
            verbose=verbose,
        )

        if verbose:
            evaluate_model(fresh_model, test_loader, test_labels, device)

            if budget == max(cumulative_budget):
                _, embeddings_proj_new, _ = extract_image_embeddings(train_loader, fresh_model, device=device)
                visualize_tsne(
                    embeddings_proj=embeddings_proj_new,
                    true_labels=oracle_labels,
                    class_names=class_names,
                    title=f"{ablation_approach} after {budget} samples",
                    color_map=color_map,
                    seed=seed,
                    selected_indices=selected_indices,
                )
                del embeddings_proj_new
                clear_memory()

        save_path = os.path.join(save_dir, f"{sampler_name}_{ablation_approach}_budget_{budget}.pth")
        save_model(fresh_model, save_path)

        del train_subset, al_dataset, al_loader, fresh_model
        clear_memory()

    del embeddings_concat, sampler_image_embeddings, master_selected_indices
    clear_memory()
