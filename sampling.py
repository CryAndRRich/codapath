from typing import List
from tqdm import tqdm

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
import torch
import torch.nn.functional as F

from set_up import clear_memory

__sampler__ = {}

def register_sampler(name: str) -> object:
    def wrapper(cls):
        if __sampler__.get(name, None) is not None:
            raise ValueError(f"Sampler with name '{name}' is already registered")
        __sampler__[name] = cls
        return cls
    return wrapper

def get_sampler(name: str, **kwargs) -> object:
    if name not in __sampler__:
        raise ValueError(
            f"Sampler with name '{name}' is not registered", 
            available_models=list(__sampler__.keys())
        )
    return __sampler__[name](**kwargs)

@register_sampler("random")
def random_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    max_budget = kwargs["max_budget"]

    num_samples = image_embeddings.shape[0]
    selected_indices = np.random.choice(num_samples, max_budget, replace=False)
    
    return selected_indices.tolist()

@register_sampler("entropy")
def entropy_sampling(**kwargs) -> List[int]:
    max_budget = kwargs["max_budget"]
    train_dataset = kwargs["train_dataset"]
    oracle_labels = kwargs["oracle_labels"]
    train_loader = kwargs["train_loader"]
    num_epochs = kwargs["num_epochs"]
    learn_rate = kwargs["learn_rate"]
    r = kwargs["r"]
    class_names = kwargs["class_names"]
    seed_worker_fn = kwargs["seed_worker_fn"]
    g_seed = kwargs["g_seed"]
    device = kwargs["device"]
    
    from model import CODAModel, extract_image_embeddings
    from contrastive import train_contrastive
    from load_data import ActiveLearningDataset
    from torch.utils.data import Subset, DataLoader
    
    num_samples = len(train_dataset)
    step_budget = max(1, int(0.2 * max_budget))
    
    selected_indices = []
    unlabeled_indices = list(range(num_samples))
    
    while len(selected_indices) < max_budget:
        current_need = min(step_budget, max_budget - len(selected_indices))
        
        if len(selected_indices) == 0:
            np.random.shuffle(unlabeled_indices)
            chosen = unlabeled_indices[:current_need]
        else:
            selected_labels = oracle_labels[selected_indices]
            train_subset = Subset(train_dataset, selected_indices)
            al_dataset = ActiveLearningDataset(train_subset, selected_labels)
            al_loader = DataLoader(
                al_dataset, 
                batch_size=min(32, len(selected_indices)), 
                shuffle=True, 
                num_workers=0,
                worker_init_fn=seed_worker_fn, 
                generator=g_seed
            )
            
            model = CODAModel(num_classes=len(class_names), r=r, lora_alpha=r*2).to(device)
            model = train_contrastive(
                model=model, labeled_loader=al_loader, 
                num_epochs=num_epochs, learn_rate=learn_rate, 
                device=device, verbose=False 
            )
            
            _, _, probs = extract_image_embeddings(train_loader, model, device=device)
            
            unlabeled_probs = probs[unlabeled_indices]
            
            entropy_scores = -np.sum(unlabeled_probs * np.log(unlabeled_probs + 1e-10), axis=1)
            
            best_local_indices = np.argsort(entropy_scores)[::-1][:current_need]
            chosen = [unlabeled_indices[idx] for idx in best_local_indices]
            
            
            del model, al_dataset, al_loader, probs, unlabeled_probs, entropy_scores
            clear_memory()
            
        selected_indices.extend(chosen)
        chosen_set = set(chosen)
        unlabeled_indices = [idx for idx in unlabeled_indices if idx not in chosen_set]
        
    return selected_indices

@register_sampler("margin")
def margin_sampling(**kwargs) -> List[int]:
    max_budget = kwargs["max_budget"]
    train_dataset = kwargs["train_dataset"]
    oracle_labels = kwargs["oracle_labels"]
    train_loader = kwargs["train_loader"]
    num_epochs = kwargs["num_epochs"]
    learn_rate = kwargs["learn_rate"]
    r = kwargs["r"]
    class_names = kwargs["class_names"]
    seed_worker_fn = kwargs["seed_worker_fn"]
    g_seed = kwargs["g_seed"]
    device = kwargs["device"]
    
    from model import CODAModel, extract_image_embeddings
    from contrastive import train_contrastive
    from load_data import ActiveLearningDataset
    from torch.utils.data import Subset, DataLoader
    
    num_samples = len(train_dataset)
    step_budget = max(1, int(0.2 * max_budget))
    
    selected_indices = []
    unlabeled_indices = list(range(num_samples))
    
    while len(selected_indices) < max_budget:
        current_need = min(step_budget, max_budget - len(selected_indices))
        
        if len(selected_indices) == 0:
            np.random.shuffle(unlabeled_indices)
            chosen = unlabeled_indices[:current_need]
        else:
            selected_labels = oracle_labels[selected_indices]
            train_subset = Subset(train_dataset, selected_indices)
            al_dataset = ActiveLearningDataset(train_subset, selected_labels)
            al_loader = DataLoader(
                al_dataset, batch_size=min(32, len(selected_indices)), 
                shuffle=True, num_workers=0,
                worker_init_fn=seed_worker_fn, generator=g_seed
            )
            
            model = CODAModel(num_classes=len(class_names), r=r, lora_alpha=r*2).to(device)
            model = train_contrastive(
                model=model, labeled_loader=al_loader, 
                num_epochs=num_epochs, learn_rate=learn_rate, 
                device=device, verbose=False
            )
            
            _, _, probs = extract_image_embeddings(train_loader, model, device=device)
            unlabeled_probs = probs[unlabeled_indices]
            
            sorted_probs = np.sort(unlabeled_probs, axis=1)
            margin_scores = sorted_probs[:, -1] - sorted_probs[:, -2]
            
            best_local_indices = np.argsort(margin_scores)[:current_need]
            chosen = [unlabeled_indices[idx] for idx in best_local_indices]
            
            del model, al_dataset, al_loader, probs, unlabeled_probs, margin_scores
            clear_memory()
            
        selected_indices.extend(chosen)
        chosen_set = set(chosen)
        unlabeled_indices = [idx for idx in unlabeled_indices if idx not in chosen_set]
        
    return selected_indices

@register_sampler("coreset")
def coreset_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    max_budget = kwargs["max_budget"]
    device = kwargs["device"]
    chunk_size = kwargs["chunk_size"]

    num_samples = image_embeddings.shape[0]
    
    unlabeled_tensor = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    unlabeled_tensor = F.normalize(unlabeled_tensor, p=2, dim=1)
    
    selected_indices = []
    
    first_idx = np.random.randint(0, num_samples)
    selected_indices.append(first_idx)
    
    min_distances = torch.full((num_samples,), float("inf"), device=device)
    
    for _ in tqdm(range(max_budget - 1), desc="CoreSet Selection"):
        latest_idx = selected_indices[-1]
        latest_feature = unlabeled_tensor[latest_idx].unsqueeze(0) 
        
        for chunk_start in range(0, num_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_samples)
            chunk_tensor = unlabeled_tensor[chunk_start:chunk_end]
            
            sim_chunk = torch.matmul(chunk_tensor, latest_feature.T).squeeze(1)
            dist_chunk = 1.0 - sim_chunk
            
            min_distances[chunk_start:chunk_end] = torch.minimum(
                min_distances[chunk_start:chunk_end], 
                dist_chunk
            )
            
        min_distances[selected_indices] = -1.0
        
        furthest_idx = torch.argmax(min_distances).item()
        selected_indices.append(furthest_idx)
        
        clear_memory()
        
    del unlabeled_tensor, min_distances
    clear_memory()
    
    return selected_indices

@register_sampler("badge")
def badge_sampling(**kwargs) -> List[int]:
    max_budget = kwargs["max_budget"]
    train_dataset = kwargs["train_dataset"]
    oracle_labels = kwargs["oracle_labels"]
    train_loader = kwargs["train_loader"]
    num_epochs = kwargs["num_epochs"]
    learn_rate = kwargs["learn_rate"]
    r = kwargs["r"]
    class_names = kwargs["class_names"]
    seed_worker_fn = kwargs["seed_worker_fn"]
    g_seed = kwargs["g_seed"]
    device = kwargs["device"]
    
    from model import CODAModel, extract_image_embeddings
    from contrastive import train_contrastive
    from load_data import ActiveLearningDataset
    from torch.utils.data import Subset, DataLoader
    
    num_samples = len(train_dataset)
    step_budget = max(1, int(0.2 * max_budget))
    
    selected_indices = []
    unlabeled_indices = list(range(num_samples))
    
    while len(selected_indices) < max_budget:
        current_need = min(step_budget, max_budget - len(selected_indices))
        
        if len(selected_indices) == 0:
            np.random.shuffle(unlabeled_indices)
            chosen = unlabeled_indices[:current_need]
        else:
            selected_labels = oracle_labels[selected_indices]
            train_subset = Subset(train_dataset, selected_indices)
            al_dataset = ActiveLearningDataset(train_subset, selected_labels)
            al_loader = DataLoader(
                al_dataset, batch_size=min(32, len(selected_indices)), 
                shuffle=True, num_workers=0,
                worker_init_fn=seed_worker_fn, generator=g_seed
            )
            
            model = CODAModel(num_classes=len(class_names), r=r, lora_alpha=r*2).to(device)
            model = train_contrastive(
                model=model, labeled_loader=al_loader, 
                num_epochs=num_epochs, learn_rate=learn_rate, 
                device=device, verbose=False
            )
            
            _, embeddings_proj, probs = extract_image_embeddings(train_loader, model, device=device)
            
            unlabeled_embs = torch.tensor(embeddings_proj[unlabeled_indices], device=device, dtype=torch.float32)
            unlabeled_probs = torch.tensor(probs[unlabeled_indices], device=device, dtype=torch.float32)
            
            preds = torch.argmax(unlabeled_probs, dim=1)
            residuals = -unlabeled_probs
            residuals.scatter_add_(1, preds.unsqueeze(1), torch.ones_like(preds.unsqueeze(1)))
            
            emb_norms_sq = torch.sum(unlabeled_embs ** 2, dim=1)
            res_norms_sq = torch.sum(residuals ** 2, dim=1)
            
            chosen_local = []
            D2 = torch.full((len(unlabeled_indices),), float("inf"), device=device)
            
            for _ in range(current_need):
                if len(chosen_local) == 0:
                    grad_magnitudes = emb_norms_sq * res_norms_sq
                    ind = torch.argmax(grad_magnitudes).item()
                else:
                    prob_dist = D2 / torch.sum(D2)
                    ind = torch.multinomial(prob_dist, 1).item()
                    
                chosen_local.append(ind)
                
                c_emb = unlabeled_embs[ind]
                c_res = residuals[ind]
                c_emb_norm_sq = emb_norms_sq[ind]
                c_res_norm_sq = res_norms_sq[ind]
                
                term1 = res_norms_sq * emb_norms_sq
                term2 = c_res_norm_sq * c_emb_norm_sq
                dot_res = torch.matmul(residuals, c_res)
                dot_emb = torch.matmul(unlabeled_embs, c_emb)
                
                dist_sq = term1 + term2 - 2 * dot_res * dot_emb
                dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))
                
                D2 = torch.minimum(D2, dist)
                D2[chosen_local] = 0.0 
                
            chosen = [unlabeled_indices[idx] for idx in chosen_local]
            
            del model, al_dataset, al_loader, unlabeled_embs, unlabeled_probs, residuals, D2
            clear_memory()
            
        selected_indices.extend(chosen)
        chosen_set = set(chosen)
        unlabeled_indices = [idx for idx in unlabeled_indices if idx not in chosen_set]
        
    return selected_indices

@register_sampler("typiclust")
def typiclust_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    max_budget = kwargs["max_budget"]
    device = kwargs["device"]
    chunk_size = kwargs["chunk_size"]

    num_samples = image_embeddings.shape[0]
    K_NN = kwargs.get("k_nn", 20)
    
    features_tensor = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features_tensor = F.normalize(features_tensor, p=2, dim=1)
    
    typicality = torch.zeros(num_samples, device=device)
    
    for chunk_start in range(0, num_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk = features_tensor[chunk_start:chunk_end]
        
        sim_matrix = torch.matmul(chunk, features_tensor.T)
        dist_matrix = 1.0 - sim_matrix
        
        k_val = min(K_NN + 1, num_samples)
        topk_dist, _ = torch.topk(dist_matrix, k=k_val, dim=1, largest=False)
        
        mean_dist = topk_dist[:, 1:].mean(dim=1)
        
        typicality[chunk_start:chunk_end] = 1.0 / (mean_dist + 1e-5)
        
        del sim_matrix, dist_matrix, topk_dist
        clear_memory()
        
    typicality = typicality.cpu().numpy()
    features_np = features_tensor.cpu().numpy()
    
    num_clusters = max_budget
    
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters, n_init="auto", random_state=42)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000, n_init="auto", random_state=42)
        
    cluster_ids = km.fit_predict(features_np)
    
    cluster_sizes = np.bincount(cluster_ids, minlength=num_clusters)
    
    sorted_clusters = np.argsort(cluster_sizes)[::-1]
    valid_clusters = [c for c in sorted_clusters if cluster_sizes[c] > 0]
    
    selected_indices = []
    selected_set = set()
    
    i = 0
    with tqdm(total=max_budget, desc="TypiClust Selection") as pbar:
        while len(selected_indices) < max_budget:
            cluster = valid_clusters[i % len(valid_clusters)]
            
            in_cluster_mask = (cluster_ids == cluster)
            in_cluster_indices = np.where(in_cluster_mask)[0]
            available_indices = [idx for idx in in_cluster_indices if idx not in selected_set]
            
            if len(available_indices) > 0:
                cluster_typ = typicality[available_indices]
                best_local_idx = np.argmax(cluster_typ)
                best_global_idx = available_indices[best_local_idx]
                
                selected_indices.append(best_global_idx)
                selected_set.add(best_global_idx)
                
                pbar.update(1)
                
            i += 1
            if i > len(valid_clusters) * max_budget:
                break
                
    if len(selected_indices) < max_budget:
        remaining_unlabeled = list(set(range(num_samples)) - selected_set)
        missing = max_budget - len(selected_indices)
        fallback_indices = np.random.choice(remaining_unlabeled, missing, replace=False)
        selected_indices.extend(fallback_indices.tolist())
        
    del features_tensor, typicality
    clear_memory()
    
    return selected_indices

@register_sampler("activeft")
def activeft_sampling(**kwargs) -> List[int]:
    image_embeddings = kwargs["image_embeddings"]
    max_budget = kwargs["max_budget"]
    device = kwargs["device"]
    
    num_samples = image_embeddings.shape[0]
    
    lr = kwargs.get("lr", 0.01) 
    tau = kwargs.get("temperature", 0.07)
    iterations = kwargs.get("iterations", 100)
    lambda_reg = kwargs.get("balance", 1.0)
    
    features = torch.tensor(image_embeddings, device=device, dtype=torch.float32)
    features = F.normalize(features, p=2, dim=1)
    
    init_indices = np.random.choice(num_samples, max_budget, replace=False)
    theta = features[init_indices].detach().clone()
    theta.requires_grad_(True)
    
    optimizer = torch.optim.Adam([theta], lr=lr)
    
    for _ in tqdm(range(iterations), desc="ActiveFT Optimization"):
        optimizer.zero_grad()
        
        theta_norm = F.normalize(theta, p=2, dim=1)
        
        sim_matrix = torch.matmul(features, theta_norm.t()) / tau
        max_sim, _ = torch.max(sim_matrix, dim=1)
        loss_dist = -torch.mean(max_sim)
        
        theta_sim = torch.matmul(theta_norm, theta_norm.t()) / tau
        
        mask = ~torch.eye(max_budget, device=device, dtype=torch.bool)
        theta_sim_filtered = theta_sim[mask].view(max_budget, max_budget - 1)
        
        loss_reg = torch.mean(torch.log(torch.sum(torch.exp(theta_sim_filtered), dim=1)))
        
        loss = loss_dist + lambda_reg * loss_reg
        
        loss.backward()
        optimizer.step()

    selected_indices = set()
    with torch.no_grad():
        theta_final = F.normalize(theta, p=2, dim=1)
        
        dist_to_real = torch.matmul(theta_final, features.t())
        
        _, ids_sort = torch.sort(dist_to_real, dim=1, descending=True)
        ids_sort = ids_sort.cpu().numpy()
        
        for i in tqdm(range(max_budget), desc="ActiveFT Selection"):
            for j in range(num_samples):
                candidate_idx = ids_sort[i, j]
                if candidate_idx not in selected_indices:
                    selected_indices.add(candidate_idx)
                    break 
                    
    del features, theta, sim_matrix, theta_sim, mask, dist_to_real, ids_sort
    clear_memory()
    
    return list(selected_indices)

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
    
    sim_matrix_0 = torch.matmul(img_norm, txt_norm.T)
    
    probs_for_margin = F.softmax(sim_matrix_0 / 0.05, dim=1)
    
    sorted_probs, _ = torch.sort(probs_for_margin, dim=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    
    predicted_classes = torch.argmax(probs_for_margin, dim=1)
    one_hot_preds = torch.zeros_like(probs_for_margin)
    one_hot_preds.scatter_(1, predicted_classes.unsqueeze(1), 1.0)
    
    M = 0.5 * (probs_for_margin + one_hot_preds)
    
    kl_p_m = torch.sum(probs_for_margin * torch.log(probs_for_margin / M + 1e-10), dim=1)
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
                
            objective_scores = (1 - alpha) * normalized_coverage + (alpha) * chunk_uncert
            
            for selected_idx in selected_indices:
                if chunk_start <= selected_idx < chunk_end:
                    local_idx_in_chunk = selected_idx - chunk_start
                    objective_scores[local_idx_in_chunk] = -float("inf")
                    
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