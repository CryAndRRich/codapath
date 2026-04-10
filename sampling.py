from typing import List
from tqdm import tqdm

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
from scipy.spatial.distance import jensenshannon
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

@register_sampler("codapath")
def coda_sampling(image_embeddings: np.ndarray, 
                  text_embeddings: np.ndarray, 
                  max_budget: int, 
                  alpha: float, 
                  device: torch.device, 
                  chunk_size: int = 10000) -> List[int]:
    
    num_samples = image_embeddings.shape[0]
    
    sim_matrix_0 = cosine_similarity(image_embeddings, text_embeddings)
    probs_for_margin = softmax(sim_matrix_0 / 0.05, axis=1) 
        
    sorted_probs = np.sort(probs_for_margin, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    
    predicted_classes = np.argmax(probs_for_margin, axis=1)
    one_hot_preds = np.zeros_like(probs_for_margin)
    one_hot_preds[np.arange(num_samples), predicted_classes] = 1.0
    
    jsd_scores = jensenshannon(probs_for_margin, one_hot_preds, axis=1) ** 2
    uncertainty_weights = (1.0 - margins) * (1.0 + jsd_scores)
    
    if uncertainty_weights.max() > 0:
        uncertainty_weights = uncertainty_weights / uncertainty_weights.max()

    queries_idx = []
    
    unlabeled_tensor = torch.tensor(image_embeddings, device=device)
    unlabeled_tensor = F.normalize(unlabeled_tensor, p=2, dim=1)
    uncertainty_tensor = torch.tensor(uncertainty_weights, device=device)
    
    max_sim_to_S = torch.full((num_samples, 1), -float('inf'), device=device)
    
    for step in tqdm(range(max_budget)):
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
            
            for selected_idx in queries_idx:
                if chunk_start <= selected_idx < chunk_end:
                    local_idx_in_chunk = selected_idx - chunk_start
                    objective_scores[local_idx_in_chunk] = -float('inf')
                    
            local_best_idx = torch.argmax(objective_scores).item()
            local_best_score = objective_scores[local_best_idx].item()
            
            if local_best_score > best_objective_score or best_sim_column is None:
                best_objective_score = local_best_score
                best_candidate_global = chunk_start + local_best_idx
                best_sim_column = sim_chunk[:, local_best_idx].reshape(-1, 1).clone() 
                
            del chunk_tensor, chunk_uncert, sim_chunk, gain_chunk, coverage_gain_chunk, objective_scores
            clear_memory()
            
        if best_sim_column is not None:
            queries_idx.append(best_candidate_global)
            max_sim_to_S = torch.maximum(max_sim_to_S, best_sim_column)
            del best_sim_column
            clear_memory()
        else:
            print("Cannot find a valid candidate, stopping early")
            break
            
    del unlabeled_tensor, uncertainty_tensor, max_sim_to_S
    clear_memory()
    
    return queries_idx