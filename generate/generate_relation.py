import torch
import os
import math
import numpy as np
import torch.nn.functional as F
import random
import json
from GNN import RGAT
from dataset_wo_compare import HeteGraphDataset


def importance_sampling_fn(t, max_t, alpha):
    """
    Importance Sampling Function f(t)
    Borrow from https://github.com/ziqihuangg/ReVersion
    """
  
    return 1 / max_t * (1 - alpha * math.cos(math.pi * t / max_t))
    
def get_paired_latent(vae, img1, img2, device, bsz=1):
    latents0 = vae.encode(img1.to(device)).latent_dist.sample().detach()    #latent_dist 表示编码过程中生成的潜在分布（通常是一个正态分布的均值和方差)  sample:从 latent_dist 这个潜在分布中进行采样
    latents0 = latents0 * vae.config.scaling_factor 
    latents0 = torch.cat([latents0] * bsz)
    
    latents1 = vae.encode(img2.to(device)).latent_dist.sample().detach()
    latents1 = latents1 * vae.config.scaling_factor
    latents1 = torch.cat([latents1] * bsz)
    
    latents_norm = torch.cat([latents0, latents1]).to(device)
    latents_inver = torch.cat([latents1, latents0]).to(device)
    
    
    
    
    return latents_norm, latents_inver

def get_loss_of_one_pair(vae, gnn_model, noise_scheduler, unet, img1, img2, 
                         sentence1_embedding, sentence1_eot_pos, sentence1_type,
                         sentence2_embedding, sentence2_eot_pos, sentence2_type,
                         h_graph1, node_features1,
                         h_graph2, node_features2,
                         device, bsz=1, importance_sampling=True, noise_weight=0.05):
    
    # The importance sampling is applied, borrowed from https://github.com/ziqihuangg/ReVersion
    if importance_sampling:
            list_of_candidates = [
                x for x in range(noise_scheduler.config.num_train_timesteps)
            ]
            prob_dist = [
                importance_sampling_fn(x,
                                       noise_scheduler.config.num_train_timesteps,
                                       0.5)
                for x in list_of_candidates
            ]
            prob_sum = 0
            for i in prob_dist:
                prob_sum += i
            prob_dist = [x / prob_sum for x in prob_dist]
    
    timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps, (bsz, ),
                    device=device)
    timesteps = timesteps.long()
    timesteps = torch.cat([timesteps, timesteps]).to(device)

    if importance_sampling:
            timesteps = np.random.choice(
                list_of_candidates,
                size=bsz,
                replace=True,
                p=prob_dist)
            timesteps = torch.tensor(timesteps).to(device)
        
    latents_norm, latents_inver = get_paired_latent(vae, img1, img2, device, bsz)
    noise = torch.randn_like(latents_norm).to(device)
    
    noisy_latents_norm = noise_scheduler.add_noise(
                        latents_norm, noise, timesteps)
    noisy_latents_inver = noise_scheduler.add_noise(
                        latents_inver, noise, timesteps)
    
    
    sentence1_embedding = torch.cat([sentence1_embedding] * bsz)
    sentence2_embedding = torch.cat([sentence2_embedding] * bsz)
    
    if sentence1_type == "r":
        delta1 = gnn_model(h_graph1, node_features1)['eot'][1]   #[2, 768]
    else:    
        delta1 = gnn_model(h_graph1, node_features1)['eot'][0]
        noise_eot = torch.randn_like(delta1)
        delta1 += noise_weight * noise_eot
        # for object, the eot is the first one 
    if sentence2_type == "r_a":
        delta2 = gnn_model(h_graph2, node_features2)['eot'][0]
        noise_eot = torch.randn_like(delta2)
        delta2 = delta2 +noise_weight * noise_eot
    elif sentence2_type == "r_b":
        delta2 = gnn_model(h_graph2, node_features2)['eot'][2]
        noise_eot = torch.randn_like(delta2)
        delta2 = delta2 +noise_weight * noise_eot
        # for object, the eot is the first one 
        
    sentence1_embedding[:, sentence1_eot_pos] += delta1
    sentence2_embedding[:, sentence2_eot_pos] += delta2
    
    
    emb_pair = torch.cat([sentence1_embedding, sentence2_embedding])
    
    model_pred = unet(noisy_latents_norm, timesteps, emb_pair).sample
    model_pred_inv = unet(noisy_latents_inver, timesteps, emb_pair).sample
    
    denoise_loss = F.mse_loss(
                        model_pred.float(), noise.float(), reduction="mean")
    neg_loss = F.mse_loss(
                        model_pred_inv.float(), noise.float(), reduction="mean")
    
    cosine_sim_eot = F.cosine_similarity(sentence1_embedding[0][sentence1_eot_pos], sentence2_embedding[0][sentence2_eot_pos], dim=0)
    
    loss = 10*denoise_loss - 2*neg_loss
    
    return loss, denoise_loss, neg_loss, cosine_sim_eot.item()

