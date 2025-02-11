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



def generate_sample(text_encoder, gnn_model, hete_dataset, data, args):

    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    text_encoder.text_model.requires_grad_(False)
    embeddings = text_encoder.get_input_embeddings()
    
   
    A = data['objectA']
    R = data['rel']
    B = data['objectB']
    ARB_prompt = data['text']
    
    text_encoder.eval()


    with torch.no_grad():
        
        ARB_sentence_embedding, ARB_eot_pos, \
                h_graph_ARB, node_features_ARB \
        =  hete_dataset.sample(A, R, B, ARB_prompt, text_encoder, embeddings, device)

        text_encoder.text_model.requires_grad_(False)
        delta_norm = gnn_model(h_graph_ARB, node_features_ARB)['eot'][1]
        #delta_inver = gnn_model(h_graph_BRA, node_features_BRA)['eot'][1]
        ARB_sentence_embedding[0, ARB_eot_pos] += delta_norm * args.weight

        return ARB_sentence_embedding 


def multi_generate_ARB_sample(text_encoder, gnn_model, hete_dataset, data, args):

    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    text_encoder.text_model.requires_grad_(False)
    embeddings = text_encoder.get_input_embeddings()
    
    structure_triples = data['triple']
    length = len(structure_triples)
    text_encoder.eval()

    ARB_prompt = data['text']

    delta_norms = []
    with torch.no_grad():

        for stru_triple in structure_triples:
            A = stru_triple[0]
            R = stru_triple[1]
            B = stru_triple[2]
            
            ARB_sentence_embedding, ARB_eot_pos, \
                    h_graph_ARB, node_features_ARB \
            =  hete_dataset.sample(A, R, B, ARB_prompt, text_encoder, embeddings, device)

            text_encoder.text_model.requires_grad_(False)
            
            delta_norm = gnn_model(h_graph_ARB, node_features_ARB)['eot'][1]
            #delta_inver = gnn_model(h_graph_BRA, node_features_BRA)['eot'][1]
            delta_norms.append(delta_norm)
            
        for del_nor in delta_norms:

            #ARB_sentence_embedding[0, ARB_eot_pos] += del_nor * args.weight * 1/length
            ARB_sentence_embedding[0, ARB_eot_pos] += del_nor * args.weight

    return ARB_sentence_embedding 