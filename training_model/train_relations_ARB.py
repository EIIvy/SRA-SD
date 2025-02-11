import torch
import os
import math
import numpy as np
import torch.nn.functional as F
import random
import json
from GNN import RGAT, HeteroGraphSAGE, RGCN
from dataset_wo_compare_ARB import HeteGraphDataset
from tqdm import tqdm

def importance_sampling_fn(t, max_t, alpha):
    """
    Importance Sampling Function f(t)
    Borrow from https://github.com/ziqihuangg/ReVersion
    """
  
    return 1 / max_t * (1 - alpha * math.cos(math.pi * t / max_t))


def get_list_paired_latent(vae, img1, device):
    latents0 = [vae.encode(img.to(device)).latent_dist.sample().detach() * vae.config.scaling_factor for img in img1]    #latent_dist 表示编码过程中生成的潜在分布（通常是一个正态分布的均值和方差)  sample:从 latent_dist 这个潜在分布中进行采样
    latents0 = torch.cat(latents0, dim=0)

    return latents0



def get_list_loss_of_ARB(vae, gnn_model, noise_scheduler, unet, img1,
                         sentence1_embedding, sentence1_eot_pos, sentence1_type,
                         h_graph1, node_features1,
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

    for i in range(len(h_graph1)):
        if sentence1_type == "r":
            delta1 = gnn_model(h_graph1[i], node_features1[i])['eot'][1]   #[768]
            # for object, the eot is the first one 
        # for object, the eot is the first one 
        
        sentence1_embedding[i, sentence1_eot_pos[i]] = sentence1_embedding[i, sentence1_eot_pos[i]] + delta1   # [101, 77, 768]
        #print("computing graph---",i)
    
    for i, eot_pos in enumerate(sentence1_eot_pos):
        sentence1_embedding[i, eot_pos] = sentence1_embedding[i, eot_pos] + delta1[i]

    
    
    # a
    latents_norm_a  = get_list_paired_latent(vae, img1, device)  #[202,4,64,64]
    noise_a = torch.randn_like(latents_norm_a).to(device)
    
    noisy_latents_norm_a = noise_scheduler.add_noise(
                        latents_norm_a, noise_a, timesteps)   #[101, 2, 4, 64, 64]

   
    model_pred_a = unet(noisy_latents_norm_a, timesteps, sentence1_embedding).sample
    
    denoise_loss_a = F.mse_loss(
                        model_pred_a.float(), noise_a.float(), reduction="mean")
    
    #cosine_sim_eot = F.cosine_similarity(sentence1_embedding[0][sentence1_eot_pos], sentence2_embedding[0][sentence2_eot_pos], dim=0)
    
    loss = 10*denoise_loss_a
    
    return loss, denoise_loss_a


def process_data(base_path, batch_datas):
    A_names = []
    R_names = []
    B_names = []
    ARB_prompts = []
    ARB_paths = []

    for b_data in batch_datas:
        A_names.append(b_data['objectA'])
        B_names.append(b_data['objectB'])
        R_names.append(b_data['rel'])
        ARB_prompts.append(b_data['text'])
        ARB_paths.append(os.path.join(base_path, b_data['id']+'.png'))

    return A_names, R_names, B_names, ARB_prompts, ARB_paths


def train_list(prompt, vae, unet, noise_scheduler, text_encoder, tokenizer, image_processor, args):
# def train(device=1, num_sample=50, save_folder="./generation_result", data_folder="./RRdataset-v1/inside"):
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.requires_grad_(False)
    
    gnn_model = RGAT(768, args.hidden_size, 768, ['link_oe', 'link_or', 'link_re', 'link_ro', 'link_oo']).to(device)
    #gnn_model = HeteroGraphSAGE(768, args.hidden_size, 768, ['link_oe', 'link_or', 'link_re', 'link_ro', 'link_oo'], aggregator_type='mean').to(device)
    #gnn_model = RGCN(768, args.hidden_size, 768, ['link_oe', 'link_or', 'link_re', 'link_ro', 'link_oo']).to(device)
    gnn_model.requires_grad_(True)
    embeddings = text_encoder.get_input_embeddings()

    optimizer = torch.optim.AdamW(
        gnn_model.parameters(
        ), 
        lr=args.lr, weight_decay=0.01
    )
    
    folder = args.data_folder

    with open(args.train_data_path, 'r') as file:
        all_datas = json.load(file)
    

    hete_dataset = HeteGraphDataset(image_processor, tokenizer)
    
    batch_size = args.batch_size
   
    
    for epoch in tqdm(range(0, args.epochs)):
        
        print(f"epoch {epoch}:")
        
        gnn_model.train()
        text_encoder.eval()
        vae.eval()
        unet.eval()

        random.shuffle(all_datas)
        num_batches = len(all_datas) //  batch_size

        for i in range(num_batches):
            batch_data = all_datas[i*batch_size:(i+1)*batch_size]
            A, R, B, ARB, ARB_path = process_data(folder, batch_data)

            ARB_img, ARB_sentence_embedding, ARB_eot_pos,\
            h_graph_ARB, node_features_ARB\
            = hete_dataset.sample_list(A, R, B, ARB, ARB_path,text_encoder, embeddings, device)
            
            # ARB A
            loss, denoise_loss_a = get_list_loss_of_ARB(vae, gnn_model, noise_scheduler, unet,ARB_img,
                                ARB_sentence_embedding, ARB_eot_pos, "r",
                                h_graph_ARB, node_features_ARB,
                                device, bsz=batch_size)
            print(f"loss: {loss:.3f}, denoise_a: {denoise_loss_a:.3f}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        
        
        

    save_path = os.path.join(args.output_directory, "pretrain_ARB")
    

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # plt.savefig(os.path.join(save_path, f'ARB_BRA_sim.png'))
    
    torch.save(gnn_model.state_dict(), os.path.join(save_path, f"gat_R_A_ARB_B_RGAT"))

    print("train_GNN_done!!!", save_path)
    exit()