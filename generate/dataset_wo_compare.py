from GNN import create_hg_A_ARB_self_loop, create_hg_A_ARB_B_self_loop, create_hg_ARB_self_loop
from torch.utils.data import Dataset
import random
import os
from PIL import Image
import torch




class HeteGraphDataset:
    def __init__(self, image_processor, tokenizer, transform=None):
          
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.transform = transform
    
    def get_eot_postiton(self, sentence):
        untruncated_ids = self.tokenizer([sentence], padding="longest", return_tensors="pt").input_ids
        eot_pos = len(untruncated_ids[0])-1
        return eot_pos


    def get_word_embeddings(self, embedding, word, device):

        txt_id = self.tokenizer(
                [word],
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)

        word_embedding = embedding(txt_id[0, 1]) # 跳过了[CLS], 只获取单词的id
        word_embedding = word_embedding.unsqueeze(0)
        return word_embedding
    
    def get_sentence_eot_embeddings(self, text_encoder, sentence, device):

        txt_id = self.tokenizer(
                [sentence],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)
        
        eot_pos = self.get_eot_postiton(sentence)
        sentence_embedding = text_encoder(txt_id)[0].to(device)[0]
        
        eot_embedding = sentence_embedding[eot_pos]
        eot_embedding = eot_embedding.unsqueeze(0)
        return eot_embedding, sentence_embedding.unsqueeze(0), eot_pos

    def sample(self, A_word, R_word, B_word, ARB_sentence, text_encoder, embeddings, device, self_loop=True, add_noise_on_A=True):
        
        def get_object_sentence(objs):
            
            obj_sentences = "this is a photo of {obj}"
            
            
            return obj_sentences
        
        A_word_embedding = self.get_word_embeddings(embeddings, A_word, device)  #[101, 1, 768]
        R_word_embedding = self.get_word_embeddings(embeddings, R_word, device)  #[101, 1, 768]
        B_word_embedding = self.get_word_embeddings(embeddings, B_word, device)  #[101, 1l 768]
        
        ARB_eot, ARB_sentence_embedding, ARB_eot_pos = self.get_sentence_eot_embeddings(text_encoder, ARB_sentence, device)
         
        A_eot_embedding, A_sentence_embedding, A_eot_pos = self.get_sentence_eot_embeddings(text_encoder, get_object_sentence(A_word), device)
        B_eot_embedding, B_sentence_embedding, B_eot_pos = self.get_sentence_eot_embeddings(text_encoder, get_object_sentence(B_word), device)
        
        # if self_loop:
            
        h_graph_ARB, node_features_ARB = create_hg_A_ARB_B_self_loop(A_word_embedding, R_word_embedding, B_word_embedding, A_eot_embedding, ARB_eot, B_eot_embedding, device)
        
        # else:
        #     h_graph_ARB, node_features_ARB = create_hg_A_ARB(A_word_embedding, R_word_embedding, B_word_embedding, A_eot_embedding, ARB_eot, device)
        #     h_graph_BRA, node_features_BRA = create_hg_A_ARB(B_word_embedding, R_word_embedding, A_word_embedding, B_eot_embedding, BRA_eot, device)
            
        

        return ARB_sentence_embedding, ARB_eot_pos, \
                h_graph_ARB, node_features_ARB