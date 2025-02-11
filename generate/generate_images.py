import argparse
import os
import math
import torch
from syngen_add_relations_neg3 import SynGenDiffusionPipeline_add_relation
import json
from GNN import RGAT, HeteroGraphSAGE, RGCN
from dataset_wo_compare import HeteGraphDataset

def main(args):
    seed = args.seed
    output_directory = args.output_directory 
    model_path = args.model_path
    step_size = args.step_size
    attn_res = args.attn_res
    include_entities = args.include_entities
    pipe = load_model(args, model_path, include_entities)

    image_processor = pipe.get_image_processor()
    tokenizer = pipe.get_tokenizer()

    hete_dataset = HeteGraphDataset(image_processor, tokenizer)

    #gnn_model = RGAT(768, args.hidden_size, 768, ['link_oe', 'link_or', 'link_re', 'link_ro', 'link_oo']).to(args.device)
    gnn_model = HeteroGraphSAGE(768, args.hidden_size, 768, ['link_oe', 'link_or', 'link_re', 'link_ro', 'link_oo'], aggregator_type='mean').to(args.device)
    #gnn_model = RGCN(768, args.hidden_size, 768, ['link_oe', 'link_or', 'link_re', 'link_ro', 'link_oo']).to(args.device)
    gnn_model.requires_grad_(False)
    gnn_model.load_state_dict(torch.load(os.path.join(args.save_weight_path, f"gat_R_A_ARB_B_SAGE"), map_location=torch.device(f'cuda:{args.device}')))
    
    with open(args.file_path, "r") as json_file:
        datas = json.load(json_file)

    for data in datas:
        prompt = data['text']
        image = generate(args, pipe, prompt, seed, step_size, attn_res, data, hete_dataset, gnn_model)
        save_image(args, image, prompt, seed, output_directory, data)
        
        print("well_done")


def load_model(args, model_path, include_entities):
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    pipe = SynGenDiffusionPipeline_add_relation.from_pretrained(model_path, include_entities=include_entities, args=args).to(device)

    return pipe


def generate(args, pipe, prompt, seed, step_size, attn_res, data, hete_dataset, gnn_model):
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    generator = torch.Generator(device.type).manual_seed(seed)
    result = pipe(prompt=prompt, generator=generator, syngen_step_size=step_size, num_images_per_prompt = args.number_images_per_prompt, 
                  attn_res=(int(math.sqrt(attn_res)), int(math.sqrt(attn_res))), data=data, hete_dataset=hete_dataset, gnn_model=gnn_model)
    return result['images'][0]


def save_image(args, image, prompt, seed, output_directory, data):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)



    file_name = f"{output_directory}/{str(data['id'])}.png"
    image.save(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Cuda id",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="A black dog in a white bag"
    )

    parser.add_argument(
        "--file_path",
        type=str,
        default="dataset/SRD-Bench.json"
    )

    parser.add_argument(
        "--save_weight_path",
        type=str,
        default="training_model/save_weight/pretrain_ARB"
    )


    parser.add_argument(
        '--seed',
        type=int,
        default=1269
    )
    #1924

    parser.add_argument(
        '--number_images_per_prompt',
        type=int,
        default=1
    )

    parser.add_argument(
        '--output_directory',
        type=str,
        default='output_data'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='stable_diffusion_v1_4',
        help='The path to the model (this will download the model if the path doesn\'t exist)'
    )

    parser.add_argument(
        '--step_size',
        type=float,
        default=20.0,
        help='The SynGen step size'
    )

    parser.add_argument(
        "--state",
        type=str,
        default="eval",
        help="train or eval",
    )


    parser.add_argument(
        '--weight',
        type=float,
        default=0.4,
        help='The SynGen step size'
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=384,
        help="The hidden size of graph neural network",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="The training epochs of RRNet",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="The training learning rate",
    )

    parser.add_argument(
        '--attn_res',
        type=int,
        default=256,
        help='The attention resolution (use 256 for SD 1.4, 576 for SD 2.1)'
    )

    parser.add_argument(
        '--include_entities',
        type=bool,
        default=False,
        help='Apply negative-only loss for entities with no modifiers'
    )

    args = parser.parse_args()
    main(args)
