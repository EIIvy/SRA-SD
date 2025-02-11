import argparse
import os
import math
import torch
from syngen_add_relation import SynGenDiffusionPipeline_add_relation
from train_relations_ARB import train_list

def main(args):

    torch.autograd.set_detect_anomaly(True)
    prompt = args.prompt
    seed = args.seed
    output_directory = args.output_directory 
    model_path = args.model_path
    step_size = args.step_size
    attn_res = args.attn_res
    include_entities = args.include_entities
    pipe = load_model(args, model_path, include_entities)
    vae, unet, scheduler, text_encoder, tokenizer, image_processor = pipe.get_models()
    train_list(prompt, vae, unet, scheduler, text_encoder, tokenizer, image_processor, args)
    


def load_model(args, model_path, include_entities):
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    pipe = SynGenDiffusionPipeline_add_relation.from_pretrained(model_path, include_entities=include_entities, args=args).to(device)

    return pipe



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
        default=" "
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
        '--batch_size',
        type=int,
        default=40
    )

    parser.add_argument(
        '--output_directory',
        type=str,
        default='save_weight'
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
        "--data_folder",
        type=str,
        default="images",
        help="Path to your dataset",
    )

    parser.add_argument(
        "--train_data_path",
        type=str,
        default="training_data.json",
        help="Path to your dataset",
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
        default=50,
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
