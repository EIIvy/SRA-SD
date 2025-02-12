SRA-SD
===
This is a two-stage process. First, we need to train HGNN model, then we combain the trained HGNN model with FCL to generate images
### 1. Training HGNN model
python training_model/train.py --device " " --model_path ""
### 2. Generating images
python generate/generate_images.py --device "" --file_path "" --model_path "" --output_directory ""
