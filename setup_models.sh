# Save current directory
wd=$(pwd);

echo "Downloading Huggingface Models";
mkdir -p $1/vlm-models;
cd $1/vlm-models;

# Initilize git lfs
git lfs install;

# Download models
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
git clone https://huggingface.co/liuhaotian/llava-v1.5-13b
git clone https://huggingface.co/zhiqings/LLaVA-RLHF-7b-v1.5-224
git clone https://huggingface.co/zhiqings/LLaVA-RLHF-13b-v1.5-336
git clone https://huggingface.co/zxmonent/llava-phi
git clone https://huggingface.co/mtgv/MobileVLM_V2-1.7B
git clone https://huggingface.co/mtgv/MobileVLM_V2-3B
git clone https://huggingface.co/mtgv/MobileVLM_V2-7B
git clone https://huggingface.co/Qwen/Qwen-VL-Chat
git clone https://huggingface.co/SkunkworksAI/BakLLaVA-1
git clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b
git clone https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b
git clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b
git clone https://huggingface.co/liuhaotian/llava-v1.6-34b
git clone https://huggingface.co/OpenGVLab/InternVL2-1B
git clone https://huggingface.co/OpenGVLab/InternVL2-2B
git clone https://huggingface.co/OpenGVLab/InternVL2-4B
git clone https://huggingface.co/OpenGVLab/InternVL2-8B
git clone https://huggingface.co/OpenGVLab/InternVL2-26B
git clone https://huggingface.co/OpenGVLab/InternVL2-40B
git clone https://huggingface.co/BAAI/Bunny-v1_0-3B
git clone https://huggingface.co/BAAI/Bunny-v1_1-4B
git clone https://huggingface.co/BAAI/Bunny-v1_1-Llama-3-8B-V
git clone https://huggingface.co/microsoft/Phi-3.5-vision-instruct

# Download phi
git clone https://huggingface.co/susnato/phi-2
cd phi-2;
git checkout f3d2d295eea90f7ff6515f2d7649eb0ea0137a14;
cd ..;

# Download bert base uncased
git clone https://huggingface.co/google-bert/bert-base-uncased

# Change back to original directory
cd $wd;

# Update the model config
cd configs;
which python;
python update_model_config.py --path $1;

# Finish
echo "Download complete";
cd ..;
