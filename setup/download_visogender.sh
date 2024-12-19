# Make data folder
mkdir -p data-scratch
cd data-scratch

# Download visogender
rm -r visogender
mkdir -p visogender
cd visogender
wget https://raw.githubusercontent.com/oxai/visogender/main/data/visogender_data/OO/OO_Visogender_02102023.tsv
wget https://raw.githubusercontent.com/oxai/visogender/main/data/visogender_data/OP/OP_Visogender_11012024.tsv
cd ..

cd ..

python process_visogender.py --path ./data-scratch/visogender/