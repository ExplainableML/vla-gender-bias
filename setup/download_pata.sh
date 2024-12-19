# Make data folder
mkdir -p data-scratch
cd data-scratch

mkdir -p pata
cd pata
wget https://raw.githubusercontent.com/pata-fairness/pata_dataset/main/pata_fairness.files.lst
cd ..

cd ..

python process_pata.py --path ./data-scratch/pata/