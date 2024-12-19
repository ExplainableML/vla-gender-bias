# Make data folder
mkdir -p data-scratch
cd data-scratch

# Download Phase
mkdir -p phase
cd phase
gdown 1-GGvJjkIrDjQlSnoPmOKY3KqCeeGyDGT
gdown 1COJCqMj4Jdj7Vu_87uJ4tKaQZyGISMTm
unzip phase_annotations.zip
unzip phase_images.zip
rm phase_annotations.zip
rm phase_images.zip
rm annotators.csv

cd ..
cd ..

python process_phase.py --path ./data-scratch/phase/
rm -r data-scratch/phase/train/
rm -r data-scratch/phase/val/
rm data-scratch/phase/*.json
rm data-scratch/phase/README
