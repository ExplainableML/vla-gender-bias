# Make data folder
mkdir -p data-scratch
cd data-scratch

# Download FairFace
# rm -r fairface
mkdir -p fairface
cd fairface
gdown 1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86
gdown 1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL
mkdir margin025
mkdir margin125
mv fairface-img-margin025-trainval.zip margin025/
mv fairface-img-margin125-trainval.zip margin125/
cd margin025
unzip fairface-img-margin025-trainval.zip
rm fairface-img-margin025-trainval.zip
cd ..
cd margin125
unzip fairface-img-margin125-trainval.zip
rm fairface-img-margin125-trainval.zip
cd ..

gdown 1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH
gdown 1wOdja-ezstMEp81tX1a-EYkFebev4h7D
cd ..

cd ..
python process_fairface.py --path ./data-scratch/fairface/
rm -r data-scratch/fairface
