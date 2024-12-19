# Make data folder
mkdir -p data-scratch
cd data-scratch

mkdir -p miap
cd miap
# Get bounding boxes
# wget https://storage.googleapis.com/openimages/open_images_extended_miap/open_images_extended_miap_boxes_train.csv
wget https://storage.googleapis.com/openimages/open_images_extended_miap/open_images_extended_miap_boxes_val.csv
wget https://storage.googleapis.com/openimages/open_images_extended_miap/open_images_extended_miap_boxes_test.csv
# # Get images
# wget https://storage.googleapis.com/openimages/open_images_extended_miap/open_images_extended_miap_images_train.lst
wget https://storage.googleapis.com/openimages/open_images_extended_miap/open_images_extended_miap_images_val.lst
wget https://storage.googleapis.com/openimages/open_images_extended_miap/open_images_extended_miap_images_test.lst
# # Get downloader
wget https://raw.githubusercontent.com/openimages/dataset/master/downloader.py
# Download images
# mkdir -p train
# python downloader.py open_images_extended_miap_images_train.lst --download_folder "./train/" --num_processes=5
mkdir -p val
python downloader.py open_images_extended_miap_images_val.lst --download_folder "./val/" --num_processes=5
mkdir -p test
python downloader.py open_images_extended_miap_images_test.lst --download_folder "./test/" --num_processes=5
cd ..

cd ..
python process_miap.py --path ./data-scratch/miap/

# rm -r ./data-scratch/train
rm -r ./data-scratch/miap/val
rm -r ./data-scratch/miap/test
# rm ./data-scratch/miap/open_images_extended_miap_boxes_train.csv
rm ./data-scratch/miap/open_images_extended_miap_boxes_val.csv
rm ./data-scratch/miap/open_images_extended_miap_boxes_test.csv
rm ./data-scratch/miap/downloader.py