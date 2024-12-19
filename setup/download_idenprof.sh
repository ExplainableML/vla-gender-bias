# Make data folder
mkdir -p data-scratch
cd data-scratch

# Download IdenProf
mkdir -p idenprof
cd idenprof
wget https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-jpg.zip
unzip idenprof-jpg.zip
rm idenprof-jpg.zip
cd ..

cd ..

python process_idenprof.py --idenprof-path ./data-scratch/idenprof/idenprof/ --output-path ./data-scratch/idenprof/
