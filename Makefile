# Peter Yang
# 2021-05-18
# 
# This make file is for lipohypertrophy classification on ultrasound images
# 
# Usage:
# make all

all : results/test_accuracy.csv results/test_confusion_matrix.csv results/test_recall.csv

# train validation test split
data/train/ data/val/ data/test/ : src/image_folder_split.py raw/ negative/
	python src/image_folder_split.py --positive_dir=raw/ --negative_dir=negative/

# model training
models/cnn_model.pt : data/train/ data/val/ src/model.py src/cnn_utils.py
	python src/model.py --train_dir=data/train/ --valid_dir=data/val/

# model evaluation
results/test_accuracy.csv results/test_confusion_matrix.csv results/test_recall.csv : models/cnn_model.pt src/evaluation.py data/test/ src/cnn_utils.py
	python src/evaluation.py --test_dir=data/test/

clean :
	rm -rf data/
	rm -rf results/*
	rm -rf models/*
