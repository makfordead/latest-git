import os, shutil

original_dataset_dir = "/home/kapilkodwani/Music/AI_DEEP_Learning_Project"


dataset_directory = "/home/kapilkodwani/Music/AI_DEEP_Learning_Project/dataset"
os.mkdir(dataset_directory)

train_dir = os.path.join(dataset_directory,"train")
test_dir = os.path.join(dataset_directory,"test")

os.mkdir(train_dir)
os.mkdir(test_dir)
