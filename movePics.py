import os
import shutil


source_folder = 'input/midipics'
destination_folder = 'input/halfpics'

# Check if destination folder exists, and create it if necessary
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Use os.listdir() to get a list of all files in the source folder
files = os.listdir(source_folder)

# Loop through each file and move it if it's a PNG file
for file in files:
    if file.endswith(".png"):
        source_file = os.path.join(source_folder, file)
        destination_file = os.path.join(destination_folder, file)
        shutil.move(source_file, destination_file)