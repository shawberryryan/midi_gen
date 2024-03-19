import os

# specify the path of the directory containing the images
directory_path = "input/midipics"

# create an empty dictionary to store the counts of parts for each image name
counts = {}

# iterate over all the files in the directory
for filename in os.listdir(directory_path):
    # check if the file is a PNG image
    if filename.endswith(".png"):
        # extract the image name from the file name
        image_name = filename.rsplit("_", 1)[0]
        # increment the count for this image name in the dictionary
        if image_name in counts:
            counts[image_name] += 1
        else:
            counts[image_name] = 1

# find the image name with the least number of parts
min_count = float('inf')
min_name = None
for name, count in counts.items():
    if count < min_count:
        min_count = count
        min_name = name

# print the result
print(f"The image name with the least number of parts is {min_name} with {min_count} parts.")