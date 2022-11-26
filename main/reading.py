import os

# Gets the current working directory
current_directory = os.getcwd()
# Go up one directory from working directory
os.chdir("..")
# Get a tuple of all the directories in the folder
o = [os.path.join(current_directory, o) for o in os.listdir(current_directory)
     if os.path.isdir(os.path.join(current_directory, o))]
# Search the tuple for the directory you want and open the file
for item in o:
    if os.path.exists(item + filename):
        file = item + filename
