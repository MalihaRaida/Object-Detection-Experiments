import os

import shutil


def list_files_without_extension(directory):
    files = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            # Split name and extension
            # name, _ = os.path.splitext(filename)
            files.append(os.path.join(directory, filename))
    return files

directory_path = "C:\\Users\\USER\\Downloads\\archive\\Stanford_Dogs_Yolov8\\Annotations\\val"
file_names = list_files_without_extension(directory_path)


for filename in file_names:
    new_lines = []
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip("\n")  # remove newline for processing
            if line:
                space_index = line.find(" ")
                if space_index != -1:
                    # Replace up to the first space
                    line = "1" + line[space_index:]
                else:
                    # If no space, replace whole line with "1"
                    line = "1"
            new_lines.append(line)

    # Write updated lines back
    with open(filename, "w") as f:
        f.write("\n".join(new_lines) + "\n")

print("Updated all lines in all files.")



