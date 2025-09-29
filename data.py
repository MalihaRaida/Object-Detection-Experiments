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


# def copy_files(file_list, source_dir, destination_dir):
#     # Create destination folder if it doesn't exist
#     os.makedirs(destination_dir, exist_ok=True)

#     all_files = os.listdir(source_dir)

#     for name in file_list:
#         # Find all files that match "name" without extension
#         matching_files = [f for f in all_files if os.path.splitext(f)[0] == name]

#         if not matching_files:
#             print(f"No match found for: {name}")
#             continue

#         for f in matching_files:
#             src = os.path.join(source_dir, f)
#             dst = os.path.join(destination_dir, f)
#             if os.path.isfile(src):
#                 shutil.copy2(src, dst)
#                 # print(f"Copied: {src} → {dst}")

directory_path = "C:\\Users\\USER\\Downloads\\archive\\Stanford_Dogs_Yolov8\\Annotations\\val"
file_names = list_files_without_extension(directory_path)

for filename in file_names:
    with open(filename, "r") as f:
        content = f.read()

    if content:
        # Find first space
        space_index = content.find(" ")
        if space_index != -1:
            # Replace everything before first space with "1"
            content = "1" + content[space_index:]
        else:
            # If no space, replace whole content with "1"
            content = "1"

    with open(filename, "w") as f:
        f.write(content)

print("Updated all files: replaced text up to first space with 1.")


# Example usage

# copy_files(file_names,"C:\\Users\\USER\\Downloads\\archive\\Stanford_Dogs_Yolov8\\Annotations\\merge_folder",)
