import os
from os import system
import glob
import json
import gzip
import pdb

def list_json_gz_files(directory):
    # Create the search pattern for JSON.gz files
    search_pattern = os.path.join(directory, "**/*.json.gz")
    
    # Use glob to find all files matching the pattern
    json_gz_files = glob.glob(search_pattern, recursive=True)
    
    return json_gz_files

def combine_json_gz_files(json_gz_files, output_file):
    in_list = ""
    for i in json_gz_files:
        in_list = in_list + " " +str(i)
    command = "cat" + in_list + " > " + output_file
    print(command)
    system(command)
    print("done")

directory_path = "./data/stack-code/"
folder_count = 0
for folder in os.listdir(directory_path):
    print(f"working for folder {folder} {os.path.join(directory_path, folder)}")
    folder_count = folder_count + 1
    json_gz_files = list_json_gz_files(os.path.join(directory_path, folder))
    out_path = os.path.join("./fused_stack", folder)
    os.makedirs(out_path, exist_ok=True)
    output_file = os.path.join(out_path, 'fused.json.gz')
    combine_json_gz_files(json_gz_files, output_file)

