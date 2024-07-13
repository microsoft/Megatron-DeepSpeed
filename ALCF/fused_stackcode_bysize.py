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
    print("done ?")

directory_path = "./fused_stack/"
out_path = "./fused_by_size"
folder_count = 0
file_list = list_json_gz_files(directory_path)
size_dict = {}
for efile in file_list:
    size_of_files = os.stat(efile)
    size_dict[efile] = size_of_files.st_size / (1024*1024) # in MBs

sorted_size_dict = dict(sorted(size_dict.items(), key=lambda item: item[1]))
vol = 0
sublist = []
super_list = {}
i=1
for key, val in sorted_size_dict.items():
    if vol + val > 4608:
        # add this item to list and reset vol, sublist
        vol = 0
        sublist.append(key)
        #print(sublist)
        print("************")
        super_list[i] = sublist
        output_file = out_path + "/fused_stack_" + str(i) + ".json.gz"
        print(output_file)
        combine_json_gz_files(sublist, output_file)
        sublist = []
        i=i+1
    else:
        vol = vol + val
        sublist.append(key)
#print(t)
#for folder in os.listdir(directory_path):
#    print(f"working for folder {folder} {os.path.join(directory_path, folder)}")
#    folder_count = folder_count + 1
#    json_gz_files = list_json_gz_files(os.path.join(directory_path, folder))
#    out_path = os.path.join("./fused_stack", folder)
#    os.makedirs(out_path, exist_ok=True)
#    output_file = os.path.join(out_path, 'fused.json.gz')
#    combine_json_gz_files(json_gz_files, output_file)
