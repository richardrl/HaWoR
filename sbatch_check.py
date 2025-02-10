import os
import json

# root_dir = '/data/pulkitag/hamer_diffusion_policy_project/labels/02082025_egoexo_train_hands23'
root_dir = '/data/pulkitag/hamer_diffusion_policy_project/labels/02082025_egoexo_val_hands23'

missing = []
found = 0

def get_subdirs(root_path):
    # Get immediate subdirectories using list comprehension
    subdirs = [os.path.join(root_path, d) for d in os.listdir(root_path)
               if os.path.isdir(os.path.join(root_path, d))]
    return subdirs

takes_json_lst = json.load(open(os.path.join("/data/pulkitag/models/rli14/data/egoexo", "takes.json")))
valid_take_names = [take['take_name'] for take in takes_json_lst if
                    take['parent_task_name'] in ["Cooking", "Bike Repair"]]

seq_folder_paths = get_subdirs(root_dir)

seq_foldernames = [os.path.basename(dir_) for dir_ in seq_folder_paths if os.path.basename(dir_) in valid_take_names]

for subdir in os.listdir(root_dir):
    if not os.path.basename(subdir) in seq_foldernames:
        continue
    full_path = os.path.join(root_dir, subdir)
    if os.path.isdir(full_path):
       file_path = os.path.join(full_path, 'world_space_res.pth')
       exists = os.path.exists(file_path)
       print(f"{subdir}: {'✓' if exists else '✗'}")
       if exists:
           found += 1
       else:
           missing.append(subdir)

print("\nMissing world_space_res.pth in:")
for m in missing:
   print(m)
print(f"\nTotal found: {found}")
print(f"Total missing: {len(missing)}")