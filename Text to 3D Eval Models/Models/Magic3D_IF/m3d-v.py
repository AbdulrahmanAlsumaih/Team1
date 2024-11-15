import os
import time
import json
import subprocess
from tqdm import tqdm
import glob

PROMPT_FILE = "prompts.json"
LOG_FILE = "logs.txt"
VERBOSE_LOG_FILE = "logs-v.txt"
START_PROMPT = "A sunflower"  # The specific prompt to start from
HUGGINGFACE_TOKEN = "################"

def load_prompts(prompt_file):
    try:
        print(f"Loading prompts from {prompt_file}")
        with open(prompt_file, 'r') as f:
            prompts = json.load(f)
        print(f"Successfully loaded {len(prompts)} prompts.")
        return prompts
    except Exception as e:
        print(f"Error loading prompts: {e}")
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"Error loading prompts: {e}\n")
        exit(1)

def find_checkpoint_path(prompt):
    prompt_dir_name = prompt.replace(" ", "_")
    search_pattern = f"outputs/magic3d-coarse-if/{prompt_dir_name}@*/ckpts/last.ckpt"
    print(f"Searching for checkpoint with pattern: {search_pattern}")
    matching_files = glob.glob(search_pattern)
    if matching_files:
        print(f"Checkpoint found: {matching_files[0]}")
        return matching_files[0]
    else:
        print(f"No matching checkpoint found for prompt '{prompt}'.")
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"No matching checkpoint found for prompt '{prompt}'.\n")
        return None

# Log into Hugging Face using the token and add to git credentials
print("Starting Hugging Face login")
login_command = f'huggingface-cli login --token {HUGGINGFACE_TOKEN} --add-to-git-credential'
login_result = subprocess.run(login_command, shell=True, capture_output=True, text=True)
if login_result.returncode != 0:
    print(f"Login failed: {login_result.stderr}")
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"Login failed: {login_result.stderr}\n")
    exit(1)
else:
    print("Login successful.")
    with open(LOG_FILE, 'a') as log_file:
        log_file.write("Login successful.\n")

# Load prompts
prompts = load_prompts(PROMPT_FILE)
total_prompts = len(prompts)

# Find the starting index of the specified prompt
if START_PROMPT in prompts:
    start_index = prompts.index(START_PROMPT)
else:
    print(f"Prompt '{START_PROMPT}' not found in the list. Exiting.")
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"Prompt '{START_PROMPT}' not found in the list. Exiting.\n")
    exit(1)

# Loop through the prompts starting from the found index
for i, prompt in enumerate(tqdm(prompts[start_index:], total=total_prompts-start_index, desc="Processing prompts", unit="prompt")):
    actual_index = start_index + i
    print(f"Starting Step 1 for prompt {actual_index + 1}/{total_prompts}: '{prompt}'")

    # Step 1: Run the initial command for coarse processing
    command_step1 = f'python launch.py --config configs/magic3d-coarse-if.yaml --train --gpu 0 system.prompt_processor.prompt="{prompt}"'
    print(f"Executing command: {command_step1}")
    
    with open(VERBOSE_LOG_FILE, 'a') as verbose_log:
        result_step1 = subprocess.run(command_step1, shell=True, stdout=verbose_log, stderr=verbose_log)
    
    if result_step1.returncode != 0:
        print(f"Step 1 failed for prompt '{prompt}'. Check verbose log for details.")
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"Step 1 failed for prompt {actual_index + 1}/{total_prompts}: '{prompt}'\n")
        continue
    else:
        print(f"Step 1 completed successfully for prompt '{prompt}'")
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"Step 1 Prompt {actual_index + 1}/{total_prompts}: '{prompt}' - Command executed successfully.\n")
    
    time.sleep(3)  # Wait 3 seconds before starting Step 2

    # Step 2: Locate the checkpoint file and run the refine command
    checkpoint_path = find_checkpoint_path(prompt)
    if checkpoint_path:
        command_step2 = f'python launch.py --config configs/magic3d-refine-sd.yaml --train --gpu 0 system.prompt_processor.prompt="{prompt}" system.geometry_convert_from={checkpoint_path}'
        print(f"Executing Step 2 for prompt '{prompt}' with command: {command_step2}")

        with open(VERBOSE_LOG_FILE, 'a') as verbose_log:
            result_step2 = subprocess.run(command_step2, shell=True, stdout=verbose_log, stderr=verbose_log)
        
        if result_step2.returncode != 0:
            print(f"Step 2 failed for prompt '{prompt}'. Check verbose log for details.")
            with open(LOG_FILE, 'a') as log_file:
                log_file.write(f"Step 2 failed for prompt {actual_index + 1}/{total_prompts}: '{prompt}'\n")
        else:
            print(f"Step 2 completed successfully for prompt '{prompt}'")
            with open(LOG_FILE, 'a') as log_file:
                log_file.write(f"Step 2 Prompt {actual_index + 1}/{total_prompts}: '{prompt}' - Command executed successfully.\n")
    else:
        print(f"Skipping Step 2 for prompt '{prompt}' due to missing checkpoint.")
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"Step 2 Prompt {actual_index + 1}/{total_prompts}: '{prompt}' - Checkpoint not found, skipping.\n")

print("Processing complete.")
