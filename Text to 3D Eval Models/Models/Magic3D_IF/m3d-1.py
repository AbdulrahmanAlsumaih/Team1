import os
import time
import json
import subprocess
from tqdm import tqdm
import glob

PROMPT_FILE = "prompts.json"
LOG_FILE = "logs-5.txt"
VERBOSE_LOG_FILE = "logs-v-5.txt"
START_PROMPT = "A tricycle"  # The specific prompt to start from
END_PROMPT = "A red scarf"  # The specific prompt to end at
HUGGINGFACE_TOKEN = "################"

def load_prompts(prompt_file):
    with open(prompt_file, 'r') as f:
        return json.load(f)

def find_checkpoint_path(prompt):
    # Convert the prompt to the directory format
    prompt_dir_name = prompt.replace(" ", "_")
    # Use glob to find the matching directory with timestamp pattern
    search_pattern = f"outputs/magic3d-coarse-if/{prompt_dir_name}@*/ckpts/last.ckpt"
    matching_files = glob.glob(search_pattern)
    if matching_files:
        return matching_files[0]
    else:
        print(f"No matching checkpoint found for prompt '{prompt}'.")
        return None

# Log into Hugging Face using the token
login_command = f'huggingface-cli login --token {HUGGINGFACE_TOKEN} --add-to-git-credential'
subprocess.run(login_command, shell=True, check=True)

# Load prompts
prompts = load_prompts(PROMPT_FILE)
total_prompts = len(prompts)

# Find the starting and ending indices
if START_PROMPT in prompts:
    start_index = prompts.index(START_PROMPT)
else:
    print(f"Prompt '{START_PROMPT}' not found in the list. Exiting.")
    exit(1)

if END_PROMPT in prompts:
    end_index = prompts.index(END_PROMPT) + 1  # +1 to include the END_PROMPT itself
else:
    print(f"Prompt '{END_PROMPT}' not found in the list. Exiting.")
    exit(1)

# Loop through the prompts within the specified range
for i, prompt in enumerate(tqdm(prompts[start_index:end_index], total=end_index-start_index, desc="Processing prompts", unit="prompt")):
    actual_index = start_index + i

    # Step 1: Run the initial command for coarse processing
    command_step1 = f'python launch.py --config configs/magic3d-coarse-if.yaml --train --gpu 0 system.prompt_processor.prompt="{prompt}"'
    print(f"Running Step 1 for prompt {actual_index + 1}/{total_prompts}: '{prompt}'")
    
    with open(VERBOSE_LOG_FILE, 'a') as verbose_log:
        result_step1 = subprocess.run(command_step1, shell=True, stdout=verbose_log, stderr=verbose_log)
    
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"Step 1 Prompt {actual_index + 1}/{total_prompts}: '{prompt}' - Command executed with result: {result_step1.returncode}\n")
    
    time.sleep(3)  # Wait 3 seconds before starting Step 2

    # Step 2: Locate the checkpoint file and run the refine command
    checkpoint_path = find_checkpoint_path(prompt)
    if checkpoint_path:
        command_step2 = f'python launch.py --config configs/magic3d-refine-sd.yaml --train --gpu 0 system.prompt_processor.prompt="{prompt}" system.geometry_convert_from={checkpoint_path}'
        print(f"Running Step 2 for prompt {actual_index + 1}/{total_prompts}: '{prompt}'")

        with open(VERBOSE_LOG_FILE, 'a') as verbose_log:
            result_step2 = subprocess.run(command_step2, shell=True, stdout=verbose_log, stderr=verbose_log)
        
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"Step 2 Prompt {actual_index + 1}/{total_prompts}: '{prompt}' - Command executed with result: {result_step2.returncode}\n")
    else:
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"Step 2 Prompt {actual_index + 1}/{total_prompts}: '{prompt}' - Checkpoint not found, skipping.\n")

print("Processing complete.")
