import os
import time
import json
import subprocess
from tqdm import tqdm

PROMPT_FILE = "prompts.json"
LOG_FILE = "logs.txt"
VERBOSE_LOG_FILE = "logs-v.txt"
START_PROMPT = "A sunflower"  # The specific prompt to start from

def load_prompts(prompt_file):
    with open(prompt_file, 'r') as f:
        return json.load(f)

# Load prompts
prompts = load_prompts(PROMPT_FILE)
total_prompts = len(prompts)

# Find the starting index of the specified prompt
if START_PROMPT in prompts:
    start_index = prompts.index(START_PROMPT)
else:
    print(f"Prompt '{START_PROMPT}' not found in the list. Exiting.")
    exit(1)

# Loop through the prompts starting from the found index
for i, prompt in enumerate(tqdm(prompts[start_index:], total=total_prompts-start_index, desc="Processing prompts", unit="prompt")):
    actual_index = start_index + i
    command = f'python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 system.prompt_processor.prompt="{prompt}" trainer.max_steps=10000 system.prompt_processor.spawn=false'
    print(f"Running prompt {actual_index + 1}/{total_prompts}: '{prompt}'")

    # Run the command and capture both stdout and stderr
    with open(VERBOSE_LOG_FILE, 'a') as verbose_log:
        result = subprocess.run(command, shell=True, stdout=verbose_log, stderr=verbose_log)

    # Write a summary to the standard log file after each prompt
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"Prompt {actual_index + 1}/{total_prompts}: '{prompt}' - Command executed with result: {result.returncode}\n")

    time.sleep(1)

print("Processing complete.")