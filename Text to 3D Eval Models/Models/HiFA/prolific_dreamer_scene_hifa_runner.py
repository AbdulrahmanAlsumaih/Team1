import subprocess
import time

# command for ProlificDreamer-scene-HiFA
commands = [
    [
        "python", "launch.py",
        "--config", "configs/prolificdreamer-scene-hifa.yaml",
        "--train",
        "--gpu", "0",
        "system.prompt_processor.prompt=\"A DSLR photo of a hamburger inside a restaurant\""
    ]
]

# Check for multiple commands
multiple_commands = len(commands) > 1

# Sequential Execution 
for i, command in enumerate(commands, 1):
    try:
        print(f"Running process {i}...")
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"Process {i} Output:\n", result.stdout)
        
        # Add a buffer of 3 seconds only; multiple commands
        if multiple_commands and i < len(commands):
            time.sleep(3)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred in process {i}:\n", e.stderr)
        break
