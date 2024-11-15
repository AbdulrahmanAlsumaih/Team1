import subprocess
import time

# Define each stage command as a list
commands = [
    # Zero123 + Stable Diffusion (Coarse Stage)
    [
        "python", "launch.py",
        "--config", "configs/magic123-coarse-sd.yaml",
        "--train",
        "--gpu", "0",
        "data.image_path=load/images/hamburger_rgba.png",
        "system.prompt_processor.prompt=\"a delicious hamburger\""
    ],
    # Zero123 + Stable Diffusion (Refine Stage)
    [
        "python", "launch.py",
        "--config", "configs/magic123-refine-sd.yaml",
        "--train",
        "--gpu", "0",
        "data.image_path=load/images/hamburger_rgba.png",
        "system.prompt_processor.prompt=\"a delicious hamburger\"",
        "system.geometry_convert_from=/home/user/projects/bitsandbytes/threestudio/outputs/magic123-refine-sd/hamburger_rgba.png-a_delicious_hamburger@20241114-010609/ckpts/last.ckpt"
    ]
]

# 3-second buffer
for i, command in enumerate(commands, 1):
    try:
        print(f"Running process {i}...")
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"Process {i} Output:\n", result.stdout)
        
        # Wait for 3 seconds
        if i < len(commands):
            time.sleep(3)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred in process {i}:\n", e.stderr)
        break 
