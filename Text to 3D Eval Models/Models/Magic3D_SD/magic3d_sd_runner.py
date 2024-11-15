import subprocess
import time


commands = [
    # Magic3D Coarse Stage with Stable Diffusion
    [
        "python", "launch.py",
        "--config", "configs/magic3d-coarse-sd.yaml",
        "--train",
        "--gpu", "0",
        "system.prompt_processor.prompt=\"a delicious hamburger\""
    ],
    # Magic3D Refine Stage with Stable Diffusion
    [
        "python", "launch.py",
        "--config", "configs/magic3d-refine-sd.yaml",
        "--train",
        "--gpu", "0",
        "system.prompt_processor.prompt=\"a delicious hamburger\"",
        "system.geometry_convert_from=path/to/coarse/stage/trial/dir/ckpts/last.ckpt"
    ]
]

# Execute each command sequentially
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
