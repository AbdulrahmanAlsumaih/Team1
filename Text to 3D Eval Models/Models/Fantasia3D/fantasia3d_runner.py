import subprocess
import time

# Defining each stage command as a list
commands = [
    # Geometry Stage
    [
        "python", "launch.py",
        "--config", "configs/fantasia3d.yaml",
        "--train",
        "--gpu", "0",
        "system.prompt_processor.prompt=\"a DSLR photo of an ice cream sundae\""
    ],
    # Texture Stage
    [
        "python", "launch.py",
        "--config", "configs/fantasia3d-texture.yaml",
        "--train",
        "--gpu", "0",
        "system.prompt_processor.prompt=\"a DSLR photo of an ice cream sundae\"",
        "system.geometry_convert_from=path/to/geometry/stage/trial/dir/ckpts/last.ckpt"
    ]
]

# Execution with a 3-second buffer
for i, command in enumerate(commands, 1):
    try:
        print(f"Running process {i}...")
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"Process {i} Output:\n", result.stdout)
        
        # Wait for 3 seconds before the next stage
        if i < len(commands):
            time.sleep(3)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred in process {i}:\n", e.stderr)
        break 
