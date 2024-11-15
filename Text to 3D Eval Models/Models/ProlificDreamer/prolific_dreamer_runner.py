import subprocess
import time

# Defining each stage command as a list
commands = [
    # Stage 1 (NeRF)
    [
        "python", "launch.py",
        "--config", "configs/prolificdreamer.yaml",
        "--train",
        "--gpu", "0",
        "system.prompt_processor.prompt=\"a pineapple\""
    ],
    # Stage 2 (Geometry Refinement)
    [
        "python", "launch.py",
        "--config", "configs/prolificdreamer-geometry.yaml",
        "--train",
        "--gpu", "0",
        "system.prompt_processor.prompt=\"a pineapple\"",
        "system.geometry_convert_from=/home/user/projects/bitsandbytes/threestudio/outputs/prolificdreamer/a_pineapple@20241112-145808/ckpts/last.ckpt"
    ],
    # Stage 3 (Texturing)
    [
        "python", "launch.py",
        "--config", "configs/prolificdreamer-texture.yaml",
        "--train",
        "--gpu", "0",
        "system.prompt_processor.prompt=\"a pineapple\"",
        "system.geometry_convert_from=/home/user/projects/bitsandbytes/threestudio/outputs/prolificdreamer/a_pineapple@20241112-145808/ckpts/last.ckpt"
    ]
]

# Execution
for i, command in enumerate(commands, 1):
    try:
        print(f"Running Stage {i}...")
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"Stage {i} Output:\n", result.stdout)
        # Wait for 3 seconds before the next stage
        time.sleep(3)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred in Stage {i}:\n", e.stderr)
        break  # Stop if an error occurs
