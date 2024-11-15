import subprocess
import time

# Defining each command as a list
commands = [
    # Train Latent-NeRF in Stable Diffusion latent space
    [
        "python", "launch.py",
        "--config", "configs/latentnerf.yaml",
        "--train",
        "--gpu", "0",
        "system.prompt_processor.prompt=\"a delicious hamburger\""
    ],
    # Refine Latent-NeRF in RGB space
    [
        "python", "launch.py",
        "--config", "configs/latentnerf-refine.yaml",
        "--train",
        "--gpu", "0",
        "system.prompt_processor.prompt=\"a delicious hamburger\"",
        "system.weights=/home/user/projects/bitsandbytes/threestudio/outputs/latentnerf/a_delicious_hamburger@20241113-175800/ckpts/last.ckpt"
    ]
]

# Execute each command sequentially with a 3-second buffer
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
