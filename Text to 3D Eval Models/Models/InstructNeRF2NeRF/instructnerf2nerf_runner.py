import subprocess

# Define the command for InstructNeRF2NeRF
command = [
    "python", "launch.py",
    "--config", "configs/instructnerf2nerf.yaml",
    "--train",
    "--gpu", "0",
    "data.dataroot=\"face\"",
    "data.camera_layout=\"front\"",
    "data.camera_distance=1",
    "data.eval_interpolation=[1,3,50]",
    "system.prompt_processor.prompt=\"Turn him into Albert Einstein\""
]

# Execute the command
try:
    print("Running InstructNeRF2NeRF 3D editing...")
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print("Output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("An error occurred:\n", e.stderr)
