import subprocess

# command for DreamFusion with Stable Diffusion
command = [
    "python", "launch.py",
    "--config", "configs/dreamfusion-sd.yaml",
    "--train",
    "--gpu", "0",
    "system.prompt_processor.prompt=\"a delicious hamburger\""
]

# Execute the command
try:
    print("Running DreamFusion with Stable Diffusion configuration...")
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print("Output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("An error occurred:\n", e.stderr)
