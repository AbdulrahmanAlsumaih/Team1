import subprocess

# Define the command for Zero123 configuration
command = [
    "python", "launch.py",
    "--config", "configs/zero123.yaml",
    "--train",
    "--gpu", "0",
    "data.image_path=./load/images/dog1_rgba.png"
]

# Execute the command
try:
    print("Running Zero123 configuration...")
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print("Output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("An error occurred:\n", e.stderr)
