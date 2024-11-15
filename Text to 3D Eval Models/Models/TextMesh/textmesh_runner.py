import subprocess

# Defining the command for DeepFloyd IF with TextMesh configuration
command = [
    "python", "launch.py",
    "--config", "configs/textmesh-if.yaml",
    "--train",
    "--gpu", "0",
    "system.prompt_processor.prompt=\"lib:cowboy_boots\""
]

# Execute the command
try:
    print("Running DeepFloyd IF with TextMesh configuration...")
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print("Output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("An error occurred:\n", e.stderr)
