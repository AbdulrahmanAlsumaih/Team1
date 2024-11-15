import subprocess

# Defining the command for SJC configuration
command = [
    "python", "launch.py",
    "--config", "configs/sjc.yaml",
    "--train",
    "--gpu", "0",
    "system.prompt_processor.prompt=\"A high quality photo of a delicious burger\""
]

# Executing the command
try:
    print("Running SJC configuration...")
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print("Output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("An error occurred:\n", e.stderr)
