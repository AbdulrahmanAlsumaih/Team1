import subprocess

# Defining the command for Control4D static editing
command = [
    "python", "launch.py",
    "--config", "configs/control4d-static.yaml",
    "--train",
    "--gpu", "0",
    "data.dataroot=\"twindom\"",
    "system.prompt_processor.prompt=\"Elon Musk wearing red shirt, RAW photo, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3\""
]

# Executing the command
try:
    print("Running Control4D static editing...")
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print("Output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("An error occurred:\n", e.stderr)
