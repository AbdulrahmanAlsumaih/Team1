import os
import subprocess
from huggingface_hub import login
from datetime import datetime

token = "***********"

def huggingface_login():
    login(token=token)

def run_dreamfusion(prompt, index):
    start_time = datetime.now()
    command = (
        f"python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 "
        f"system.prompt_processor.prompt='{prompt}' trainer.max_steps=10000 "
        f"system.prompt_processor.spawn=false"
    )
    subprocess.run(command, shell=True, check=True)
    end_time = datetime.now()
    duration = end_time - start_time
    log_run(index, prompt, duration)

def log_run(index, prompt, duration):
    log_path = os.path.join(os.getcwd(), "run_log.txt")
    formatted_duration = str(duration).split(".")[0]
    with open(log_path, "a") as log_file:
        log_file.write(f"prompt#{index} \"{prompt}\" - {formatted_duration}\n")

if __name__ == "__main__":
    huggingface_login()

    prompts = [
        "a highland cow",
        "a bulldozer made out of toy bricks",
        "an English castle, aerial view",
        "a beautifully carved wooden knight chess piece",
        "a pug made out of metal",
        "an amigurumi bulldozer",
        "a corgi puppy",
        "a crab, low poly",
        "a bald eagle carved out of wood",
        "delicious hamburger",
        "a teal moped",
        "a fresh cinnamon roll covered in glaze",
        "a red convertible car with the top down",
        "a majestic sailboat",
        "a wedge of cheese on a silver platter",
        "a silver platter piled high with fruits",
        "a delicious croissant",
        "a palm tree, low poly",
        "a pair of headphones sitting on a desk",
        "a yellow schoolbus",
        "an origami motorcycle",
        "a wooden desk and chair from an elementary school",
        "an ice cream sundae",
        "a ladybug",
        "A DMC Delorean car",
        "a roast turkey on a platter",
        "a ripe strawberry",
        "a blue tulip",
        "a shiny red stand mixer",
        "a snail on a leaf"
    ]

    for i, prompt in enumerate(prompts, start=1):
        run_dreamfusion(prompt, i)