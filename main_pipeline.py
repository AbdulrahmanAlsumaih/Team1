from SDXL.sdxl_bg_remove import sdxl_bg_remove
from zero123.myinference import get_front_and_back 
from imageTo3D.imagesTo3D import preprocess, reconstruct_and_export
import os
import argparse

def main(args):

    prompts = ["a crab, low poly",
               "a bald eagle carved out of wood",
               "a delicious hamburger"]
    
    # generate images from the prompts by SDXL and remove background
    images = sdxl_bg_remove(prompts)
    
    # get front and back image from zero123
    image_pairs = get_front_and_back(cond_images=images)
    for i in range(len(image_pairs)):
        image_pairs[i][0].save(f"{args.output_path}/{i}_front.png")
        image_pairs[i][1].save(f"{args.output_path}/{i}_back.png")
    # forward the front and back image to gassian splatting

    for i in range(len(image_pairs)):
        a = preprocess(image_pairs[i][0], preprocess_background=True, foreground_ratio=0.65)
        ply_out_path, loop_out_path = reconstruct_and_export(np.array(a), f"{args.output_path}/{i}_")

        print(f"3D model saved to {ply_out_path}")
        print(f"Video render saved to {loop_out_path}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Run text to image')

    parser.add_argument('--output_path',
                        type=str,
                        default="./myoutput",
                        help='Output path')
    args = None
    args, unparsed = parser.parse_known_args()

    isExist = os.path.exists(args.output_path)
    if isExist == False:
        os.makedirs(args.output_path)

    main(args)
