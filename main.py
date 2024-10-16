from SDXL.sdxl_bg_remove import sdxl_bg_remove
from zero123.myinference import get_front_and_back 


def main(args)

    prompts = ["a crab, low poly",
               "a bald eagle carved out of wood",
               "a delicious hamburger"]
    
    # generate images from the prompts by SDXL and remove background
    images = sdxl_bg_remove(prompts)

    for img in images:
        # get front and back image from zero123
        front, back = get_front_and_back(cond_image=img)
        
        # forward the front and back image to gassian splatting


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


