import os
import argparse
def extract_pure_name(img_dir) :
    parent, img_name = os.path.split(img_dir)
    name, ext = os.path.splitext(img_name)
    return name, ext

def main(args) :

    print(f' step 1. call generated image folder')
    generated_image_folders = os.listdir(args.generated_image_folder)
    for generated_image_folder in generated_image_folders :
        generated_image_folder_dir = os.path.join(args.generated_image_folder, generated_image_folder)
        sample_dir = os.path.join(generated_image_folder_dir, 'sample')
        images = os.listdir(sample_dir)
        for image in images :
            image_dir = os.path.join(sample_dir, image)
            name, ext = os.path.splitext(image_dir)
            epoch = name.split('_')[-2]
            epoch = int(epoch.split('e')[-1])

            epoch_dir = os.path.join(sample_dir, str(epoch))
            os.makedirs(epoch_dir, exist_ok=True)
            new_dir = os.path.join(epoch_dir, image)
            os.rename(image_dir, new_dir)




if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_image_folder',
                        default='/data7/sooyeon/LyCORIS/lyco_v2/kohya_ss/result/perfusion_data_experience/one_image/mcat_without_caption')
    args = parser.parse_args()
    main(args)
