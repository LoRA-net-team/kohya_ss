import torch
from torchvision import transforms as pth_transforms
import numpy as np
import argparse, os, shutil
import csv
import torch
import clip
from PIL import Image, ImageFile
from . import VITs16, get_dino_dim
from aesthetic_model import Aesthetic_MLP, normalized
import time

def compute_cosine_distance(image_features, image_features2):
    # normalized features
    image_features = image_features / np.linalg.norm(image_features, ord=2)
    image_features2 = image_features2 / np.linalg.norm(image_features2, ord=2)
    return np.dot(image_features, image_features2)


def main(args):
    print(f'\n step 1. model')

    print(f' (1.1) DINO model')
    dino_model = VITs16(dino_model=args.dino_model, device=args.device)
    dino_transform = pth_transforms.Compose([pth_transforms.Resize(224, interpolation=3),
                                             pth_transforms.CenterCrop(224),
                                             pth_transforms.ToTensor(),
                                             pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])

    print(f' (1.2) Aesthetic model')
    mlp_model = Aesthetic_MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    mlp_model.load_state_dict(
        torch.load("../../pretrained/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth"))
    mlp_model.to(args.device)
    mlp_model.eval()

    print(f' (1.3) clip model')
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=args.device)

    print(f'\n step 2. reference image')
    ref_img_folder = args.ref_img_folder
    files = os.listdir(ref_img_folder)
    for file in files:
        name, ext = os.path.splitext(file)
        if ext != '.txt':
            img_dir = os.path.join(ref_img_folder, f'{name}.jpg')
            pil_img = Image.open(img_dir)
            ref_emb = clip_model.encode_image(clip_preprocess(pil_img).unsqueeze(0).to('cuda'))


    print(f'\n step 3. generated image')
    base_img_folder = args.base_img_folder
    conditions = os.listdir(base_img_folder)

    for condition in conditions:

        elems = []
        elems.append(['condition', 'epoch', 'average_dino_sim', 'average_ccip_diff', 'average_aes', 'avg_t2i_sim'])
        condition_dir = os.path.join(base_img_folder, condition, 'sample')
        """
        asethetic_csv_dir = os.path.join(condition_dir, 'metric')
        os.makedirs(asethetic_csv_dir, exist_ok=True)
        asethetic_csv = os.path.join(asethetic_csv_dir, 'average_sim_aesthetic.csv')
        print(f'save on {asethetic_csv}')
        """
        sample_dir = os.path.join(condition_dir, 'sample')
        epochs = os.listdir(sample_dir)
        for epoch in epochs:
            epoch_dir = os.path.join(condition_dir, epoch)
            files = os.listdir(epoch_dir)
            img_num = 0
            for image in files:
                name, ext = os.path.splitext(image)
                if ext != '.txt':
                    img_num += 1
                    # -----------------------------------------------------------------------------------------------------
                    image_dir = os.path.join(epoch_dir, image)
                    pil_img = Image.open(image_dir)  # RGB
                    # -----------------------------------------------------------------------------------------------------
                    txt_dir = os.path.join(epoch_dir, f'{name}.txt')
                    with open(txt_dir, 'r') as f:
                        prompt = f.readlines()[0]
                    prompt = prompt.replace(args.concept_token,args.class_token)
                    text = clip.tokenize([prompt]).to(args.device)

                    # -----------------------------------------------------------------------------------------------------
                    with torch.no_grad():
                        image_features = clip_model.encode_image(clip_preprocess(pil_img).unsqueeze(0).to('cuda'))
                        text_features = clip_model.encode_text(text)
                    # -----------------------------------------------------------------------------------------------------
                    # 1) I2I similarity
                    i2i_sim = torch.nn.functional.cosine_similarity(image_features,ref_emb, dim=1, eps=1e-8)
                    # 2) T2I similarity
                    t2i_sim = torch.nn.functional.cosine_similarity(image_features,text_features, dim=1, eps=1e-8)
                    print(f'epoch {epoch} image {img_num} : {i2i_sim.item()}')
                    time.sleep(10)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dino_model', type=str, default='dino_vits16')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ref_img_folder',
                        type=str,
                        default=r'/data7/sooyeon/MyData/perfusion_dataset/iom/200_iom')
    parser.add_argument('--base_img_folder',
                        type=str,
                        default=r'/data7/sooyeon/PersonalizeOverfitting/kohya_ss/result/iom_experiment/one_image')
    parser.add_argument('--concept_token', type=str, default='iom')
    parser.add_argument('--class_token', type=str, default='kitten')
    args = parser.parse_args()
    main(args)