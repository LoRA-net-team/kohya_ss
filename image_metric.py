import torch
from torchvision import transforms as pth_transforms
from PIL import Image, ImageFile
import numpy as np
import argparse, os, shutil
import csv
import torch
import clip
from PIL import Image, ImageFile
from Dino_Scoring import VITs16, get_dino_dim
from Dino_Scoring.aesthetic_model import Aesthetic_MLP, normalized


def compute_cosine_distance(image_features, image_features2):
    # normalized features
    image_features = image_features / np.linalg.norm(image_features, ord=2)
    image_features2 = image_features2 / np.linalg.norm(image_features2, ord=2)
    return np.dot(image_features, image_features2)


def main(args) :

    print(f'step 1. model')
    print(f' (1.1) DINO model')
    dino_model = VITs16(dino_model=args.dino_model, device=args.device)
    dino_transform = pth_transforms.Compose([pth_transforms.Resize(224, interpolation=3),
                                             pth_transforms.CenterCrop(224),
                                             pth_transforms.ToTensor(),
                                             pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])

    print(f' (1.2) Aesthetic model')
    mlp_model = Aesthetic_MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    mlp_model.load_state_dict(torch.load("../../../pretrained/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth"))
    mlp_model.to(args.device)
    mlp_model.eval()

    print(f' (1.3) clip model')
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=args.device)

    print(f'step 2. ref image')
    ref_img_dict = {}
    ref_img_folder = args.ref_img_folder
    files = os.listdir(ref_img_folder)
    for file in files :
        name, ext = os.path.splitext(file)
        if ext != '.txt' :
            txt_dir = os.path.join(ref_img_folder, f'{name}.txt')
            with open(txt_dir, 'r') as f :
                caption = f.readlines()[0]
            caption = caption.strip()
            try :
                img_dir = os.path.join(ref_img_folder, f'{name}.png')
                img_emb = get_dino_dim(img_dir, dino_model, dino_transform, args)
            except :
                img_dir = os.path.join(ref_img_folder, f'{name}.jpg')
                img_emb = get_dino_dim(img_dir, dino_model, dino_transform, args)
            print(f'caption : {caption}')
            ref_img_dict[caption] = img_emb

    print(f'step 3. image')
    base_img_folder = args.base_img_folder
    conditions = os.listdir(base_img_folder)
    elems = []
    best_sim = []
    best_sim.append(['Condition', 'Best_epoch', 'Avg_I2I_sim', 'Aes_score', 'Avg_T2I_sim'])
    for condition in conditions:
        #trg_seed = condition.split('_')[-1]
        condition_dir = os.path.join(base_img_folder, condition, 'sample')
        epochs = os.listdir(condition_dir)
        best_epoch_dict = {}
        for epoch in epochs:
            if 'record' not in epoch and 'sample' not in epoch and 'safetensors' not in epoch :
                epoch_dir = os.path.join(condition_dir, epoch)
                images = os.listdir(epoch_dir)
                similarities = []
                aesthetic_predictions = []
                img_num = 0
                for image in images:
                    name, ext = os.path.splitext(image)
                    if ext != '.txt':
                        img_num += 1
                        image_dir = os.path.join(epoch_dir, image)
                        txt_dir = os.path.join(epoch_dir, f'{name}.txt')
                        pil_img = Image.open(image_dir)  # RGB
                        # (1) Aesthetic
                        with torch.no_grad():
                            image_features = clip_model.encode_image(clip_preprocess(pil_img).unsqueeze(0).to('cuda'))
                        im_emb_arr = normalized(image_features.cpu().detach().numpy())
                        aesthetic_prediction = mlp_model(torch.from_numpy(im_emb_arr).to('cuda').type(torch.cuda.FloatTensor))
                        aesthetic_predictions.append(aesthetic_prediction.item())
                        # (2) Dino
                        image_embedding = get_dino_dim(image_dir,dino_model,dino_transform, args)
                        try :
                            with open(txt_dir, 'r') as f:
                                caption = f.readlines()[0]
                        except :
                            print(txt_dir)
                        if caption in ref_img_dict.keys() :
                            ref_emb = ref_img_dict[caption]
                            sim = torch.nn.functional.cosine_similarity(image_embedding,
                                                                        ref_emb,
                                                                        dim=1, eps=1e-8)
                            similarities.append(sim.item())
                        #else :
                        #    print(f'{caption} is not in ref_img_dict.keys()')
                #print(F'img_num : {img_num}')
                average_aes = sum(aesthetic_predictions) / len(aesthetic_predictions)
                average_sim = sum(similarities) / len(similarities)
                best_epoch_dict[epoch] = [average_sim, average_aes]
                elem = [condition, epoch, average_sim, average_aes]
                elems.append(elem)
        best_epoch_dict = sorted(best_epoch_dict.items(), key=lambda x: x[1][0], reverse=True)
        print(f'{condition} best_epoch_dict : {best_epoch_dict}')
        best_epoch, score = best_epoch_dict[0]
        avg_sim, aes_score = score

        # -----------------------------------------------------------------------------------------
        # best epoch dit
        best_epoch_dir = os.path.join(condition_dir, best_epoch)
        files = os.listdir(best_epoch_dir)
        t2i_sim_list = []
        for file in files :
            name, ext = os.path.splitext(file)
            if ext != '.txt' :
                img_dir = os.path.join(best_epoch_dir, f'{name}.png')
                txt_dir = os.path.join(best_epoch_dir, f'{name}.txt')
                with open(txt_dir, 'r') as f :
                    txt = f.readlines()[0]
                gen_image = clip_preprocess(Image.open(img_dir)).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    clip_img_emb = clip_model.encode_image(gen_image)  # [1,512]
                    target_text = txt.split(',')
                    text = clip.tokenize(target_text).to(args.device)
                    clip_txt_emb = clip_model.encode_text(text)
                    t2i_sim = torch.nn.functional.cosine_similarity(clip_img_emb, clip_txt_emb, dim=1, eps=1e-8).mean().item()
                    t2i_sim_list.append(t2i_sim)
        avg_t2i_sim = sum(t2i_sim_list) / len(t2i_sim_list)
        # -----------------------------------------------------------------------------------------
        best_sim.append([condition, best_epoch, avg_sim, aes_score,avg_t2i_sim])

    with open('generated_images/temp2/average_sim_aesthetic_park_second.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(elems)

    with open('generated_images/temp2/best_sim_second_1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(best_sim)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dino_model', type=str, default='dino_vits16')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ref_img_folder', type=str, default=r'/data7/sooyeon/MyData/haibara_dataset/not_test/haibara_19/2_girl')
    parser.add_argument('--base_img_folder', type=str, default=r'./result/haibara_experience/one_image/name_3_without_caption')

    args = parser.parse_args()
    main(args)
    '../PersonalizeOverfitting/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth'