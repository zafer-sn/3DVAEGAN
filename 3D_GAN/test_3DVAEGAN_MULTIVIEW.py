import torch
from torch import optim
from torch import  nn
from collections import OrderedDict
from utils import make_hyparam_string, save_new_pickle, read_pickle, SavePloat_Voxels, generateZ
import os


from utils import ShapeNetMultiviewDataset, var_or_cuda
from model import _G, _D, _E_MultiView
from lr_sh import  MultiStepLR
from train_multiview import calculate_iou  # IoU fonksiyonunu import et
from datetime import datetime

import numpy as np
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def test_3DVAEGAN_MULTIVIEW(args):
    hyparam_list = [("model", args.model_name),
                    ("cube", args.cube_len),
                    ("bs", args.batch_size),
                    ("g_lr", args.g_lr),
                    ("d_lr", args.d_lr),
                    ("z", args.z_dis),
                    ("bias", args.bias),
                    ("sl", args.soft_label), ]

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)
    # datset define
    dsets_path = args.input_dir + args.data_dir + "test/"
    print(dsets_path)
    dsets = ShapeNetMultiviewDataset(dsets_path, args)
    dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # model define
    E = _E_MultiView(args)
    G = _G(args)
    D = _D(args)

    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)
    E_solver = optim.Adam(E.parameters(), lr=args.g_lr, betas=args.beta)
    D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)

    if torch.cuda.is_available():
        print("using cuda")
        G.cuda()
        E.cuda()

    pickle_path = args.output_dir + args.pickle_dir + log_param
    read_pickle(pickle_path, G, G_solver, D, D_solver,E,E_solver)

    # Test sonuçları için değişkenler
    recon_loss_total = 0
    iou_total = 0
    sample_count = 0
    test_results = []

    # Test sonuçları için dosya hazırla
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, 'test_results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    results_file = os.path.join(results_path, f'test_results_{timestamp}.txt')

    for i, (images, model_3d) in enumerate(dset_loaders):

        X = var_or_cuda(model_3d)

        Z_vae, z_mus, z_vars = E(images)
        G_vae = G(Z_vae)

        # Recon loss hesapla
        recon_loss = torch.sum(torch.pow((G_vae - X), 2), dim=(1,2,3))
        avg_recon_loss = torch.mean(recon_loss).item()
        
        # IoU hesapla
        iou = calculate_iou(G_vae, X.view(-1, 1, args.cube_len, args.cube_len, args.cube_len)).item()

        # Toplam değerleri güncelle
        recon_loss_total += avg_recon_loss
        iou_total += iou
        sample_count += 1

        # Her batch için sonuçları kaydet
        test_results.append(f"Sample {i}: Recon Loss = {avg_recon_loss:.4f}, IoU = {iou:.4f}")
        print(f"Sample {i}: Recon Loss = {avg_recon_loss:.4f}, IoU = {iou:.4f}")

        # Görselleştirme
        samples = G_vae.cpu().data[:8].squeeze().numpy()
        image_path = args.output_dir + args.image_dir + '3DVAEGAN_MULTIVIEW_MAX_test'
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        SavePloat_Voxels(samples, image_path, i)

    # Ortalama değerleri hesapla
    avg_recon_loss = recon_loss_total / sample_count
    avg_iou = iou_total / sample_count

    # Sonuçları dosyaya yaz
    with open(results_file, 'w') as f:
        f.write(f"Test Results - {timestamp}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Samples: {sample_count}\n")
        f.write(f"Average Reconstruction Loss: {avg_recon_loss:.4f}\n")
        f.write(f"Average IoU: {avg_iou:.4f}\n")
        f.write("\nDetailed Results:\n")
        f.write("-"*50 + "\n")
        for result in test_results:
            f.write(result + "\n")

    # Özet sonuçları ekrana yazdır
    print("\n" + "="*50)
    print("Test Results Summary:")
    print(f"Results saved to: {results_file}")
    print(f"Total samples: {sample_count}")
    print(f"Average Reconstruction Loss: {avg_recon_loss:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print("="*50)