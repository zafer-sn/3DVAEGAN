import torch
from torch import optim
from torch import nn
from collections import OrderedDict
from utils import make_hyparam_string, save_new_pickle, read_pickle, SavePloat_Voxels, generateZ
from utils import calculate_iou, calculate_metrics
import os
import time
import datetime


from utils import ShapeNetPlusImageDataset, var_or_cuda
from model import _G, _D, _E
from lr_sh import MultiStepLR

import numpy as np
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def test_3DVAEGAN(args):
    hyparam_list = [("model", args.model_name),
                    ("cube", args.cube_len),
                    ("bs", args.batch_size),
                    ("g_lr", args.g_lr),
                    ("d_lr", args.d_lr),
                    ("z", args.z_dis),
                    ("bias", args.bias),
                    ("sl", args.soft_label),]

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)
    # datset define
    dsets_path = args.input_dir + args.data_dir + "test/"
    print(dsets_path)
    dsets = ShapeNetPlusImageDataset(dsets_path, args)
    dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # model define
    E = _E(args)
    G = _G(args)
    D =_D(args)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)
    E_solver = optim.Adam(E.parameters(), lr=args.g_lr, betas=args.beta)
    D_solver= optim.Adam(D.parameters(), lr=args.g_lr, betas=args.beta)
    
    criterion = nn.BCELoss()
    
    if torch.cuda.is_available():
        print("using cuda")
        G.cuda()
        E.cuda()
        D.cuda()

    pickle_path = args.output_dir + args.pickle_dir + log_param
    read_pickle(pickle_path, G, G_solver, D, D_solver, E, E_solver)
    
    # ÖNEMLİ: Modelleri eval() moduna al
    G.eval()
    E.eval()
    D.eval()
    
    # Metrik hesaplama için değişkenler
    total_recon_loss = 0
    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    batch_count = 0
    
    # Test zamanını al
    test_start_time = time.time()
    test_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Sonuçlar için .txt dosyası oluştur
    results_dir = f"{args.output_dir}/test_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_file = f"{results_dir}/test_results_{test_date}.txt"
    
    with open(results_file, 'w') as f:
        f.write(f"3D-VAE-GAN Test Sonuçları - {test_date}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Batch No | Recon Loss | IoU | Precision | Recall | F1 Score\n")
        f.write("-" * 70 + "\n")
    
    with torch.no_grad():  # Torch no_grad ile gradyanların hesaplanmasını önlüyoruz
        for i, (image, model_3d) in enumerate(dset_loaders):
            batch_start_time = time.time()
            
            X = var_or_cuda(model_3d)
            image = var_or_cuda(image)
            
            # X'i doğru şekilde yeniden şekillendir
            X_reshaped = X.view(-1, 1, args.cube_len, args.cube_len, args.cube_len)

            # Gerçek model için discriminator sonuçları
            d_real = D(X).squeeze()
            real_labels = var_or_cuda(torch.ones(X.size(0)))
            
            # Encoder ve Generator ile yeniden oluşturma
            z_mu, z_var = E(image)
            Z_vae = E.reparameterize(z_mu, z_var)
            G_vae = G(Z_vae)
            
            # Yeniden oluşturulan model için discriminator sonuçları
            d_fake = D(G_vae).squeeze()
            fake_labels = var_or_cuda(torch.zeros(X.size(0)))
            
            # Recon loss - MSE ile hesapla
            recon_loss = torch.mean(torch.sum(torch.pow((G_vae - X_reshaped), 2), dim=(1, 2, 3, 4)))
            
            # IoU hesapla - threshold değerini 0.5 olarak kullan (eğitimle aynı olmalı)
            batch_iou = calculate_iou(G_vae, X_reshaped, threshold=0.5)
            
            # Precision, Recall, F1 Score
            all_preds = torch.cat([d_real, d_fake])
            all_labels = torch.cat([real_labels, fake_labels])
            precision, recall, f1 = calculate_metrics(all_preds, all_labels)
            
            # Toplam değerlere ekle
            total_recon_loss += recon_loss.item()
            total_iou += batch_iou
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            batch_count += 1
            
            batch_time = time.time() - batch_start_time
            
            # Batch sonuçlarını yazdır
            print(f"Batch {i+1} - Recon Loss: {recon_loss.item():.4f}, IoU: {batch_iou:.4f}, "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Time: {batch_time:.2f}s")
            
            # Batch sonuçlarını dosyaya yaz
            with open(results_file, 'a') as f:
                f.write(f"{i+1} | {recon_loss.item():.4f} | {batch_iou:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f}\n")
            
            # Görselleştirmek için ilk 8 örneği kaydet
            if i < 5:  # İlk 5 batch'in görüntülerini kaydet
                samples = G_vae.cpu().data[:8].squeeze().numpy()
                image_path = args.output_dir + args.image_dir + '3DVAEGAN_test'
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                SavePloat_Voxels(samples, image_path, i)
    
    # Ortalama metrikleri hesapla
    if batch_count > 0:
        avg_recon_loss = total_recon_loss / batch_count
        avg_iou = total_iou / batch_count
        avg_precision = total_precision / batch_count
        avg_recall = total_recall / batch_count
        avg_f1 = total_f1 / batch_count
    else:
        avg_recon_loss = 0
        avg_iou = 0
        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
    
    # Toplam test süresini hesapla
    total_test_time = time.time() - test_start_time
    
    # Genel sonuçları yazdır
    print("\n" + "=" * 50)
    print("Test Sonuçları")
    print("=" * 50)
    print(f"Ortalama Recon Loss: {avg_recon_loss:.4f}")
    print(f"Ortalama IoU: {avg_iou:.4f}")
    print(f"Ortalama Precision: {avg_precision:.4f}")
    print(f"Ortalama Recall: {avg_recall:.4f}")
    print(f"Ortalama F1 Score: {avg_f1:.4f}")
    print(f"Toplam Test Süresi: {total_test_time:.2f} saniye")
    print("=" * 50)
    
    # Genel sonuçları dosyaya yaz
    with open(results_file, 'a') as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write("Test Sonuçları\n")
        f.write("=" * 50 + "\n")
        f.write(f"Ortalama Recon Loss: {avg_recon_loss:.4f}\n")
        f.write(f"Ortalama IoU: {avg_iou:.4f}\n")
        f.write(f"Ortalama Precision: {avg_precision:.4f}\n")
        f.write(f"Ortalama Recall: {avg_recall:.4f}\n")
        f.write(f"Ortalama F1 Score: {avg_f1:.4f}\n")
        f.write(f"Test Edilen Batch Sayısı: {batch_count}\n")
        f.write(f"Toplam Test Süresi: {total_test_time:.2f} saniye\n")
        f.write(f"Test Tarihi: {test_date}\n")
        f.write("=" * 50 + "\n")
    
    print(f"\nSonuçlar {results_file} dosyasına kaydedildi.")
