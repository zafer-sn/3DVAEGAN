import scipy.ndimage as nd
import scipy.io as io
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle
from skimage.io import imread
import trimesh
from PIL import Image
from torchvision import transforms
import binvox_rw
import glob

def getVoxelFromMat(path, cube_len=64):
    """Mat 데이터로 부터 Voxel 을 가져오는 함수"""
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels

def getVolumeFromBinvox(path):
    with open(path, 'rb') as file:
        data = np.int32(binvox_rw.read_as_3d_array(file).data)
    return data

def getVolumeFromOFF(path, sideLen=32):
    mesh = trimesh.load(path)
    volume = trimesh.voxel.VoxelMesh(mesh, 0.5).matrix
    (x, y, z) = map(float, volume.shape)
    volume = nd.zoom(volume.astype(float),
                     (sideLen/x, sideLen/y, sideLen/z),
                     order=1,
                     mode='nearest')
    volume[np.nonzero(volume)] = 1.0
    return volume.astype(np.bool)


def getVFByMarchingCubes(voxels, threshold=0.5):
    """Voxel 로 부터 Vertices, faces 리턴 하는 함수"""
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def plotFromVoxels(voxels):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig('test')
    plt.show()


def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    #plt.show()
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

    with open(path + '/{}.pkl'.format(str(iteration).zfill(3)), "wb") as f:
        pickle.dump(voxels, f, protocol=pickle.HIGHEST_PROTOCOL)


def make_hyparam_string(hyparam_dict):
    str_result = ""
    for i in hyparam_dict.keys():
        str_result = str_result + str(i) + "=" + str(hyparam_dict[i]) + "_"
    return str_result[:-1]

class ShapeNetDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root, args):
        """Set the path for Data.

        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        self.root = root
        self.listdir = os.listdir(self.root)
        self.args = args

    def __getitem__(self, index):
        #with open(self.root + self.listdir[index], "rb") as f:
            #volume = np.asarray(getVoxelFromMat(f, self.args.cube_len), dtype=np.float32)
            #plotFromVoxels(volume)
        model_3d_file = [name for name in self.listdir if name.endswith('.' + "binvox")][index]
        volume = np.array(getVolumeFromBinvox(self.root + model_3d_file), dtype=np.float32)
        return torch.FloatTensor(volume)

    def __len__(self):
        return len([name for name in self.listdir if name.endswith('.' + "binvox")])

class ShapeNetPlusImageDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root, args):
        """Set the path for Data.

        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        self.root = root
        self.listdir = os.listdir(self.root)
        self.args = args
        self.img_size = args.image_size
        self.p = transforms.Compose([transforms.Resize((self.img_size, self.img_size))])

    def __getitem__(self, index):

        model_3d_file = [name for name in self.listdir if name.endswith('.' + "binvox")][index]

        model_2d_file = model_3d_file[:-7] + "_002.png"
        model_2d_file_depth = model_3d_file[:-7] + "_002" + "_depth.png"
        # Copy arrays to make them writable
        volume = np.array(getVolumeFromBinvox(self.root + model_3d_file), dtype=np.float32).copy()
        image = Image.open(self.root + model_2d_file)
        image_depth = Image.open(self.root + model_2d_file_depth)
        image = np.array(self.p(image), dtype=np.float32).copy()
        image_depth = np.array(self.p(image_depth), dtype=np.float32).copy()

        image_depth = image_depth[..., np.newaxis]
        combined_image = np.dstack((image, image_depth))        
        
        return (torch.FloatTensor(combined_image), torch.FloatTensor(volume))

    def __len__(self):
        return len( [name for name in self.listdir if name.endswith('.' + "binvox")])

class ShapeNetMultiviewDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root, args):
        """Set the path for Data.

        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        self.root = root
        self.listdir = os.listdir(self.root)
        self.args = args
        self.img_size = args.image_size
        self.p = transforms.Compose([transforms.Resize((self.img_size, self.img_size))])

    def __getitem__(self, index):

        model_3d_file = [name for name in self.listdir if name.endswith('.' + "binvox")][index]

        model_2d_files = [name for name in self.listdir if name.startswith(model_3d_file[:-7]) and name.endswith(".png")][:3]
        #with open(self.root + model_3d_file, "rb") as f:
        volume = np.array(getVolumeFromBinvox(self.root + model_3d_file), dtype=np.float32)
        #print(volume.shape)
        #plotFromVoxels(volume)
        #with open(self.root + model_2d_file, "rb") as g:
        #image = np.array(imread(self.root + model_2d_file))
        images = [torch.FloatTensor(np.array(self.p(Image.open(self.root +x )))) for x in model_2d_files]
        return (images, torch.FloatTensor(volume) )

    def __len__(self):
        return len( [name for name in self.listdir if name.endswith('.' + "binvox")])


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def generateZ(args):

    if args.z_dis == "norm":
        Z = var_or_cuda(torch.Tensor(args.batch_size, args.z_size).normal_(0, 0.33))
    elif args.z_dis == "uni":
        Z = var_or_cuda(torch.randn(args.batch_size, args.z_size))
    else:
        print("z_dist is not normal or uniform")

    return Z

########################## Pickle helper ###############################


def read_pickle(path, G, G_solver, D_, D_solver,E_=None,E_solver = None ):
    try:

        files = os.listdir(path)
        file_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
        file_list.sort()
        recent_iter = str(file_list[-1])
        print(recent_iter, path)

        with open(path + "/G_" + recent_iter + ".pkl", "rb") as f:
            G.load_state_dict(torch.load(f))
        with open(path + "/G_optim_" + recent_iter + ".pkl", "rb") as f:
            G_solver.load_state_dict(torch.load(f))
        with open(path + "/D_" + recent_iter + ".pkl", "rb") as f:
            D_.load_state_dict(torch.load(f))
        with open(path + "/D_optim_" + recent_iter + ".pkl", "rb") as f:
            D_solver.load_state_dict(torch.load(f))
        if E_ is not None:
            with open(path + "/E_" + recent_iter + ".pkl", "rb") as f:
                E_.load_state_dict(torch.load(f))
            with open(path + "/E_optim_" + recent_iter + ".pkl", "rb") as f:
                E_solver.load_state_dict(torch.load(f))


    except Exception as e:

        print("fail try read_pickle", e)



def save_new_pickle(path, iteration, G, G_solver, D_, D_solver, E_=None,E_solver = None):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/G_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G.state_dict(), f)
    with open(path + "/G_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G_solver.state_dict(), f)
    with open(path + "/D_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_.state_dict(), f)
    with open(path + "/D_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_solver.state_dict(), f)
    if E_ is not None:
        with open(path + "/E_" + str(iteration) + ".pkl", "wb") as f:
            torch.save(E_.state_dict(), f)
        with open(path + "/E_optim_" + str(iteration) + ".pkl", "wb") as f:
            torch.save(E_solver.state_dict(), f)

def calculate_iou(pred_voxel, gt_voxel, threshold=0.5):
    """
    3D Voxel tensörler için IoU (Intersection over Union) hesaplar.
    
    Args:
        pred_voxel: Tahmin edilen voxel tensor
        gt_voxel: Ground truth voxel tensor
        threshold: Tahmin edilen voxel'i binary'ye çevirmek için eşik değeri
        
    Returns:
        float: IoU değeri
    """
    # Tensörlerin uygun şekilde biçimlendirilmiş olduğundan emin olalım
    if pred_voxel.dim() != gt_voxel.dim():
        print(f"Uyarı: Tensör boyutları eşleşmiyor. pred_voxel: {pred_voxel.shape}, gt_voxel: {gt_voxel.shape}")
        if pred_voxel.dim() == 5 and gt_voxel.dim() == 4:  # Yaygın bir durum
            gt_voxel = gt_voxel.view(-1, 1, gt_voxel.size(1), gt_voxel.size(2), gt_voxel.size(3))
        elif pred_voxel.dim() == 4 and gt_voxel.dim() == 5:
            pred_voxel = pred_voxel.view(-1, 1, pred_voxel.size(1), pred_voxel.size(2), pred_voxel.size(3))
    
    # Tahmin edilen voxel'i binary'ye çevirme (eşik değerinden büyükse 1, değilse 0)
    pred_binary = (pred_voxel > threshold).float()
    gt_binary = (gt_voxel > threshold).float()  # GT'nin de binary olduğundan emin olalım
    
    # Kesişim ve birleşim hesaplama
    intersection = torch.sum(pred_binary * gt_binary).float()
    union = torch.sum(torch.clamp(pred_binary + gt_binary, 0, 1)).float()
    
    # Sıfıra bölme hatasını önlemek için kontrol
    if union.item() < 1e-6:
        return 0.0
        
    return (intersection / union).item()

def calculate_metrics(y_pred, y_true, threshold=0.5):
    """
    Discriminator için precision, recall ve F1 score hesaplar.
    
    Args:
        y_pred: Tahmin edilen değerler (discriminator çıktısı)
        y_true: Gerçek etiketler
        threshold: İkili sınıflandırma için eşik değeri
        
    Returns:
        tuple: (precision, recall, f1_score) değerleri
    """
    # Tahminleri binary'ye çevirme
    y_pred_binary = (y_pred >= threshold).float()
    
    # True Positive (TP), False Positive (FP), False Negative (FN) hesaplama
    tp = torch.sum(y_pred_binary * y_true).item()
    fp = torch.sum(y_pred_binary * (1 - y_true)).item()
    fn = torch.sum((1 - y_pred_binary) * y_true).item()
    
    # Sıfıra bölme hatasını önlemek için kontrol
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 Score hesaplama
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score