import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import yaml
import math
try:
    from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test
except(ImportError):
    from .model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test


def loadConfig(configRoot):
    configPath = os.path.join(configRoot, "opts.yaml")
    with open(configPath, "r") as stream:
        config = yaml.load(stream)
    return config


def setGPU(sGpuids):
    """
    set GPU IDs into CUDA.

    Args:
        sGpuids (str): gpu ids in string (e.g.,"0,1,2")

    Returns:
        None
    """
    gpuids = []
    for sid in sGpuids.split(","):
        iid = int(sid)
        gpuids.append(iid)
    if len(gpuids) > 0:
        torch.cuda.set_device(gpuids[0])  # edit later
        cudnn.benchmark = True


def defineTransforms():
    """
    Define transform function. Edit if needed.
    """
    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return data_transforms


def defineDataloaders(query_ids, query_paths, gallery_ids, gallery_paths,
                      batch_size=256, shuffle=False, num_workers=0):
    """
    define dataloader with pre-defined transform functions
    and IO modules (Dataset class).
    """
    data_transforms = defineTransforms()
    image_datasets = {"query": Dataset(query_ids,
                                       query_paths,
                                       data_transforms),
                      "gallery": Dataset(gallery_ids,
                                         gallery_paths,
                                         data_transforms)}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
                   for x in ["query", "gallery"]}
    return dataloaders, image_datasets


def getScaleFromString(sScale):
    """
    get multiple scale from string argument

    Args:
        sScale (str): multiple scale

    Returns:
        list: scales in list
    """
    str_ms = sScale.split(",")
    ms = []
    for s in str_ms:
        sf = float(s)
        ms.append(math.sqrt(sf))
    return ms


def extractFeature(model, dataloaders, config, scale="1"):
    features = torch.FloatTensor()
    ms = getScaleFromString(scale)
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, 512).zero_().cuda()
        if config["PCB"]:
            ff = torch.FloatTensor(n, 2048, 6).zero_().cuda()
            # we have six parts

        input_img = Variable(img.cuda())
        for scale in ms:
            if scale != 1:
                # bicubic is only  available in pytorch>= 1.1
                input_img = nn.functional.interpolate(input_img,
                                                      scale_factor=scale,
                                                      mode='bicubic',
                                                      align_corners=False)
            outputs = model(input_img)
            ff += outputs
        # norm feature
        if config["PCB"]:
            # feature size (n,2048,6)
            # 1. To treat every part equally,
            # calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1,
            # sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    return features


def loadNetwork(network, stateDictPath):
    network.load_state_dict(torch.load(stateDictPath))
    return network


def loadModel(modelName, config, stateDictPath):
    nclasses = config["nclasses"]
    if modelName == "densenet121":
        model_structure = ft_net_dense(nclasses)
    elif modelName == "NAS":
        model_structure = ft_net_NAS(nclasses)
    elif modelName == "ResNet50":
        model_structure = ft_net(nclasses, stride=config["stride"])
    elif modelName == "PCB":
        model_structure = PCB(nclasses)
    else:
        raise KeyError("Undefined modelName, %s" % modelName)

    return loadNetwork(model_structure, stateDictPath)


def changeToEvalMode(model, modelname, usegpu=True):
    if modelname == "PCB":
        model = PCB_test(model)
    else:
        model.classifier.classifier = nn.Sequential()

    model = model.eval()
    if usegpu:
        model = model.cuda()
    return model


def extractFeatureInBatch(model, dataloaders, config,
                          gallery_label, gallery_cam,
                          query_label, query_cam):
    """
    Notes:
        Index order of each list and dataloaders must be identical.
    """
    with torch.no_grad():
        gallery_feature = extractFeature(model, dataloaders["gallery"], config)
        query_feature = extractFeature(model, dataloaders["query"], config)
    result = {"gallery_f": gallery_feature, "gallery_label": gallery_label,
              'gallery_cam': gallery_cam, 'query_f': query_feature,
              'query_label': query_label, 'query_cam': query_cam}
    return result


def calcScore(qf, gf):
    # cosine similarlity
    score = np.dot(gf, qf)
    return score


def calcPairwiseScores(gf):
    # cosine similarlity
    similarityMatrix = np.dot(gf, torch.t(gf))
    #print(similarityMatrix)
    return similarityMatrix


def match(similarityMatrix, ids, threshold=0.4):
    out = []
    for i in range(similarityMatrix.shape[0]):
        cand_idx = np.where(similarityMatrix[i, :] > threshold)[0]
        #print(cand_idx)
        matched = [ids[i]]
        for idx in set(cand_idx) - set([i]):
            #print(idx)
            #print(similarityMatrix[:, idx])
            max_idx = np.argsort(similarityMatrix[:, idx])[-2]
            if max_idx == i:
                matched.append(ids[idx])
        out.append(matched)
    return out


class Dataset(torch.utils.data.Dataset):
    """
    IO modules for pytorch models.
    """

    def __init__(self, ids, paths, transform=None):
        self.transform = transform
        self.ids = ids
        self.paths = paths

    def __getitem__(self, idx):
        data = self.__readImg(self.paths[idx])
        label = self.ids[idx]
        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.paths)

    def __readImg(self, path):
        from torchvision import get_image_backend
        if get_image_backend() == "accimage":
            return self.__accimageLoader(path)
        else:
            return self.__pilLoader(path)

    def __accimageLoader(self, path):
        import accimage
        try:
            return accimage.Image(path)
        except:
            return self.__pilLoader(path)

    def __pilLoader(self, path):
        with open(path, "rb") as stream:
            img = Image.open(stream)
            return img.convert("RGB")


class DatasetCV(torch.utils.data.Dataset):
    """
    IO modules for pytorch models with opencv objects.
    """

    def __init__(self, ids, cvimgs, transform=None):
        import cv2
        self.transform = transform
        self.ids = ids
        self.cvimgs = cvimgs

    def __len__(self):
        return len(self.cvimgs)

    def __getitem__(self, idx):
        data = self.__cv2pil(self.cvimgs[idx])
        label = self.ids[idx]
        if self.transform:
            data = self.transform(data)
        return data, label

    def __cv2pil(self, cvimg):
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(cvimg).convert("RGB")
        return pilimg


class Reidentify(object):
    """
    A higher API for re-identification. Edit if needed.
    """

    def __init__(self, modelName="ResNet50", usegpu=True, gpuids="0"):
        self.modelName = modelName
        self.usegpu = usegpu
        self.gpuids = gpuids
        self.epoch = 59  # which epoch
        self.threshold = 0.4
        cdir = os.getcwd()
        if os.path.exists(os.path.join(cdir, "deepreid/model/ft_%s") % modelName):
            self.configPath = os.path.join(cdir, "deepreid/model/ft_%s") % modelName
        elif os.path.exists(os.path.join(cdir, "/model/ft_%s") % modelName):
            self.configPath = os.path.join(cdir, "/model/ft_%s") % modelName
        else:
            raise IOError("A path model/ft_%s does not exist under current directory." % modelName)
        self.stateDictPath = os.path.join(self.configPath, "net_%d.pth" % self.epoch)
        print("Read configuration from: %s" % self.configPath)
        self.config = loadConfig(self.configPath)
        if usegpu:
            setGPU(gpuids)
        self.model = changeToEvalMode(loadModel(self.modelName, self.config, self.stateDictPath), self.modelName, usegpu=self.usegpu)

    def getInfoFromID(self, ids):
        info = [id.split("_") for id in ids]
        info = np.array(info).T
        camNames = info[0].tolist()
        camids = info[1].tolist()
        return camNames, camids

    def reidentify(self, ids, paths):
        camNames, camids = self.getInfoFromID(ids)
        # full search among every detected person including same image
        dataloaders, image_datasets = defineDataloaders(ids, paths, ids, paths)
        # predict feature
        result = extractFeatureInBatch(self.model, dataloaders, self.config, camids, camNames, camids, camNames)
        # use pairwise score calculation as gf.shape == qf.shape.
        similarityMatrix = calcPairwiseScores(result["gallery_f"])
        matchedIDs = match(similarityMatrix, ids, threshold=self.threshold)
        return matchedIDs
