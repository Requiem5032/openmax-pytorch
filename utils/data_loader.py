import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import transform
from utils.data_loader_utils import *
from utils.data_split import get_species_map
from utils.preprocess import alb_transform_train, alb_transform_test


class MosDataset(Dataset):
    def __init__(self, config, data_df, num_species, transformer=None):
        super().__init__()
        self.tfms = None
        self.config = config
        self.num_species = num_species
        self.transformer = transformer
        self.mixup = config.mixup
        self.images_df = data_df

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        imagename = self.images_df.loc[idx, 'Id']

        image = load_image(imagename)
        image = make_square(image)

        label = self.images_df['Species'][idx]

        if self.transformer:
            image = self.transformer(image=image)['image']

        else:
            image = transform.resize(
                image, (self.config.imsize, self.config.imsize))

        image = torch.from_numpy(image).permute(-1, 0, 1).float()

        return image, label

    def getimage(self, idx):
        image, targets = self.__getitem__(idx)
        image = image.permute(1, 2, 0).numpy()
        imagename = self.images_df.loc[idx, 'Id']
        return image, targets, imagename

    def cmix(self, idx):
        return mixup_loader(idx, self.images_df, self.dataset, self.colors)


def get_data_loaders(config, eval_mode=False, get_dataset=False, train_df=None,
                     valid_df=None, DatasetClass=MosDataset):
    if train_df is None:
        print("Reading data split from {}".format(config.DATA_CSV_PATH))
        data_df = pd.read_csv(config.DATA_CSV_PATH)

        train_df = data_df[data_df['Split'] == 'train'].reset_index(drop=True)
        valid_df = data_df[data_df['Split'] == 'val'].reset_index(drop=True)

        if config.known_only:
            train_df = train_df[train_df['Species'].apply(
                lambda x: x not in config.unknown_classes)].reset_index(drop=True)
            valid_df = valid_df[valid_df['Species'].apply(
                lambda x: x not in config.unknown_classes)].reset_index(drop=True)

    if config.debug and config.reduce_dataset:
        train_df = train_df.loc[:200]

    if eval_mode:
        train_tf = alb_transform_test(config.imsize)
    else:
        print("Data Augmemtation with probability ", config.augment_prob)
        train_tf = alb_transform_train(config.imsize, p=config.augment_prob)
    valid_tf = alb_transform_test(config.imsize)

    # set up the datasets
    train_dataset = DatasetClass(
        config=config, data_df=train_df, num_species=config.num_species.sum(), transformer=train_tf)
    valid_dataset = DatasetClass(
        config=config, data_df=valid_df, num_species=config.num_species.sum(), transformer=valid_tf)

    if get_dataset:
        return train_dataset, valid_dataset

    train_sampler = SubsetRandomSampler(range(len(train_dataset)))

    # set up the data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              sampler=train_sampler,
                              num_workers=config.num_workers,
                              pin_memory=True,
                              drop_last=False)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=config.num_workers,
                              pin_memory=True,
                              drop_last=False)

    return train_loader, valid_loader


def get_test_loader(config, test_df=None, DatasetClass=None):
    '''sets up the torch data loaders for testing'''
    if test_df is None:
        data_df = pd.read_csv(config.DATA_CSV_PATH)
        test_df = data_df[data_df['Split'].apply(
            lambda x: 'test' in x)].reset_index(drop=True)
        
    class_map = get_species_map(test_df)
    
    test_tf = alb_transform_test(config.imsize)

    # test_tf = test_transformer(imsize)

    # set up the datasets
    if DatasetClass is None:
        DatasetClass = MosDataset
    test_dataset = DatasetClass(
        config=config, data_df=test_df, num_species=21, transformer=test_tf)

    # set up the data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers,
                             pin_memory=True,
                             drop_last=False)

    return test_loader, class_map
