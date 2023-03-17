import dataIterator
from Config import Config
from torch.utils.data import DataLoader, Dataset

# We can load many often-used augmentations together 
# e.g., random crop, random vertical and horizontal flip, random contrast&brightness,...
from albumentations import (
    Compose, Rotate, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, VerticalFlip, HorizontalFlip, CenterCrop,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, HueSaturationValue, CoarseDropout
    )
from albumentations.pytorch import ToTensorV2

def img_augmentation():
  # define augmentations for images on train dataset
  train_transform = Compose(
      [
      Resize(Config.IMG_SIZE,Config.IMG_SIZE),
      CenterCrop(p=1.0,height=Config.CROP_SIZE,width=Config.CROP_SIZE),
      HorizontalFlip(p=0.5),
      VerticalFlip(p=0.5),
      RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
      HueSaturationValue(p=0.2, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
      ShiftScaleRotate(p=0.5),
      CoarseDropout(p=0.2),
      Normalize(
                  mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225],
              ),
              ToTensorV2(), # convert to tensor
      ])

  # define augmentations for images on validation dataset
  valid_transform = Compose([
      Resize(Config.IMG_SIZE,Config.IMG_SIZE),
      CenterCrop(p=1.0,height=Config.CROP_SIZE,width=Config.CROP_SIZE),
      Normalize(
          mean = [0.485, 0.456, 0.406],
          std = [0.229, 0.224, 0.225]
      ),
      ToTensorV2()    
  ])

  # define augmentations for images on test dataset
  test_transform = Compose([
      Resize(Config.IMG_SIZE,Config.IMG_SIZE),
      CenterCrop(p=1.0,height=Config.CROP_SIZE,width=Config.CROP_SIZE),
      Normalize(
          mean = [0.485, 0.456, 0.406],
          std = [0.229, 0.224, 0.225]
      ),
      ToTensorV2()    
  ])
  return (train_transform,valid_transform,test_transform)

def MyDataloader(train_meta_table,train_transform,val_meta_table,valid_transform,test_meta_table,test_transform):
# we need to make iterators for train, valid and test individually
  

  train_iterator = dataIterator.DiabeticIterator(train_meta_table,train_transform)
  valid_iterator = dataIterator.DiabeticIterator(val_meta_table,valid_transform)
  test_iterator = dataIterator.DiabeticIterator(test_meta_table,test_transform)
  
  print('length of train_iterator =',train_iterator.__len__()) # the total length of the train data
  print('length of valid_iterator =',valid_iterator.__len__()) # the total length of the validation data
  print('length of test_iterator =',test_iterator.__len__()) # the total length of the test data

  # we can wrap up iterators to create data loaders for batch loading
  # Remind: only train data needs to be shuffled!
  train_dataloader = DataLoader(train_iterator,batch_size=Config.BATCH_SIZE,shuffle=True,drop_last=True,num_workers=Config.NUM_WORKERS)
  valid_dataloader = DataLoader(valid_iterator,batch_size=Config.BATCH_SIZE,shuffle=False,drop_last=False,num_workers=Config.NUM_WORKERS)
  test_dataloader = DataLoader(test_iterator,batch_size=Config.BATCH_SIZE,shuffle=False,drop_last=False,num_workers=Config.NUM_WORKERS)

  return (train_dataloader,valid_dataloader,test_dataloader)

