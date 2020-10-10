import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

# To check if the file is a valid image type.
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

# Provide valid crop size. Returns multiple of upscale_factor
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

# Transform for High res images. Only RandomCrop to desired crop size.
def train_HR_transform(crop_size):
    return Compose([
                    RandomCrop(crop_size),
                    ToTensor()
    ])

# Transform for Low res images. Uses the randomly cropped high res images, downsizes by upscale_factor.
def train_LR_transform(crop_size, upscale_factor):
    return Compose([
                    ToPILImage(),
                    Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
                    ToTensor()
    ])

# Transfor to display images.
def display_transform():
    return Compose([
                    ToPILImage(),
                    Resize(400),
                    CenterCrop(400),
                    ToTensor()
    ])

class TrainDatasetFromFolder(Dataset):
    '''
    Dataset class to provide images for training.
    Output:
    High res images (random cropped) 
    low res images (resized by up_scale factor).
    '''
    def __init__(self, dataset_directory, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(dataset_directory, x) for x in os.listdir(dataset_directory) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.HR_transform = train_HR_transform(crop_size)
        self.LR_transform = train_LR_transform(crop_size, upscale_factor)
    
    def __getitem__(self, index):
        HR_image = self.HR_transform(Image.open(self.image_filenames[index]))
        LR_image = self.LR_transform(HR_image)
        return LR_image, HR_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    '''
    Dataset class to provide images for Validation.
    Output:
    High res images (Centercropped image)
    Restored high res images (resized LR images)
    Low res images (downsized HR images)
    '''
    def __init__(self, dataset_directory, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [os.path.join(dataset_directory, x) for x in os.listdir(dataset_directory) if is_image_file(x)]

    def __getitem__(self, index):
        HR_image = Image.open(self.image_filenames[index])
        w, h = HR_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        LR_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        HR_scale = Resize(crop_size, interpolation=Image.BICUBIC)

        HR_image = CenterCrop(crop_size)(HR_image)
        LR_image = LR_scale(HR_image)
        HR_restored_image = HR_scale(LR_image)
        return ToTensor()(LR_image), ToTensor()(HR_restored_image), ToTensor()(HR_image)

    def __len__(self):
        return len(self.image_filenames)