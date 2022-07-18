from PIL import ImageOps, ImageEnhance, ImageFilter, Image, ImageDraw
import torchvision.transforms.functional as F
aug_img = Image.new("RGB", (256, 256), (255, 255, 255))
print(F.to_tensor(aug_img))