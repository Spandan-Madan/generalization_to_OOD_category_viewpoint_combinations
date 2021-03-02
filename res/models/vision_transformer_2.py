from models.vit_pytorch.efficient import ViT
from linformer import Linformer

def random_function():
    print(1)

class vision_transformer_model():
    def __init__(self, num_classes = 1000):
        super(vision_transformer_model, self).__init__()
        efficient_transformer = Linformer(
            dim=128,
            seq_len=49+1,  # 7x7 patches + 1 cls-token
            depth=12,
            heads=8,
            k=64)

        transformer_model = ViT(
            dim=128,
            image_size=224,
            patch_size=32,
            num_classes=num_classes,
            transformer=efficient_transformer,
            channels=3)
