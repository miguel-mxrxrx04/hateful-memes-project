from pathlib import Path
from typing import Dict, List, Optional, Union
import os 
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class HatefulMemesDataset(Dataset):
    """
    Dataset híbrido para el Hateful Memes Challenge optimizado para restricciones de VRAM.
    
    Carga imágenes en streaming local y vectorizaciones de texto en memoria.
    Soporta submuestreo reproducible para prototipado rápido y flags de Zero-ing
    para auditoría causal de sesgo modal.
    """
    def __init__(
        self,
        jsonl_path: Union[str, Path],
        img_dir: Union[str, Path] = "data/raw/img",
        features_dir: Union[str, Path] = "data/features",
        is_train: bool = True,
        max_samples: Optional[int] = None,
        seed: int = 42,
        zero_text: bool = False,
        zero_image: bool = False,
        img_size: int = 224
    ) -> None:
        
        self.img_dir = Path(img_dir)
        self.features_dir = Path(features_dir)
        self.zero_text = zero_text
        self.zero_image = zero_image
        self.img_size = img_size

        self.samples = List[Dict]
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line.strip()))

        if max_samples is not None and max_samples < len(self.samples):
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(self.samples), generator=generator)[:max_samples].to_list()
            

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_name = os.path.basename(sample['img'])
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        feature_path = os.path.join(self.features_dir, f"{sample['id']}.pt")
        text_features = torch.load(feature_path, weights_only=True).squeeze()

        label = torch.tensor(sample['label'], dtype=torch.float32)

        return image, text_features, label