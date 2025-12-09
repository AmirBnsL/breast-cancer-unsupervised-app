import os, json, cv2, numpy as np, torch, joblib
from torchvision import transforms
import timm
import torch
import torch.nn as nn
from timm.layers.helpers import to_2tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HistologyPatchDataset(Dataset):
    def __init__(self, data_dir, classes, max_patches_per_class=None, transform=None):
        self.samples = []  
        self.transform = transform

        for class_idx, class_name in enumerate(classes):
            folder_path = os.path.join(data_dir, class_name)
            if not os.path.exists(folder_path):
                continue
            image_files = [f for f in os.listdir(folder_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = 0
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = remove_black_padding(image)
                h, w, _ = image.shape
                for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
                    for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
                        patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                        if not is_patch_usable(patch):
                            continue
                        self.samples.append((img_path, x, y, class_idx))
                        count += 1
                        if max_patches_per_class is not None and count >= max_patches_per_class:
                            break
                    if max_patches_per_class is not None and count >= max_patches_per_class:
                        break
            print(f"Collected {count} patch coords for class {class_name}")

        # cache full images to avoid re-reading each time (optional)
        self._image_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, x, y, label = self.samples[idx]
        if img_path not in self._image_cache:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = remove_black_padding(img)
            self._image_cache[img_path] = img
        image = self._image_cache[img_path]
        patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        if self.transform is not None:
            patch = self.transform(patch)
        else:
            patch = torch.from_numpy(patch.astype("float32") / 255.0).permute(2, 0, 1)

        return patch, label

class ConvStem(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        **kwargs,           
    ):
        super().__init__()
        assert patch_size == 4, "Patch size must be 4"
        assert embed_dim % 8 == 0, "Embed dim must be divisible by 8"

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        stem = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for _ in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2,
                                  padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


def build_ctranspath_backbone():
    # use HF hub alias so timm loads correct cfg and pretrained weights
    model = timm.create_model(
        "hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath",
        pretrained=True,
        num_classes=0,
        embed_layer=ConvStem,
    )
    return model

class CTransPathWithHead(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = build_ctranspath_backbone()
        feat_dim = self.backbone.num_features  # usually 768 for swin_tiny CTransPath[web:41]
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)        # (B, D)
        logits = self.head(feats)       # (B, num_classes)
        return logits, feats

def remove_black_padding(image, threshold=10):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]
def is_patch_usablle(patch, tissue_threshold=0.2):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    tissue_ratio = np.count_nonzero(mask) / mask.size
    return tissue_ratio > tissue_threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./saved_models_ctranspath_rf"

# load meta
with open(os.path.join(SAVE_DIR, "meta.json")) as f:
    meta = json.load(f)
CLASSES = meta["classes"]
PATCH_SIZE = meta["patch_size"]

# rebuild transform exactly as in training
ctranspath_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.0),  # TURN OFF RANDOM AUGS FOR TEST
    transforms.RandomVerticalFlip(p=0.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=meta["normalize_mean"],
                         std=meta["normalize_std"]),
])

# rebuild model and load weights
model = CTransPathWithHead(num_classes=len(CLASSES)).to(DEVICE)
state = torch.load(os.path.join(SAVE_DIR, "ctranspath_with_head_state_dict.pth"),
                   map_location=DEVICE)
model.load_state_dict(state)
model.eval()
backbone = model.backbone  # used to extract embeddings

# load RF
rf_clf = joblib.load(os.path.join(SAVE_DIR, "rf_on_ctranspath_embeddings.joblib"))

def predict_slide(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = remove_black_padding(image)       # same as training
    h, w, _ = image.shape

    patch_feats = []
    for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if not is_patch_usablle(patch):   # same tissue filter as training
                continue

            inp = ctranspath_transform(patch).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feats = backbone(inp)        # (1, D)
            vec = feats.cpu().numpy().reshape(1, -1)
            patch_feats.append(vec)

    if len(patch_feats) == 0:
        raise ValueError("No usable patches extracted from image")

    X_patches = np.concatenate(patch_feats, axis=0)        # (N_patches, D)
    patch_preds = rf_clf.predict(X_patches)                # per‑patch labels

    # majority vote for slide label
    counts = np.bincount(patch_preds, minlength=len(CLASSES))
    slide_idx = counts.argmax()
    slide_label = CLASSES[slide_idx]
    return slide_idx, slide_label, patch_preds

if __name__ == "__main__":
  test_img = "/content/BRACS_264_N_4.png"
  idx, label, patch_preds = predict_slide(test_img)
  print("Predicted slide label:", idx, label)
