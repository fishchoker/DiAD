import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np
from tqdm import tqdm

model_path = "/root/autodl-tmp/clip-vit-large-patch14"
data_root = "/root/autodl-tmp/mvtecad"

classes_map = {
    'bottle': 'bottle', 'cable': 'cable', 'capsule': 'capsule', 'carpet': 'carpet',
    'grid': 'grid', 'hazelnut': 'hazelnut', 'leather': 'leather', 'metal_nut': 'metal nut',
    'pill': 'pill', 'screw': 'screw', 'tile': 'tile',
    'toothbrush': 'toothbrush', 'transistor': 'transistor', 'wood': 'wood surface', 'zipper': 'zipper'
}


def get_prompts(class_name):

    states = [
        f"{class_name}",
        f"flawless {class_name}",
        f"perfect {class_name}",
        f"unblemished {class_name}",
        f"{class_name} without flaw",
        f"{class_name} without defect",
        f"{class_name} without damage"
    ]

    templates = [
        "a cropped photo of the {c}.",
        "a cropped photo of a {c}.",
        "a close-up photo of a {c}.",
        "a close-up photo of the {c}.",
        "a bright photo of a {c}.",
        "a bright photo of the {c}.",
        "a dark photo of the {c}.",
        "a dark photo of a {c}.",
        "a jpeg corrupted photo of a {c}.",
        "a jpeg corrupted photo of the {c}.",
        "a blurry photo of the {c}.",
        "a blurry photo of a {c}.",
        "a photo of a {c}.",
        "a photo of the {c}.",
        "a photo of a small {c}.",
        "a photo of the small {c}.",
        "a photo of a large {c}.",
        "a photo of the large {c}.",
        "a photo of the {c} for visual inspection.",
        "a photo of a {c} for visual inspection.",
        "a photo of the {c} for anomaly detection.",
        "a photo of a {c} for anomaly detection."
    ]

    prompts = []
    prompt_states = []

    for s in states:
        for t in templates:
            prompt = t.format(c=s)
            prompts.append(prompt)
            prompt_states.append(s)

    return prompts, prompt_states, states


def extract_image_features(model, processor, image_paths, device, batch=32):

    feats = []

    for i in range(0, len(image_paths), batch):

        batch_paths = image_paths[i:i+batch]

        imgs = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = processor(images=imgs, return_tensors="pt").to(device)

        with torch.no_grad():

            f = model.get_image_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)

        feats.append(f)

    return torch.cat(feats, dim=0)


def compute_scores(model, processor, image_features, prompts, device):

    inputs = processor(
        text=prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sim = (image_features @ text_features.T).cpu().numpy()

    mean = sim.mean(axis=0)
    std = sim.std(axis=0)

    score = mean - std

    return score, mean, std


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)

    for cls_dir, cls_name in tqdm(classes_map.items()):

        image_dir = os.path.join(data_root, cls_dir, "train/good")

        image_files = os.listdir(image_dir)

        image_paths = [os.path.join(image_dir,f) for f in image_files]

        prompts, prompt_states, states = get_prompts(cls_name)

        image_features = extract_image_features(model, processor, image_paths, device)

        scores, means, stds = compute_scores(
            model,
            processor,
            image_features,
            prompts,
            device
        )

        # prompt level ranking
        prompt_info = list(zip(prompts, scores))
        prompt_info.sort(key=lambda x:x[1], reverse=True)

        # state level aggregation
        state_scores = {s:[] for s in states}

        for p_state, sc in zip(prompt_states, scores):
            state_scores[p_state].append(sc)

        final_state_scores = {}

        for s,v in state_scores.items():

            v = np.array(v)

            final_state_scores[s] = v.mean()

        top_states = sorted(
            final_state_scores.items(),
            key=lambda x:x[1],
            reverse=True
        )[:3]

        print("\n"+"="*70)
        print(cls_dir.upper())
        print("Top States:")

        for s,v in top_states:
            print(f"{s:<30} {v:.4f}")

        print("\nTop Prompts:")

        for i in range(3):
            p,sc = prompt_info[i]
            print(f"{sc:.4f}  {p}")


if __name__ == "__main__":
    main()