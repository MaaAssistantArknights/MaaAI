import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


DEFAULT_CLASS_NAMES = ('c', 'n', 'y')
ERR_SUFFIX = '_err'


def resolve_canonical_class_name(folder_name, class_names=None, err_suffix=ERR_SUFFIX):
    class_names = tuple(class_names or DEFAULT_CLASS_NAMES)
    if folder_name in class_names:
        return folder_name, False

    if folder_name.endswith(err_suffix):
        base_name = folder_name[:-len(err_suffix)]
        if base_name in class_names:
            return base_name, True

    return None, False


def get_class_counts(root_dir, class_names=None, err_suffix=ERR_SUFFIX):
    class_names = list(class_names or DEFAULT_CLASS_NAMES)
    class_to_idx = {cls: index for index, cls in enumerate(class_names)}
    class_counts = torch.zeros(len(class_names), dtype=torch.float32)

    for folder_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, folder_name)
        if not os.path.isdir(class_dir):
            continue

        canonical_name, _ = resolve_canonical_class_name(folder_name, class_names, err_suffix)
        if canonical_name is None:
            continue

        image_count = sum(
            1 for filename in os.listdir(class_dir)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        class_counts[class_to_idx[canonical_name]] += image_count

    return class_counts

class SkillIconDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_names=None, err_suffix=ERR_SUFFIX):
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = list(class_names or DEFAULT_CLASS_NAMES)
        self.err_suffix = err_suffix
        self.classes = list(self.class_names)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.sample_is_hard_case = []
        self.samples = self._make_dataset()

    def _make_dataset(self):
        instances = []
        for folder_name in sorted(os.listdir(self.root_dir)):
            class_dir = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(class_dir):
                continue

            class_name, is_hard_case = resolve_canonical_class_name(
                folder_name,
                self.class_names,
                self.err_suffix
            )
            if class_name is None:
                continue

            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(class_dir, filename)
                    instances.append((path, self.class_to_idx[class_name]))
                    self.sample_is_hard_case.append(is_hard_case)
        return instances

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def _stratified_split_indices(samples, split_ratio, seed=42):
    label_to_indices = {}
    for index, (_, label) in enumerate(samples):
        label_to_indices.setdefault(label, []).append(index)

    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    val_indices = []

    for label in sorted(label_to_indices):
        indices = label_to_indices[label]
        shuffled = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
        train_size = int(len(shuffled) * split_ratio)
        train_indices.extend(shuffled[:train_size])
        val_indices.extend(shuffled[train_size:])

    return train_indices, val_indices


def _extract_labels(dataset):
    if isinstance(dataset, Subset):
        return [dataset.dataset.samples[index][1] for index in dataset.indices]
    return [label for _, label in dataset.samples]


def get_dataset_class_counts(dataset, num_classes):
    labels = torch.tensor(_extract_labels(dataset), dtype=torch.long)
    return torch.bincount(labels, minlength=num_classes).float()


def get_dataset_class_weights(dataset, num_classes):
    class_counts = get_dataset_class_counts(dataset, num_classes)
    safe_counts = class_counts.clamp_min(1.0)
    total_samples = class_counts.sum().clamp_min(1.0)
    return total_samples / (len(class_counts) * safe_counts)


def _extract_hard_case_flags(dataset):
    if isinstance(dataset, Subset):
        return [dataset.dataset.sample_is_hard_case[index] for index in dataset.indices]
    return dataset.sample_is_hard_case


def get_dataset_sample_weights(dataset, num_classes, err_folder_weight=1.0):
    labels = torch.tensor(_extract_labels(dataset), dtype=torch.long)
    class_weights = get_dataset_class_weights(dataset, num_classes)
    sample_weights = class_weights[labels]

    hard_case_flags = torch.tensor(_extract_hard_case_flags(dataset), dtype=torch.bool)
    if err_folder_weight > 1.0:
        sample_weights = sample_weights.clone()
        sample_weights[hard_case_flags] *= float(err_folder_weight)

    return sample_weights


def _build_weighted_sampler(dataset, err_folder_weight=1.0):
    labels = torch.tensor(_extract_labels(dataset), dtype=torch.long)
    class_counts = torch.bincount(labels).float().clamp_min(1.0)
    class_weights = class_counts.reciprocal()

    hard_case_flags = torch.tensor(_extract_hard_case_flags(dataset), dtype=torch.bool)
    hard_case_weights = torch.ones(len(labels), dtype=torch.float32)
    if err_folder_weight > 1.0:
        hard_case_weights[hard_case_flags] = float(err_folder_weight)

    sample_weights = class_weights[labels] * hard_case_weights
    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True
    )


def create_loaders(config, train_transform, val_transform):
    train_root = config['data']['dataset_path']
    val_root = config['data'].get('val_dataset_path')

    if val_root and os.path.isdir(val_root):
        train_dataset = SkillIconDataset(
            root_dir=train_root,
            transform=train_transform
        )
        val_dataset = SkillIconDataset(
            root_dir=val_root,
            transform=val_transform
        )
    else:
        full_dataset = SkillIconDataset(root_dir=train_root, transform=None)
        split_ratio = config['data']['split_ratio']
        train_indices, val_indices = _stratified_split_indices(
            full_dataset.samples,
            split_ratio,
            seed=42
        )
        train_dataset = Subset(
            SkillIconDataset(root_dir=train_root, transform=train_transform),
            train_indices
        )
        val_dataset = Subset(
            SkillIconDataset(root_dir=train_root, transform=val_transform),
            val_indices
        )

    sampler_config = config['training'].get('sampler', {})
    train_sampler = None
    if sampler_config.get('use_weighted_sampler', False):
        err_folder_weight = float(sampler_config.get('err_folder_weight', 1.0))
        train_sampler = _build_weighted_sampler(
            train_dataset,
            err_folder_weight=err_folder_weight
        )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config['training']['batch_size']) * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader
