"""
地理分层采样 (Geographically Stratified Sampling - GSS)

核心思想：
根据地理区域（如大洲、HDI等级等）对训练样本进行加权采样，
提高弱势区域（如南美、非洲）的采样概率，缓解数据不平衡。

版本说明：
- v1（旧版）：GeographicallyStratifiedSampler，强制每个batch各大洲等量（已废弃，会导致亚洲样本被严重欠采样）
- v2（新版）：WeightedRandomSampler，只调整采样概率，不破坏训练分布（推荐）
"""

import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader, WeightedRandomSampler
from typing import List, Optional, Dict, Any, Iterator, Tuple
import warnings


# ============================================================
# 新版：WeightedRandomSampler（推荐使用）
# ============================================================

def create_weighted_sampler(
        dataset: Any,
        oversample_factor: float = 2.0,
        oversample_regions: Optional[List[str]] = None
) -> WeightedRandomSampler:
    """
    创建加权采样器（正确版GSS）

    只提高弱势区域样本的采样概率，不强制每个batch等量。
    这样既能让模型"多看几眼"弱势区域，又不破坏整体训练分布。

    Args:
        dataset: GeoFairDataset对象（需包含 metadata 属性）
        oversample_factor: 过采样倍数（默认2.0）
        oversample_regions: 需要过采样的区域列表（默认 Africa, South America）

    Returns:
        WeightedRandomSampler
    """
    if oversample_regions is None:
        oversample_regions = ['Africa', 'South America']

    metadata = dataset.metadata

    weights = []
    for _, row in metadata.iterrows():
        continent = row['continent']
        if continent in oversample_regions:
            weights.append(oversample_factor)
        else:
            weights.append(1.0)

    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # 统计信息
    from collections import Counter
    counts = Counter(metadata['continent'].tolist())
    print(f"✓ 加权采样器创建成功")
    print(f"  过采样区域: {oversample_regions} (×{oversample_factor})")
    print(f"  各大洲样本数:")
    for c, n in sorted(counts.items()):
        print(f"    {c}: {n}")

    return sampler


# ============================================================
# 旧版：GeographicallyStratifiedSampler（已废弃，保留仅供对比）
# ============================================================

class GeographicallyStratifiedSampler(Sampler):
    """
    【已废弃】地理分层采样器

    问题：强制每个batch各大洲等量采样，导致样本量大的区域（如Asia）被严重欠采样。
    请使用 create_weighted_sampler() 替代。
    """

    def __init__(
            self,
            dataset: Any,
            geo_labels: List[int],
            batch_size: int,
            strata_weights: Optional[Dict[int, float]] = None,
            shuffle: bool = True,
            drop_last: bool = False,
            seed: int = 42
    ):
        warnings.warn(
            "GeographicallyStratifiedSampler is deprecated and may cause "
            "severe under-sampling of large regions. "
            "Use create_weighted_sampler() instead.",
            DeprecationWarning
        )

        self.dataset = dataset
        self.geo_labels = np.array(geo_labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        self.strata = np.unique(geo_labels)
        self.n_strata = len(self.strata)

        if self.n_strata == 0:
            raise ValueError("No valid geo_labels provided")

        self.strata_indices = {
            s: np.where(geo_labels == s)[0].tolist()
            for s in self.strata
        }

        for s, indices in self.strata_indices.items():
            if len(indices) == 0:
                warnings.warn(f"Stratum {s} has no samples")

        if strata_weights is None:
            samples_per_stratum = batch_size // self.n_strata
            self.samples_per_stratum = {s: samples_per_stratum for s in self.strata}
        else:
            self.samples_per_stratum = {
                s: int(batch_size * strata_weights.get(s, 1.0 / self.n_strata))
                for s in self.strata
            }

        total_assigned = sum(self.samples_per_stratum.values())
        if total_assigned < batch_size:
            self.samples_per_stratum[self.strata[0]] += batch_size - total_assigned
        elif total_assigned > batch_size:
            self.samples_per_stratum[self.strata[-1]] -= (total_assigned - batch_size)

        self.generator = np.random.RandomState(seed)
        self._batches = self._build_batches()

    def _build_batches(self) -> List[List[int]]:
        all_batches = []
        total_samples = len(self.dataset)
        n_batches = total_samples // self.batch_size

        if not self.drop_last and total_samples % self.batch_size != 0:
            n_batches += 1

        for batch_idx in range(n_batches):
            batch_indices = []

            for stratum in self.strata:
                n_samples = self.samples_per_stratum[stratum]
                stratum_idx = self.strata_indices[stratum]

                if len(stratum_idx) == 0:
                    continue

                if len(stratum_idx) < n_samples:
                    sampled = self.generator.choice(
                        stratum_idx, n_samples, replace=True
                    ).tolist()
                else:
                    sampled = self.generator.choice(
                        stratum_idx, n_samples, replace=False
                    ).tolist()

                batch_indices.extend(sampled)

            if self.shuffle:
                self.generator.shuffle(batch_indices)

            all_batches.append(batch_indices)

        return all_batches

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self._batches:
            yield batch

    def __len__(self) -> int:
        return len(self._batches)


def create_balanced_dataloader(
        dataset: Any,
        geo_labels: List[int],
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True
) -> DataLoader:
    """
    【已废弃】创建地理平衡的数据加载器

    请使用 create_weighted_sampler() + 标准 DataLoader 替代。
    """
    warnings.warn(
        "create_balanced_dataloader is deprecated. "
        "Use create_weighted_sampler() instead.",
        DeprecationWarning
    )

    sampler = GeographicallyStratifiedSampler(
        dataset=dataset,
        geo_labels=geo_labels,
        batch_size=batch_size,
        shuffle=shuffle
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader


# ============================================================
# 辅助函数
# ============================================================

def get_region_weights_from_performance(
        region_performance: Dict[str, float],
        temperature: float = 1.0,
        min_weight: float = 0.5,
        max_weight: float = 3.0
) -> Dict[str, float]:
    """
    根据区域性能计算采样权重（性能越差，权重越高）
    """
    raw_weights = {region: 1.0 / (iou + 1e-6) for region, iou in region_performance.items()}
    mean_weight = np.mean(list(raw_weights.values()))
    normalized_weights = {r: w / mean_weight for r, w in raw_weights.items()}
    scaled_weights = {r: w ** temperature for r, w in normalized_weights.items()}
    clipped_weights = {
        r: np.clip(w, min_weight, max_weight)
        for r, w in scaled_weights.items()
    }
    total = sum(clipped_weights.values())
    final_weights = {r: w / total for r, w in clipped_weights.items()}
    return final_weights


def encode_geo_labels(
        geo_names: List[str],
        name_to_id: Optional[Dict[str, int]] = None
) -> Tuple[List[int], Dict[str, int]]:
    """
    将地理区域名称编码为整数ID
    """
    if name_to_id is None:
        unique_names = sorted(set(geo_names))
        name_to_id = {name: idx for idx, name in enumerate(unique_names)}

    encoded_ids = [name_to_id[name] for name in geo_names]

    return encoded_ids, name_to_id


def decode_geo_labels(
        encoded_ids: List[int],
        id_to_name: Dict[int, str]
) -> List[str]:
    """
    将编码ID解码为地理区域名称
    """
    return [id_to_name[idx] for idx in encoded_ids]