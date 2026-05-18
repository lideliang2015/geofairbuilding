# debug_metadata.py
from dataset import get_test_loader
from config import DATASET_ROOT, BATCH_SIZE, NUM_WORKERS

test_loader, metadata = get_test_loader(DATASET_ROOT, BATCH_SIZE, NUM_WORKERS)

print(f"metadata 类型: {type(metadata)}")
print(f"metadata 长度: {len(metadata) if hasattr(metadata, '__len__') else 'N/A'}")

# 取第一个 batch 查看
for batch in test_loader:
    if len(batch) == 3:
        images, masks, meta = batch
        print(f"\nbatch 中的 meta 类型: {type(meta)}")
        if isinstance(meta, dict):
            print(f"meta 的 keys: {meta.keys()}")
            for k, v in meta.items():
                print(f"  {k}: {type(v)}, length={len(v) if hasattr(v, '__len__') else 'N/A'}")
        elif isinstance(meta, (list, tuple)):
            print(f"meta 长度: {len(meta)}")
            if len(meta) > 0:
                print(f"meta[0] 类型: {type(meta[0])}")
                if isinstance(meta[0], dict):
                    print(f"meta[0] keys: {meta[0].keys()}")
    break