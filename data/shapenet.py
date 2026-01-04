# data/shapenet.py
import os
import numpy as np
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, List, Optional
from utils.geometry import CameraUtils
from torchvision import transforms

DEBUG_SAVE_DIR = "debug_projections"
os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)

# ==============================================
#  ShapeNet 좌표계 변환 행렬 (Source of Truth)
# ==============================================
SHAPENET_ROTATION_MATRIX = torch.tensor([
    [ 0,  1,  0,  0],
    [-1,  0,  0,  0],
    [ 0,  0,  1,  0],
    [ 0,  0,  0,  1]
], dtype=torch.float32)

class ShapeNetDataset(Dataset):
    """ShapeNet dataset loader (Base Class)"""
    
    def __init__(self, data_root: str, split: str = 'train',
                 num_source_views: int = 1, image_size: Tuple[int, int] = (128, 128),
                 z_near: float = 1.2, z_far: float = 4.0,
                 use_imagenet_normalize: bool = True):
        self.data_root = data_root
        self.split = split
        self.num_source_views = num_source_views
        self.image_size = image_size
        self.z_near = z_near
        self.z_far = z_far
        self.use_imagenet_normalize = use_imagenet_normalize
        
        if self.use_imagenet_normalize:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize = None
        
        split_dirs = {
            'train': 'cars_train',
            'val': 'cars_val',
            'test': 'cars_test'
        }
        self.split_dir = split_dirs[split]
        
        split_path = os.path.join(data_root, self.split_dir)
        # 폴더인지 확인
        self.instance_ids = sorted([
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        ])
        
        print(f"Loaded {len(self.instance_ids)} instances for {split} split")
    
    def __len__(self):
        return len(self.instance_ids)
    
    def _load_and_process_image(self, img_path: str, normalize: bool = False) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.image_size, Image.BILINEAR)
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = img.permute(2, 0, 1)
        
        if normalize and self.normalize is not None:
            img = self.normalize(img)
        
        return img
    
    def _parse_instance_intrinsics(self, intrinsics_path: str) -> Tuple[torch.Tensor, int]:
        with open(intrinsics_path, 'r') as f:
            lines = f.readlines()
        
        line1 = [float(x) for x in lines[0].strip().split()]
        fx, cx, cy = line1[0], line1[1], line1[2]
        
        if len(lines) >= 4:
            size_line = [int(x) for x in lines[3].strip().split()]
            original_size = size_line[0]
        else:
            original_size = 512
        
        intrinsics = torch.tensor([
            [fx, 0., cx],
            [0., fx, cy],
            [0., 0., 1.]
        ], dtype=torch.float32)
        
        return intrinsics, original_size


# 병렬 처리 작업자 함수 (NumPy + Early Pruning)
def process_single_instance(args):
    """
    단일 인스턴스 처리: NumPy 변환 및 조기 가지치기(Early Pruning) 적용으로 속도 대폭 향상
    """
    (inst_idx, instance_id, data_root, split_dir, 
     min_angle, max_angle, min_dist, max_dist,
     min_elev, max_elev) = args

    instance_dir = os.path.join(data_root, split_dir, instance_id)
    pose_dir = os.path.join(instance_dir, 'pose')
    
    if not os.path.exists(pose_dir):
        return [], 0, inst_idx

    # 파일 목록 가져오기 (I/O 최소화)
    pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith('.txt')])
    n_views = len(pose_files)
    
    if n_views < 2:
        return [], 0, inst_idx

    # -----------------------------------------------------------
    # Step 1. Pose 로드 (단순 텍스트 파싱으로 속도 개선)
    # -----------------------------------------------------------
    poses = []
    valid_indices_in_folder = []
    
    for i, p_file in enumerate(pose_files):
        try:
            p_path = os.path.join(pose_dir, p_file)
            # CameraUtils 대신 직접 읽어서 numpy로 변환
            with open(p_path, 'r') as f:
                lines = f.readlines()
                vals = []
                for line in lines:
                    vals.extend([float(x) for x in line.strip().split()])
            
            pose = np.array(vals).reshape(4, 4).astype(np.float32)
            poses.append(pose)
            
            # 파일명에서 인덱스 추출 (예: '000001.txt' -> 1)
            idx_from_name = int(p_file.split('.')[0])
            valid_indices_in_folder.append(idx_from_name)
        except:
            pass

    if len(poses) < 2:
        return [], 0, inst_idx

    # (N, 4, 4) NumPy Array로 변환
    poses = np.stack(poses, axis=0) 
    centers = poses[:, :3, 3] # (N, 3)

    # -----------------------------------------------------------
    # Step 2. [Node Filtering] 바닥면/정수리 뷰 미리 제거 (Early Pruning)
    # -----------------------------------------------------------
    object_center = np.zeros(3)
    
    # 2-1. 거리 계산
    radii = np.linalg.norm(centers - object_center, axis=1)
    dist_mask = (radii >= min_dist) & (radii <= max_dist)
    
    # 2-2. Elevation(고도) 계산
    radii_safe = np.maximum(radii, 1e-6)
    z_heights = centers[:, 2] # Z축을 높이로 가정
    
    sin_elevations = np.clip(z_heights / radii_safe, -1.0, 1.0)
    elevations_deg = np.arcsin(sin_elevations) * 180 / np.pi
    
    elev_mask = (elevations_deg >= min_elev) & (elevations_deg <= max_elev)
    
    # 최종 생존자 마스크
    valid_nodes_mask = dist_mask & elev_mask
    
    if np.sum(valid_nodes_mask) < 2:
        return [], 0, inst_idx

    # -----------------------------------------------------------
    # Step 3. [Edge Filtering] 살아남은 뷰들끼리만 각도 계산
    # -----------------------------------------------------------
    valid_subset_indices = np.where(valid_nodes_mask)[0]
    subset_centers = centers[valid_subset_indices]
    
    view_dirs = object_center - subset_centers
    norms = np.linalg.norm(view_dirs, axis=1, keepdims=True)
    view_dirs = view_dirs / np.maximum(norms, 1e-6)
    
    cos_angles = view_dirs @ view_dirs.T
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles_deg = np.arccos(cos_angles) * 180 / np.pi
    
    angle_valid_matrix = (angles_deg >= min_angle) & (angles_deg <= max_angle)
    np.fill_diagonal(angle_valid_matrix, False)
    
    # -----------------------------------------------------------
    # Step 4. 결과 매핑
    # -----------------------------------------------------------
    src_local, tgt_local = np.where(angle_valid_matrix)
    
    valid_pairs_inst = []
    
    # 로컬 인덱스(M) -> 원본 배열 인덱스(N) -> 실제 파일명 인덱스 매핑
    original_src_indices = valid_subset_indices[src_local]
    original_tgt_indices = valid_subset_indices[tgt_local]
    
    for s_idx, t_idx in zip(original_src_indices, original_tgt_indices):
        real_src = valid_indices_in_folder[s_idx]
        real_tgt = valid_indices_in_folder[t_idx]
        valid_pairs_inst.append((inst_idx, real_src, real_tgt))
    
    unique_views = len(np.unique(np.concatenate([original_src_indices, original_tgt_indices]))) if len(original_src_indices) > 0 else 0
    
    return valid_pairs_inst, unique_views, inst_idx


class FilteredShapeNetDataset(Dataset):
    """
    Multiprocessing & Caching이 적용된 최적화 데이터셋
    """
    
    def __init__(self, 
                 data_root: str, 
                 split: str = 'train',
                 num_source_views: int = 1,
                 image_size: Tuple[int, int] = (128, 128),
                 z_near: float = 1.2,
                 z_far: float = 4.0,
                 use_imagenet_normalize: bool = True,
                 enable_filtering: bool = True,
                 min_view_angle: float = 10,         
                 max_view_angle: float = 45,
                 min_camera_distance: float = 1.0,  
                 max_camera_distance: float = 2.0,
                 min_elevation: float = 5.0,
                 max_elevation: float = 85.0,
                 num_instances: Optional[int] = None,
                 repeat_factor: int = 1,
                 cache_dir: str = "./cache"):
        
        self.base_dataset = ShapeNetDataset(
            data_root=data_root,
            split=split,
            num_source_views=num_source_views,
            image_size=image_size,
            z_near=z_near,
            z_far=z_far,
            use_imagenet_normalize=use_imagenet_normalize
        )
        
        self.num_source_views = num_source_views
        
        # 인스턴스 샘플링
        if num_instances is not None:
            total = len(self.base_dataset.instance_ids)
            if num_instances < total:
                import random
                random.seed(42)
                self.base_dataset.instance_ids = sorted(random.sample(self.base_dataset.instance_ids, num_instances))
        
        self.repeat_factor = repeat_factor
        
        # 필터링 파라미터 저장
        self.min_view_angle = min_view_angle
        self.max_view_angle = max_view_angle
        self.min_camera_distance = min_camera_distance
        self.max_camera_distance = max_camera_distance
        self.min_elevation = min_elevation
        self.max_elevation = max_elevation
        
        self.cache_dir = cache_dir
        self.instance_view_counts = {} 
        self.valid_indices = [] 
        self.valid_pairs = []
        self.sample_to_instance_info = {}
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(
            self.cache_dir, 
            f"pairs_{split}_a{min_view_angle}-{max_view_angle}_d{min_camera_distance}-{max_camera_distance}_el{min_elevation}-{max_elevation}_n{len(self.base_dataset.instance_ids)}.pt"
        )

        if enable_filtering:
            if os.path.exists(self.cache_file):
                print(f"\n🚀 Loading cached pairs from {self.cache_file}")
                self._load_cache()
            else:
                self._filter_all_pairs_multiprocess()
                self._save_cache()
        else:
            self._create_all_pairs()
            
        self._build_sample_mapping()
    
    def _filter_all_pairs_multiprocess(self):
        print(f"\n{'='*70}")
        print(f"🔍 Multiprocessing Filtering ({cpu_count()} cores)")
        print(f"Angle=[{self.min_view_angle}, {self.max_view_angle}] | Dist=[{self.min_camera_distance}, {self.max_camera_distance}]")
        print(f"Elevation=[{self.min_elevation}, {self.max_elevation}] (Filtering out bottom/top views)") 
        
        tasks = []
        for inst_idx, instance_id in enumerate(self.base_dataset.instance_ids):
            tasks.append((
                inst_idx, 
                instance_id, 
                self.base_dataset.data_root, 
                self.base_dataset.split_dir,
                self.min_view_angle,
                self.max_view_angle,
                self.min_camera_distance,
                self.max_camera_distance,
                self.min_elevation, 
                self.max_elevation  
            ))
        
        self.valid_pairs = []
        
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(
                pool.imap(process_single_instance, tasks), 
                total=len(tasks),
                desc="Filtering",
                ncols=100
            ))
            
        for pairs, view_count, inst_idx in results:
            if pairs:
                self.valid_pairs.extend(pairs)
                self.instance_view_counts[inst_idx] = view_count
                self.valid_indices.append(inst_idx)
        
        print(f"  Valid pairs: {len(self.valid_pairs):,} / Valid Instances: {len(self.valid_indices)}")
        if len(self.valid_pairs) == 0:
            raise ValueError("❌ No valid pairs found! Check your elevation/angle thresholds.")

    def _save_cache(self):
        print(f"💾 Saving cache to {self.cache_file}...")
        cache_data = {
            'valid_pairs': self.valid_pairs,
            'instance_view_counts': self.instance_view_counts,
            'valid_indices': self.valid_indices
        }
        torch.save(cache_data, self.cache_file)

    def _load_cache(self):
        cache_data = torch.load(self.cache_file)
        self.valid_pairs = cache_data['valid_pairs']
        self.instance_view_counts = cache_data['instance_view_counts']
        self.valid_indices = cache_data['valid_indices']
        print(f"  Loaded {len(self.valid_pairs):,} pairs from cache.")

    def _create_all_pairs(self):
        self.valid_pairs = []
        print("⚠️ Filtering disabled: Using ALL pairs.")
        for inst_idx, instance_id in enumerate(self.base_dataset.instance_ids):
            instance_dir = os.path.join(self.base_dataset.data_root, self.base_dataset.split_dir, instance_id)
            image_dir = os.path.join(instance_dir, 'rgb')
            n_views = len([f for f in os.listdir(image_dir) if f.endswith('.png')])
            
            for src in range(n_views):
                for tgt in range(n_views):
                    if src != tgt:
                        self.valid_pairs.append((inst_idx, src, tgt))
            self.valid_indices.append(inst_idx)

    def _build_sample_mapping(self):
        if len(self.valid_indices) == 0:
            return
            
        for sample_idx in range(len(self)):
            instance_idx = sample_idx % len(self.valid_indices)
            actual_inst_idx = self.valid_indices[instance_idx]
            view_count = self.instance_view_counts.get(actual_inst_idx, 0)
            repeat_num = sample_idx // len(self.valid_indices)
            
            self.sample_to_instance_info[sample_idx] = (
                instance_idx, repeat_num, view_count
            )

    def __len__(self):
        return len(self.valid_indices) * self.repeat_factor
    
    def __getitem__(self, idx):
        # 1. Index Mapping
        instance_idx = idx % len(self.valid_indices)
        actual_inst_idx = self.valid_indices[instance_idx]
        
        # 2. Select Valid Pair
        instance_pairs = [p for p in self.valid_pairs if p[0] == actual_inst_idx]
        
        if not instance_pairs:
            # Fallback: Pick a random valid pair from the entire dataset
            import random
            inst_idx, src_idx, tgt_idx = random.choice(self.valid_pairs)
            actual_inst_idx = inst_idx
            instance_pairs = [p for p in self.valid_pairs if p[0] == actual_inst_idx]
        else:
            import random
            _, src_idx, tgt_idx = random.choice(instance_pairs)

        # 3. Select Extra Source Views
        src_view_indices = [src_idx]
        
        if self.num_source_views > 1:
            valid_view_pool = set()
            for _, s, t in instance_pairs:
                valid_view_pool.add(s)
                valid_view_pool.add(t)
            
            valid_view_pool.discard(src_idx)
            valid_view_pool.discard(tgt_idx)
            
            candidates_extra = list(valid_view_pool)
            
            # Helper to sample without replacement if possible
            def sample_views(candidates, k):
                if len(candidates) >= k:
                    return np.random.choice(candidates, size=k, replace=False).tolist()
                elif len(candidates) > 0:
                    return np.random.choice(candidates, size=k, replace=True).tolist()
                else:
                    return [src_idx] * k

            extra_needed = self.num_source_views - 1
            src_view_indices.extend(sample_views(candidates_extra, extra_needed))

        # 4. Data Loading
        instance_id = self.base_dataset.instance_ids[actual_inst_idx]
        instance_dir = os.path.join(self.base_dataset.data_root, self.base_dataset.split_dir, instance_id)
        image_dir = os.path.join(instance_dir, 'rgb')
        
        intrinsics_path = os.path.join(instance_dir, 'intrinsics.txt')
        intrinsics, original_size = self.base_dataset._parse_instance_intrinsics(intrinsics_path)
        
        # Scale intrinsics
        scale = self.base_dataset.image_size[0] / original_size
        intrinsics_scaled = intrinsics.clone()
        intrinsics_scaled[0, :] *= scale
        intrinsics_scaled[1, :] *= scale

        src_images = []
        src_poses = []
        src_intrinsics_list = []

        # --- Helper to safely load image and pose ---
        def load_view_data(view_idx):
            # Try loading exactly the file '0000XX.png'
            img_name = f'{view_idx:06d}.png'
            img_path = os.path.join(image_dir, img_name)
            
            # Safety check: if specific file is missing, find a replacement from the dir
            if not os.path.exists(img_path):
                 # Fallback: List all pngs and pick one (rare case but prevents crash)
                all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
                if len(all_files) > view_idx:
                     img_path = os.path.join(image_dir, all_files[view_idx])
                else:
                     img_path = os.path.join(image_dir, all_files[0]) # Absolute fallback
                
                # Update view_idx based on the actual file found to keep pose consistent
                # (This is a simplified fallback; ideally poses match images)
                real_idx = int(os.path.basename(img_path).split('.')[0])
                view_idx = real_idx

            return img_path, view_idx

        # Load Source Views
        for v_idx in src_view_indices:
            real_img_path, real_v_idx = load_view_data(v_idx)
            
            img = self.base_dataset._load_and_process_image(real_img_path, normalize=True)
            src_images.append(img)
            
            pose_path = os.path.join(instance_dir, 'pose', f'{real_v_idx:06d}.txt')
            pose = CameraUtils.parse_pose(pose_path)
            pose = torch.matmul(SHAPENET_ROTATION_MATRIX, pose)
            
            src_poses.append(pose)
            src_intrinsics_list.append(intrinsics_scaled)
            
        # Load Target View
        real_tgt_path, real_tgt_idx = load_view_data(tgt_idx)
        tgt_img = self.base_dataset._load_and_process_image(real_tgt_path, normalize=False)
        
        tgt_pose_path = os.path.join(instance_dir, 'pose', f'{real_tgt_idx:06d}.txt')
        tgt_pose = CameraUtils.parse_pose(tgt_pose_path)
        tgt_pose = torch.matmul(SHAPENET_ROTATION_MATRIX, tgt_pose)
        
        return {
            'src_images': torch.stack(src_images, dim=0),
            'src_poses': torch.stack(src_poses, dim=0),
            'src_intrinsics': torch.stack(src_intrinsics_list, dim=0),
            'tgt_image': tgt_img,
            'tgt_pose': tgt_pose,
            'tgt_intrinsics': intrinsics_scaled,
            'instance_id': instance_id
        }

def collate_fn(batch):
    return {
        'src_images': torch.stack([item['src_images'] for item in batch]),
        'src_poses': torch.stack([item['src_poses'] for item in batch]),
        'src_intrinsics': torch.stack([item['src_intrinsics'] for item in batch]),
        'tgt_image': torch.stack([item['tgt_image'] for item in batch]),
        'tgt_pose': torch.stack([item['tgt_pose'] for item in batch]),
        'tgt_intrinsics': torch.stack([item['tgt_intrinsics'] for item in batch]),
        'instance_ids': [item['instance_id'] for item in batch]
    }