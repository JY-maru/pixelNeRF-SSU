import sys
import os
import warnings
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import imageio
import glob
import cv2
import shutil
import random
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

sys.path.append(os.getcwd())

from config.config import Config
from models.pixelnerf import PixelNeRF
from utils.geometry import CameraUtils

from data.shapenet import SHAPENET_ROTATION_MATRIX


def parse_pose_file(path):
    with open(path, 'r') as f:
        values = [float(x) for x in f.read().split()]
    return torch.tensor(values, dtype=torch.float32).reshape(4, 4)

def parse_shapenet_intrinsic(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    all_values = []
    for line in lines:
        all_values.extend([float(x) for x in line.split()])
    if len(all_values) == 9:
        return torch.tensor(all_values, dtype=torch.float32).reshape(3, 3)
    elif len(all_values) >= 3:
        focal, cx, cy = all_values[0], all_values[1], all_values[2]
        intrinsic = torch.eye(3, dtype=torch.float32)
        intrinsic[0, 0] = focal
        intrinsic[1, 1] = focal
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy
        return intrinsic
    else:
        return torch.eye(3, dtype=torch.float32)

def load_data(folder_path, target_size, device='cuda'):
    print(f"📂 Loading data from: {folder_path} (Size: {target_size})", flush=True)
    rgb_dir = os.path.join(folder_path, 'rgb')
    pose_dir = os.path.join(folder_path, 'pose')
    intrinsics_dir = os.path.join(folder_path, 'intrinsics')
    global_intrinsic_path = os.path.join(folder_path, 'intrinsics.txt')

    img_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")) + glob.glob(os.path.join(rgb_dir, "*.jpg")))
    pose_paths = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    
    if not img_paths: raise FileNotFoundError(f"❌ No images found in {rgb_dir}")
    num_views = min(len(img_paths), len(pose_paths))
    
    use_per_view_intrinsic = False
    intrinsic_paths = []
    if os.path.isdir(intrinsics_dir):
        intrinsic_paths = sorted(glob.glob(os.path.join(intrinsics_dir, "*.txt")))
        if len(intrinsic_paths) > 0: use_per_view_intrinsic = True

    transform = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])
    src_images, src_poses, src_intrinsics = [], [], []

    for i in range(num_views):
        img = Image.open(img_paths[i]).convert("RGB")
        src_images.append(transform(img))
        pose = parse_pose_file(pose_paths[i])
        pose = torch.matmul(SHAPENET_ROTATION_MATRIX, pose)
        src_poses.append(pose)
        
        if use_per_view_intrinsic and i < len(intrinsic_paths):
            intrinsic = parse_shapenet_intrinsic(intrinsic_paths[i])
        elif os.path.exists(global_intrinsic_path):
            intrinsic = parse_shapenet_intrinsic(global_intrinsic_path)
        else:
            orig_w, orig_h = img.size
            focal = 0.5 * orig_w / np.tan(0.5 * np.deg2rad(50))
            intrinsic = torch.tensor([[focal, 0, orig_w/2], [0, focal, orig_h/2], [0, 0, 1]], dtype=torch.float32)

        scale_x = target_size[1] / img.size[0]
        scale_y = target_size[0] / img.size[1]
        intrinsic_scaled = intrinsic.clone()
        intrinsic_scaled[0, 0] *= scale_x
        intrinsic_scaled[1, 1] *= scale_y
        intrinsic_scaled[0, 2] *= scale_x
        intrinsic_scaled[1, 2] *= scale_y
        src_intrinsics.append(intrinsic_scaled)

    return {
        'src_images': torch.stack(src_images).to(device),       
        'src_poses': torch.stack(src_poses).to(device),         
        'src_intrinsics': torch.stack(src_intrinsics).to(device)
    }

class PixelNeRFInference:
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing Model on {self.device}...", flush=True)
        
        self.model = PixelNeRF(
            encoder_type=config.model.encoder_type, encoder_pretrained=False, 
            feature_dim=config.model.feature_dim, d_hidden=config.model.d_hidden,
            n_blocks=config.model.n_blocks, combine_type=config.model.combine_type,
            n_coarse=config.model.n_coarse, n_fine=config.model.n_fine,
            white_bkgd=config.model.white_bkgd, use_pe=config.model.use_pe,
            pe_freq_pos=config.model.pe_freq_pos, pe_freq_dir=config.model.pe_freq_dir
        ).to(self.device)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(self.device)
        
        ckpt_path = checkpoint_path if checkpoint_path else config.inference.checkpoint_path
        self._load_checkpoint(ckpt_path)

    def _load_checkpoint(self, path):
        if not os.path.exists(path): raise FileNotFoundError(f"❌ Checkpoint not found: {path}")
        
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError: 
            checkpoint = torch.load(path, map_location=self.device)
        
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        
        print("✅ Weights loaded successfully")

    def optimize(self, src_data):
        steps = self.config.tto.num_steps
        lr = self.config.tto.learning_rate
        batch_rays = self.config.tto.batch_rays
        print(f"\n[TTO] Starting Optimization for {steps} steps...", flush=True)
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        images = src_data['src_images']
        poses = src_data['src_poses']
        intrinsics = src_data['src_intrinsics']
        
        N, _, H, W = images.shape
        coords = torch.stack(torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device)), -1).reshape(-1, 2)
        
        mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(self.device))
        pbar = tqdm(range(steps), desc="    Optimizing", leave=True, file=sys.stdout)
        
        for step in pbar:
            img_idx = np.random.randint(0, N)
            select_inds = torch.randperm(H * W, device=self.device)[:batch_rays]
            select_coords = coords[select_inds]
            
            target_rgb = images[img_idx, :, select_coords[:, 0], select_coords[:, 1]].T
            
            rays_o, rays_d = CameraUtils.get_rays(H, W, intrinsics[img_idx], poses[img_idx])
            rays_o = rays_o.reshape(-1, 3)[select_inds].unsqueeze(0)
            rays_d = rays_d.reshape(-1, 3)[select_inds].unsqueeze(0)
            norm_images = self.normalize(images)
            
            out = self.model(norm_images.unsqueeze(0), intrinsics.unsqueeze(0), poses.unsqueeze(0), rays_o, rays_d, z_near=self.config.data.z_near, z_far=self.config.data.z_far)
            
            # === [Loss 계산: train.py 로직 적용] ===
            
            # 1. MSE Loss (기본 화질)
            mse_loss = F.mse_loss(out['fine']['rgb_map'].squeeze(0), target_rgb)
            
            loss = mse_loss 
            # ========================================
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                with torch.no_grad():
                    psnr = mse2psnr(mse_loss) # PSNR은 순수 화질(MSE)로만 표기
                    pbar.set_description(f"   Optimizing | Loss: {loss.item():.4f} | PSNR: {psnr.item():.2f} dB")
                    
        print("[TTO] Finished!")
        self.model.eval()

    def render_batch(self, src_data, tgt_poses, tgt_intrinsic, H, W, desc="Rendering"):
        frames = []
        src_images = self.normalize(src_data['src_images']).unsqueeze(0)        
        src_poses = src_data['src_poses'].unsqueeze(0)          
        src_intrinsics = src_data['src_intrinsics'].unsqueeze(0)
        
        chunk_size = getattr(self.config.inference, 'chunk_size', 4096)
        if H > 256: chunk_size = 1024 

        z_near = self.config.data.z_near
        z_far = self.config.data.z_far

        for pose in tqdm(tgt_poses, desc=desc, file=sys.stdout, ncols=100):
            rays_o, rays_d = CameraUtils.get_rays(H, W, tgt_intrinsic, pose)
            rays_o = rays_o.reshape(-1, 3).unsqueeze(0) 
            rays_d = rays_d.reshape(-1, 3).unsqueeze(0)
            
            rgb_chunks = []
            acc_chunks = []
            
            for i in range(0, rays_o.shape[1], chunk_size):
                chunk_o = rays_o[:, i:i+chunk_size]
                chunk_d = rays_d[:, i:i+chunk_size]
                with torch.no_grad():
                    out = self.model(src_images, src_intrinsics, src_poses, chunk_o, chunk_d, z_near, z_far)
                
                rgb_chunks.append(out['fine']['rgb_map'].cpu())
                acc_chunks.append(out['fine']['acc_map'].cpu())
            
            # 1. 텐서 합치기
            pred_rgb = torch.cat(rgb_chunks, dim=1).reshape(H, W, 3)
            pred_acc = torch.cat(acc_chunks, dim=1).reshape(H, W, 1)
            
            # ==============================================================
            # [복구] 표준 화이트 배경 처리 (Alpha Blending)
            # ==============================================================
            # Hard Masking(잘라내기)을 제거하고, NeRF의 표준 공식으로 돌아갑니다.
            # 공식: 결과색 = 객체색 + (1 - 투명도) * 흰색
            
            # 1. 배경이 투명할수록(acc가 낮을수록) 흰색(1.0)을 더합니다.
            # 2. 부드럽게 그라데이션되므로 끊기는 아티팩트가 사라집니다.
            RGB_WHITHE = 1.0
            pred_img = pred_rgb + (1.0 - pred_acc) * RGB_WHITHE
            
            # ==============================================================
            
            # 3. 값 클램핑 및 변환
            pred_img = torch.clamp(pred_img, 0, 1).numpy()
            pred_img_uint8 = (pred_img * 255).astype(np.uint8)
            
            frames.append(pred_img_uint8)
            torch.cuda.empty_cache()
            
        return frames

    def look_at(self, eye, center, up):
        forward = center - eye; forward /= np.linalg.norm(forward)
        right = np.cross(forward, up); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        pose = np.eye(4); pose[:3, 0] = right; pose[:3, 1] = down; pose[:3, 2] = forward; pose[:3, 3] = eye
        return pose

    def get_orbit_poses(self, num_frames, radius, center, elevation=30):
        poses = []
        # Elevation angle (phi)
        phi = np.deg2rad(90 - elevation)
        world_up = np.array([0, 0, 1])
        
        # 좌측면에서 시작 (초기 위상 조정 0으로 복구)
        azimuths = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
        phase_shift = 0  
        
        for theta in azimuths:
            theta_shifted = theta + phase_shift
            
            x = radius * np.sin(phi) * np.cos(theta_shifted)
            y = radius * np.sin(phi) * np.sin(theta_shifted)
            z = radius * np.cos(phi)
            
            cam_pos = np.array([x, y, z]) + center
            pose = self.look_at(cam_pos, center, world_up)
            pose_tensor = torch.from_numpy(pose).float()
            poses.append(torch.matmul(SHAPENET_ROTATION_MATRIX, pose_tensor))
            
        return torch.stack(poses).to(self.device)

    def run(self, input_folder, output_dir, mode='video', threshold=25.0, **kwargs):
        if self.config.tto.enabled:
            tto_h, tto_w = self.config.tto.image_size
            tto_data = load_data(input_folder, (tto_h, tto_w), self.device)
            self.optimize(tto_data)
            src_data = tto_data 
            render_input_size = (tto_h, tto_w)
        else:
            render_input_size = self.config.data.image_size
            src_data = load_data(input_folder, render_input_size, self.device)

        print(f"\n[Rendering] Generating Clean 360 Orbit Video...")
        render_size = kwargs.get('render_size') or render_input_size[0]
        num_frames = kwargs.get('num_frames') or 60
        
        cli_fps = kwargs.get('fps')
        config_fps = getattr(self.config.inference, 'video_fps', 30)
        final_fps = cli_fps if cli_fps is not None else config_fps

        src_center = np.array([0.0, 0.0, 0.0])
        cam_positions = src_data['src_poses'][:, :3, 3].cpu().numpy()
        radius = np.mean(np.linalg.norm(cam_positions, axis=1))
        
        # Create orbit path
        tgt_poses = self.get_orbit_poses(num_frames, radius, src_center, elevation=40)
        
        tgt_intrinsic = src_data['src_intrinsics'][0].clone()
        scale = render_size / render_input_size[0]
        tgt_intrinsic[:2] *= scale
        
        # Render
        frames = self.render_batch(src_data, tgt_poses, tgt_intrinsic, render_size, render_size, desc=f"Rendering Loop")
        
        # [핵심 수정] obj_id가 있으면 그것을 사용, 없으면 폴더명 사용
        passed_obj_id = kwargs.get('obj_id')
        instance_name = passed_obj_id if passed_obj_id else os.path.basename(os.path.normpath(input_folder))
        
        num_input_views = src_data['src_images'].shape[0]  # 입력 뷰 갯수 계산
        
        if mode == 'video':
            # 예: orbit_d34b049..._128_6.mp4
            save_filename = f"orbit_{instance_name}_{render_size}_{num_input_views}.mp4"
            save_path = os.path.join(output_dir, save_filename)
            
            imageio.mimsave(save_path, frames, fps=final_fps, quality=9)
            print(f"✨ Orbit Video saved to: {save_path} (FPS: {final_fps})")
            
        elif mode == 'views':
            save_root = os.path.join(output_dir, instance_name, 'views')
            os.makedirs(save_root, exist_ok=True)
            for i, frame in enumerate(frames):
                imageio.imwrite(os.path.join(save_root, f"{i:06d}.png"), frame)
            print(f"✨ Views (Images) saved to: {save_root}")


CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_FILE_DIR)

warnings.filterwarnings("ignore")

class DataPreprocessor:
    def __init__(self, source_root, target_root, output_dir_name):
        self.source_root = source_root
        self.target_root = target_root
        self.output_dir_name = output_dir_name

    def _clean_and_setup_dirs(self):
        if os.path.exists(self.target_root):
            try:
                shutil.rmtree(self.target_root)
                print(f"🧹 Cleared: {self.target_root}")
            except Exception as e:
                print(f"❌ Error clearing directory: {e}")
                return None
        os.makedirs(self.target_root, exist_ok=True)
        final_target_path = os.path.join(self.target_root, self.output_dir_name)
        os.makedirs(final_target_path, exist_ok=True)
        return final_target_path

    def _get_elevation_from_pose(self, pose_path):
        try:
            with open(pose_path, 'r') as f: values = [float(x) for x in f.read().split()]
            pose = np.array(values).reshape(4, 4)
            loc = pose[:3, 3]
            radius = np.linalg.norm(loc)
            return 0.0 if radius < 1e-6 else np.degrees(np.arcsin(loc[2] / radius))
        except Exception as e:
            print(f"⚠️ Pose error {pose_path}: {e}")
            return 999.0

    def prepare_data(self, num_views=6, max_elevation=50.0, specific_obj_id=None):
        dst_obj_path = self._clean_and_setup_dirs()
        if dst_obj_path is None: return None, None
        
        if not os.path.exists(self.source_root):
            print(f"❌ Source not found: {self.source_root}"); return None, None

        if specific_obj_id:
            obj_id = specific_obj_id
            src_obj_path = os.path.join(self.source_root, obj_id)
            if not os.path.exists(src_obj_path): print(f"❌ ID '{obj_id}' not found."); return None, None
        else:
            objects = [d for d in os.listdir(self.source_root) if os.path.isdir(os.path.join(self.source_root, d))]
            if not objects: print("❌ No objects found."); return None, None
            obj_id = random.choice(objects)
            src_obj_path = os.path.join(self.source_root, obj_id)

        all_indices = sorted([int(os.path.splitext(f)[0]) for f in os.listdir(os.path.join(src_obj_path, 'rgb')) if f.endswith(('.png', '.jpg'))])
        valid_indices = [idx for idx in all_indices if abs(self._get_elevation_from_pose(os.path.join(src_obj_path, 'pose', f"{idx:06d}.txt"))) <= max_elevation]
        
        selected_indices = sorted(random.sample(valid_indices, min(len(valid_indices), num_views))) if len(valid_indices) >= num_views else valid_indices
        
        for subdir in ['rgb', 'pose', 'intrinsics']: os.makedirs(os.path.join(dst_obj_path, subdir), exist_ok=True)
        if os.path.exists(os.path.join(src_obj_path, 'intrinsics.txt')): shutil.copy(os.path.join(src_obj_path, 'intrinsics.txt'), dst_obj_path)

        for idx in selected_indices:
            fname = f"{idx:06d}"
            src_img = os.path.join(src_obj_path, 'rgb', f"{fname}.png")
            if not os.path.exists(src_img): src_img = src_img.replace('.png', '.jpg')
            if os.path.exists(src_img): shutil.copy(src_img, os.path.join(dst_obj_path, 'rgb', os.path.basename(src_img)))
            
            for t in ['pose', 'intrinsics']:
                f_path = os.path.join(src_obj_path, t, f"{fname}.txt")
                if os.path.exists(f_path): shutil.copy(f_path, os.path.join(dst_obj_path, t))
        
        print(f"✅ Data ready at: {dst_obj_path}")
        return dst_obj_path, obj_id

def main():
    parser = argparse.ArgumentParser()
    default_config_path = os.path.join(CURRENT_FILE_DIR, 'config', 'default_config.yaml')
    parser.add_argument('--config', default=default_config_path)
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--mode', default='video', choices=['video', 'views']) 
    parser.add_argument('--size', type=int, default=None)
    parser.add_argument('--input_size', type=int, default=None)
    parser.add_argument('--num_frames', type=int, default=240) 
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--obj_id', type=str, default=None)
    parser.add_argument('--n_fine', type=int, default=None)
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    if args.n_fine is not None: config.model.n_fine = args.n_fine
    
    PixelNeRFInference(config, args.checkpoint).run(
        args.input_folder, args.output_dir, 
        render_size=args.size, input_size=args.input_size, 
        num_frames=args.num_frames, mode=args.mode, fps=args.fps, obj_id=args.obj_id
    )

if __name__ == '__main__':
    main()