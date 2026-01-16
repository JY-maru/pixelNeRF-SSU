import os
import sys
import glob
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import gc
from PIL import Image
from torchvision import transforms

# -----------------------------------------------------------------------------
# [설정] 경고 메시지 차단
# -----------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum")

# -----------------------------------------------------------------------------
# 1. 필수 라이브러리 및 설정
# -----------------------------------------------------------------------------
try:
    import lpips
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    print("✕ 필수 라이브러리 설치 필요: pip install lpips scikit-image")
    sys.exit(1)

sys.path.append(os.getcwd())

from config.config import Config
from models.pixelnerf import PixelNeRF
from utils.geometry import CameraUtils
from data.shapenet import SHAPENET_ROTATION_MATRIX 

# -----------------------------------------------------------------------------
# 2. 데이터 로더
# -----------------------------------------------------------------------------
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
        intrinsic[0, 0] = focal; intrinsic[1, 1] = focal
        intrinsic[0, 2] = cx;    intrinsic[1, 2] = cy
        return intrinsic
    else:
        return torch.eye(3, dtype=torch.float32)

def load_test_data(folder_path, target_size, data_cfg):
    rgb_dir = os.path.join(folder_path, 'rgb')
    pose_dir = os.path.join(folder_path, 'pose')
    
    img_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")) + glob.glob(os.path.join(rgb_dir, "*.jpg")))
    pose_paths = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    
    if not img_paths: return None
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    
    valid_images, valid_poses, valid_intrinsics = [], [], []

    # shapenet.py의 _parse_instance_intrinsics 로직 (intrinsics.txt 로드)
    intrinsics_path = os.path.join(folder_path, 'intrinsics.txt')
    if os.path.exists(intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            lines = f.readlines()
        line1 = [float(x) for x in lines[0].strip().split()]
        fx, cx, cy = line1[0], line1[1], line1[2]
        original_size = int(lines[3].strip().split()[0]) if len(lines) >= 4 else 512
        
        intrinsic = torch.tensor([[fx, 0., cx], [0., fx, cy], [0., 0., 1.]], dtype=torch.float32)
        # Scale intrinsics to target size
        scale = target_size[0] / original_size
        intrinsic[:2, :] *= scale
    else:
        return None

    for i in range(len(img_paths)):
        # 1. 원본 Pose 로드 (필터링 계산용)
        raw_pose = parse_pose_file(pose_paths[i])
        cam_pos = raw_pose[:3, 3]
        dist = torch.norm(cam_pos)
        
        # 2. 고도(Elevation) 계산 (shapenet.py 153-157행 로직 그대로 복사)
        # Z축(cam_pos[2])을 높이로 사용
        dist_safe = max(dist.item(), 1e-6)
        sin_elevation = np.clip(cam_pos[2].item() / dist_safe, -1.0, 1.0)
        elevation_deg = np.arcsin(sin_elevation) * 180 / np.pi

        # 3. 설정된 범위(5도~50도) 내의 뷰만 선택
        if data_cfg.min_elevation <= elevation_deg <= data_cfg.max_elevation:
            # 이미지 로드 및 전처리
            img = Image.open(img_paths[i]).convert("RGB")
            valid_images.append(transform(img))
            
            # 모델 입력용으로 회전 변환 적용 (SHAPENET_ROTATION_MATRIX)
            pose_converted = torch.matmul(SHAPENET_ROTATION_MATRIX, raw_pose)
            valid_poses.append(pose_converted)
            valid_intrinsics.append(intrinsic)

    if not valid_images: return None

    return {
        'images': torch.stack(valid_images),       
        'poses': torch.stack(valid_poses),         
        'intrinsics': torch.stack(valid_intrinsics)
    }

# -----------------------------------------------------------------------------
# 3. 테스터 클래스
# -----------------------------------------------------------------------------
class Tester:
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # [검증] config.yaml에 test 섹션이 있는지 확인
        if not hasattr(self.config, 'test'):
            print("\n[Error] config.yaml 파일에 'test:' 섹션이 없습니다!")
            sys.exit(1)

        self.model = PixelNeRF(
            encoder_type=config.model.encoder_type, encoder_pretrained=False, 
            feature_dim=config.model.feature_dim, d_hidden=config.model.d_hidden,
            n_blocks=config.model.n_blocks, combine_type=config.model.combine_type,
            n_coarse=config.model.n_coarse, n_fine=config.model.n_fine,
            white_bkgd=config.model.white_bkgd, use_pe=config.model.use_pe,
            pe_freq_pos=config.model.pe_freq_pos, pe_freq_dir=config.model.pe_freq_dir
        ).to(self.device)
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.base_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device)

    def _load_checkpoint(self, path):
        if not os.path.exists(path): raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)

    def reset_model(self):
        self.model.load_state_dict(self.base_model_state)

    def run_tto(self, data, src_inds):
        test_tto_cfg = self.config.test.tto
        
        steps = test_tto_cfg.num_steps          
        lr = test_tto_cfg.learning_rate          
        batch_rays = test_tto_cfg.batch_rays     
        
        self.model.train()
        
        trainable_params = []
        for name, param in self.model.named_parameters():
            if 'encoder' in name or 'resnet' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                trainable_params.append(param)
        
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        
        images_all = data['images']
        poses_all = data['poses']
        intrinsics_all = data['intrinsics']
        _, _, H, W = images_all.shape
        
        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), -1).reshape(-1, 2)
        
        src_imgs = images_all[src_inds].to(self.device)
        src_poses = poses_all[src_inds].to(self.device)
        src_intrin = intrinsics_all[src_inds].to(self.device)
        norm_src_imgs = self.normalize(src_imgs).unsqueeze(0)

        final_loss = 0.0

        # [로그] TTO 시작
        print(f" [TTO:{steps}]", end="", flush=True)

        for step in range(steps):
            target_local_idx = np.random.randint(0, len(src_inds))
            target_global_idx = src_inds[target_local_idx]

            select_inds = torch.randperm(H * W)[:batch_rays]
            select_coords = coords[select_inds]
            
            target_rgb = images_all[target_global_idx, :, select_coords[:, 0], select_coords[:, 1]].T.to(self.device)
            target_pose = poses_all[target_global_idx].to(self.device)
            target_intrin = intrinsics_all[target_global_idx].to(self.device)

            rays_o, rays_d = CameraUtils.get_rays(H, W, target_intrin, target_pose)
            rays_o = rays_o.reshape(-1, 3)[select_inds].unsqueeze(0)
            rays_d = rays_d.reshape(-1, 3)[select_inds].unsqueeze(0)
            
            out = self.model(
                norm_src_imgs,
                src_intrin.unsqueeze(0), 
                src_poses.unsqueeze(0),
                rays_o, rays_d, 
                z_near=self.config.data.z_near, z_far=self.config.data.z_far
            )
            
            loss = F.mse_loss(out['fine']['rgb_map'].squeeze(0), target_rgb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step == steps - 1: final_loss = loss.item()
            # 10 step마다 점 찍기
            if step % 10 == 0: print("⫻", end="", flush=True)
            
            del rays_o, rays_d, target_rgb, out, loss
        
        del optimizer, src_imgs, src_poses, src_intrin, norm_src_imgs
        torch.cuda.empty_cache()
        final_psnr = -10. * np.log10(final_loss + 1e-10)
        return final_loss, final_psnr

    def evaluate(self, data, src_inds):
        self.model.eval()
        psnr_list, ssim_list, lpips_list = [], [], []
        
        images_all = data['images']
        poses_all = data['poses']
        intrinsics_all = data['intrinsics']
        N, _, H, W = images_all.shape
        
        test_cfg = self.config.test
        chunk_size = test_cfg.chunk_size          
        max_eval_views = test_cfg.max_eval_views  
        
        src_imgs = images_all[src_inds].to(self.device)
        src_poses = poses_all[src_inds].to(self.device)
        src_intrin = intrinsics_all[src_inds].to(self.device)
        
        with torch.no_grad():
            net_input = self.normalize(src_imgs).unsqueeze(0)
            
            all_target_inds = [i for i in range(N) if i not in src_inds]
            
            if len(all_target_inds) > max_eval_views:
                chosen_inds = np.random.choice(all_target_inds, max_eval_views, replace=False)
                target_inds = sorted(chosen_inds)
            else:
                target_inds = all_target_inds

            # [로그] 평가 시작
            print(f" [Eval:{len(target_inds)}]", end="", flush=True)

            for idx, i in enumerate(target_inds):
                gt_tensor = images_all[i].to(self.device)
                target_pose = poses_all[i].to(self.device)
                target_intrin = intrinsics_all[i].to(self.device)
                
                rays_o, rays_d = CameraUtils.get_rays(H, W, target_intrin, target_pose)
                rays_o = rays_o.reshape(-1, 3).unsqueeze(0)
                rays_d = rays_d.reshape(-1, 3).unsqueeze(0)
                
                rgb_chunks = []
                for k in range(0, rays_o.shape[1], chunk_size):
                    chunk_o = rays_o[:, k:k+chunk_size]
                    chunk_d = rays_d[:, k:k+chunk_size]
                    
                    out = self.model(
                        net_input, 
                        src_intrin.unsqueeze(0), 
                        src_poses.unsqueeze(0),
                        chunk_o, chunk_d,
                        self.config.data.z_near, self.config.data.z_far
                    )
                    rgb_chunks.append(out['fine']['rgb_map'].cpu())
                
                pred_rgb = torch.cat(rgb_chunks, dim=1).reshape(H, W, 3)
                pred_img = torch.clamp(pred_rgb, 0, 1)
                pred_np = pred_img.numpy()
                gt_np = gt_tensor.permute(1, 2, 0).cpu().numpy()

                mse = np.mean((gt_np - pred_np) ** 2)
                psnr_list.append(-10.0 * np.log10(mse + 1e-10))
                ssim_list.append(ssim_func(gt_np, pred_np, data_range=1.0, channel_axis=-1, win_size=11))
                
                p_th = pred_img.permute(2, 0, 1).unsqueeze(0).to(self.device) * 2 - 1
                g_th = gt_tensor.unsqueeze(0) * 2 - 1
                lpips_list.append(self.loss_fn_vgg(p_th, g_th).item())
                
                # 점 찍기
                if idx % 5 == 0: print("⫻", end="", flush=True)
                
                del gt_tensor, target_pose, target_intrin, rays_o, rays_d, p_th, g_th

            del src_imgs, src_poses, src_intrin, net_input

        return {
            'PSNR': np.mean(psnr_list),
            'SSIM': np.mean(ssim_list),
            'LPIPS': np.mean(lpips_list),
            'Count': len(target_inds)
        }

# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--test_data_dir', type=str, default='/content/pixNeRF_shapeNet_v2_data/cars_test')
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    
    # [Config 확인]
    try:
        test_cfg = config.test      # 테스트 전용 Config
        data_cfg = config.data
    except AttributeError:
        print("\n[Critical Error] config.yaml에 'test:' 섹션이 누락되었습니다.")
        return

    print(f"▢ Pixel-NeRF Test (Clean Logs)")
    print(f"  - Source Views: {data_cfg.num_source_views}")
    # test_cfg.tto.enabled 확인
    print(f"  - TTO Enabled: {test_cfg.tto.enabled}") 
    print(f"  - TTO Batch Rays: {test_cfg.tto.batch_rays}") # 실제 사용하는 값 출력
    
    tester = Tester(config, config.inference.checkpoint_path)
    instance_folders = sorted(glob.glob(os.path.join(args.test_data_dir, "*")))
    instance_folders = [f for f in instance_folders if os.path.isdir(f)]
    
    os.makedirs(config.inference.output_dir, exist_ok=True)
    metrics_file = os.path.join(config.inference.output_dir, "test_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("Instance_ID, PSNR, SSIM, LPIPS\n")

    total_metrics = {'PSNR': [], 'SSIM': [], 'LPIPS': []}
    
    print(f"\n▢ Start Testing {len(instance_folders)} objects...")
    
    for idx, folder in enumerate(instance_folders):
        instance_name = os.path.basename(folder)
        
        print(f"[{idx+1:03d}/{len(instance_folders)}]", end="", flush=True)
        
        try:
            tester.reset_model()
            data = load_test_data(folder, target_size=config.data.image_size, data_cfg=config.data)

            if data is None: 
                print(" ->[✕] No Data")
                continue
            
            N = data['images'].shape[0]
            num_src = config.data.num_source_views
            
            if N > num_src:
                src_inds = np.random.choice(N, num_src, replace=False)
            else:
                src_inds = np.arange(N)
            
            # TTO (config.test.tto.enabled 확인)
            tto_status = "Skip"
            if config.test.tto.enabled:  # 또는 test_cfg.tto.enabled (객체 매핑이 정확할 경우)
                print(f" [TTO:{config.test.tto.num_steps}]", end=" ", flush=True)
                loss, psnr = tester.run_tto(data, src_inds)
                tto_status = "Done"
            else:
                print(f" [TTO:Off]", end=" ", flush=True)

            # Evaluate
            metrics = tester.evaluate(data, src_inds)
            
            print(f" -> PSNR: {metrics['PSNR']:.2f}dB")

            with open(metrics_file, "a") as f:
                f.write(f"{instance_name}, {metrics['PSNR']:.4f}, {metrics['SSIM']:.4f}, {metrics['LPIPS']:.4f}\n")
            
            for k, v in metrics.items():
                if k in total_metrics: total_metrics[k].append(v)

        except Exception as e:
            if "out of memory" in str(e):
                print(" -> [✕] OOM")
                torch.cuda.empty_cache()
            else:
                print(f" -> [✕] Err: {e}")

        del data
        gc.collect()

    print("\n" + "="*30)
    print(f"Avg PSNR : {np.mean(total_metrics['PSNR']):.4f}")
    print(f"Avg SSIM : {np.mean(total_metrics['SSIM']):.4f}")
    print(f"Avg LPIPS: {np.mean(total_metrics['LPIPS']):.4f}")

if __name__ == "__main__":
    main()