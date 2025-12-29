# config/config.py
from dataclasses import dataclass
from typing import Optional, Tuple
import yaml


@dataclass
class DataConfig:
    """Data configuration"""
    data_root: str
    num_source_views: int = 1
    image_size: Tuple[int, int] = (128, 128)
    z_near: float = 0.15
    z_far: float = 2.7
    use_imagenet_normalize: bool = True  

    # Dataset Filtering
    enable_filtering: bool = True
    min_view_angle: float = 10.0       
    max_view_angle: float = 120.0
    min_camera_distance: float = 0.1   
    max_camera_distance: float = 3.0 
    min_elevation: float = 5.0              # 바닥면 제거 (도 단위)
    max_elevation: float = 85.0  
    #Instance 수 제한 및 Repeat Factor
    num_train_instances: Optional[int] = None   # None = 전체 사용
    num_val_instances: Optional[int] = None
    num_test_instances: Optional[int] = None
    repeat_factor: int = 1     

@dataclass
class TTOConfig:
    """Test-Time Optimization configuration"""
    enabled: bool = False
    num_steps: int = 1000
    learning_rate: float = 1e-5
    batch_rays: int = 512
    image_size: Tuple[int, int] = (256, 256) # TTO 시 사용할 해상도
    
@dataclass
class ModelConfig:
    """Model configuration"""
    encoder_type: str = 'resnet34'
    encoder_pretrained: bool = True
    feature_dim: int = 512
    d_hidden: int = 512
    n_blocks: int = 5
    combine_type: str = 'average'
    n_coarse: int = 64
    n_fine: int = 128
    n_depth: int = 16          
    white_bkgd: bool = True
    noise_std: float = 1.0
    use_pe: bool = True
    pe_freq_pos: int = 6
    pe_freq_dir: int = 2
    density_lambda: float = 0.01


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 4
    num_epochs: int = 200
    learning_rate: float = 5e-4 # 기본값 (보통 yaml에서 override)
    weight_decay: float = 0.0
    num_rays_per_batch: int = 1024
    chunk_size: int = 4096
    fg_sampling_ratio: float = 0.8  
    fg_threshold: float = 0.15    
    
    step_debug: int = 100
    log_every_n_steps: int = 100
    log_every_n_epochs: int = 1
    val_every_n_steps: int = 0
    val_every_n_epochs: int = 1
    save_every_n_steps: int = 0
    save_every_n_epochs: int = 10
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    type: str = 'adam'
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0


@dataclass
class SchedulerConfig:
    """Scheduler configuration"""
    type: str = 'cosine'
   # Step LR
    step_size: int = 50
    gamma: float = 0.5
    
    # Cosine Annealing LR
    T_max: int = 200          
    eta_min: float = 1e-6    
    
    # Warmup
    warmup_steps: int = 1000


@dataclass
class InferenceConfig:
    """Inference configuration"""
    checkpoint_path: str = "./checkpoints/best.pth"
    output_dir: str = "./outputs"
    video_fps: int = 30
    chunk_size: int = 4096


@dataclass
class DebugConfig:
    """Debug configuration"""
    enabled: bool = False
    log_first_batch: bool = True
    log_every_n_steps: int = 100
    check_nan_inf: bool = True
    save_debug_images: bool = False


class Config:
    """Main configuration class"""
    
    def __init__(self, 
                 data: DataConfig, 
                 model: ModelConfig,
                 training: TrainingConfig, 
                 optimizer: OptimizerConfig,
                 scheduler: SchedulerConfig, 
                 inference: Optional[InferenceConfig] = None,
                 tto: Optional[TTOConfig] = None,
                 debug: Optional[DebugConfig] = None,  
                 device: str = "cuda", 
                 seed: int = 42):
        self.data = data
        self.model = model
        self.training = training
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.inference = inference or InferenceConfig()
        self.debug = debug or DebugConfig()  
        self.device = device  
        self.seed = seed
        self._validate_params()
        self.tto = tto or TTOConfig()
    
    def _validate_params(self):
        """Validate configuration parameters"""
        # Learning rate 범위 확인
        if not (1e-6 <= self.training.learning_rate <= 1e-2):
            print(f"⚠️ Warning: learning_rate {self.training.learning_rate} "
                  f"is outside typical range [1e-6, 1e-2]")
        
        # Grad clip 확인
        if self.optimizer.grad_clip < 0.1:
            print(f"⚠️ Warning: grad_clip {self.optimizer.grad_clip} "
                  f"is very small. Gradients might explode!")
        
        # Ray sampling 확인
        if self.training.num_rays_per_batch > 2048:
            print(f"⚠️ Warning: num_rays_per_batch {self.training.num_rays_per_batch} "
                  f"is very large. Might cause OOM!")
                  
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dicts to dataclass instances
        config_dict['data'] = DataConfig(**config_dict['data'])
        config_dict['model'] = ModelConfig(**config_dict['model'])
        config_dict['training'] = TrainingConfig(**config_dict['training'])
        config_dict['optimizer'] = OptimizerConfig(**config_dict['optimizer'])
        config_dict['scheduler'] = SchedulerConfig(**config_dict['scheduler'])
        
        # Inference config is optional
        if 'inference' in config_dict:
            config_dict['inference'] = InferenceConfig(**config_dict['inference'])
        
        if 'tto' in config_dict:
            # image_size가 리스트로 들어올 경우 튜플로 변환
            if 'image_size' in config_dict['tto'] and isinstance(config_dict['tto']['image_size'], list):
                config_dict['tto']['image_size'] = tuple(config_dict['tto']['image_size'])
            config_dict['tto'] = TTOConfig(**config_dict['tto'])

        if 'debug' in config_dict:
            config_dict['debug'] = DebugConfig(**config_dict['debug'])
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'optimizer': self.optimizer.__dict__,
            'scheduler': self.scheduler.__dict__,
            'inference': self.inference.__dict__,
            'tto': self.tto.__dict__,
            'debug': self.debug.__dict__  
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)