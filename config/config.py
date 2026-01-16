# config/config.py
from dataclasses import dataclass, field
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
    learning_rate: float = 5e-4 
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
    override_lr: Optional[float] = None


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
    """Inference configuration (For video generation etc.)"""
    checkpoint_path: str = "./checkpoints/best.pth"
    output_dir: str = "./outputs"
    video_fps: int = 30
    chunk_size: int = 4096
    max_eval_views: int = 20


# [추가됨] 테스트 전용 설정 클래스
@dataclass
class TestConfig:
    """
    Configuration specifically for test.py evaluation 
    (High Performance Settings for A100/H100)
    """
    chunk_size: int = 32768
    max_eval_views: int = 250
    tto: TTOConfig = field(default_factory=TTOConfig)


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
                 test: Optional[TestConfig] = None, # [추가됨]
                 debug: Optional[DebugConfig] = None,  
                 device: str = "cuda", 
                 seed: int = 42):
        self.data = data
        self.model = model
        self.training = training
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.inference = inference or InferenceConfig()
        self.tto = tto or TTOConfig()
        self.test = test or TestConfig() # [추가됨] 기본값 초기화
        self.debug = debug or DebugConfig()  
        self.device = device  
        self.seed = seed
        self._validate_params()
    
    def _validate_params(self):
        """Validate configuration parameters"""
        if not (1e-6 <= self.training.learning_rate <= 1e-2):
            print(f"⚠️ Warning: learning_rate {self.training.learning_rate} is outside typical range")
        
        if self.optimizer.grad_clip < 0.1:
            print(f"⚠️ Warning: grad_clip {self.optimizer.grad_clip} is very small.")
                  
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
        
        if 'inference' in config_dict:
            config_dict['inference'] = InferenceConfig(**config_dict['inference'])
        
        # Global TTO (if used generally)
        if 'tto' in config_dict:
            if 'image_size' in config_dict['tto'] and isinstance(config_dict['tto']['image_size'], list):
                config_dict['tto']['image_size'] = tuple(config_dict['tto']['image_size'])
            config_dict['tto'] = TTOConfig(**config_dict['tto'])

        # [수정] Test Config 파싱 로직 추가
        if 'test' in config_dict:
            test_data = config_dict['test']
            
            # test 섹션 안에 있는 tto도 파싱해야 함
            if 'tto' in test_data:
                # 리스트 -> 튜플 변환 (image_size 등)
                if 'image_size' in test_data['tto'] and isinstance(test_data['tto']['image_size'], list):
                    test_data['tto']['image_size'] = tuple(test_data['tto']['image_size'])
                # 딕셔너리를 TTOConfig 객체로 변환
                test_data['tto'] = TTOConfig(**test_data['tto'])
            
            # 최종적으로 TestConfig 객체 생성
            config_dict['test'] = TestConfig(**test_data)

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
            'test': self.test.__dict__, 
            'debug': self.debug.__dict__  
        }
        
        # Nested dataclass handling for YAML dump (TestConfig inside tto)
        if isinstance(config_dict['test'].get('tto'), TTOConfig):
             config_dict['test']['tto'] = config_dict['test']['tto'].__dict__

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)