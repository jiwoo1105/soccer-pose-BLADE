# Soccer Motion 3D Analysis                                                                                               
                                                                                                                            
  Single-image 3D human mesh recovery for soccer player motion analysis using NVIDIA BLADE.                                 
                                                                                                                            
  ## Overview                                                                                                               
                                                                                                                            
  This project attempts to implement 3D body mesh estimation for soccer players using [BLADE (Body Learning through Accurate
   Depth Estimation)](https://github.com/NVlabs/BLADE), a state-of-the-art human mesh recovery model from NVIDIA Research.  
                                                                                                                            
  ### What is BLADE?                                                                                                        
                                                                                                                            
  BLADE is a single-view human mesh recovery method that leverages accurate depth estimation to reconstruct 3D human body   
  meshes from monocular images. It was presented at CVPR 2025.                                                              
                                                                                                                            
  **Paper:** [BLADE: Single-view Body Mesh Learning through Accurate Depth Estimation](https://arxiv.org/abs/2410.08754)    
                                                                                                                            
  **Official Repository:** [https://github.com/NVlabs/BLADE](https://github.com/NVlabs/BLADE)                               
                                                                                                                            
  ---                                                                                                                       
                                                                                                                            
  ## Project Status                                                                                                         
                                                                                                                            
  > **Note:** This project encountered significant technical challenges during setup. See [Challenges &                     
  Discussion](#challenges--discussion) for details.                                                                         
                                                                                                                            
  | Component | Status |                                                                                                    
  |-----------|--------|                                                                                                    
  | Environment Setup | Partial |                                                                                           
  | Model Loading | Partial |                                                                                               
  | Inference | Not Working |                                                                                               
  | Soccer Integration | Not Started |                                                                                      
                                                                                                                            
  ---                                                                                                                       
                                                                                                                            
  ## System Requirements                                                                                                    
                                                                                                                            
  ### Minimum (Not Sufficient for BLADE)                                                                                    
  - GPU: 8GB VRAM                                                                                                           
  - RAM: 16GB                                                                                                               
  - Storage: 50GB                                                                                                           
                                                                                                                            
  ### Recommended for BLADE                                                                                                 
  - GPU: 24GB+ VRAM (RTX 3090, A100, etc.)                                                                                  
  - RAM: 64GB                                                                                                               
  - Storage: 100GB                                                                                                          
                                                                                                                            
  ---                                                                                                                       
                                                                                                                            
  ## Installation Attempts                                                                                                  
                                                                                                                            
  ### Environment Setup                                                                                                     
                                                                                                                            
  ```bash                                                                                                                   
  conda create -n blade_env python=3.9                                                                                      
  conda activate blade_env                                                                                                  
                                                                                                                            
  # PyTorch with CUDA                                                                                                       
  pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118               
                                                                                                                            
  # Core dependencies                                                                                                       
  pip install mmcv==2.1.0 mmengine==0.10.7                                                                                  
  pip install numpy==1.26.4  # Must be < 2.0                                                                                
                                                                                                                            
  Required Model Files                                                                                                      
  ┌───────────────────┬────────┬───────────────────────────────────────────────────────┐                                    
  │       File        │  Size  │                        Source                         │                                    
  ├───────────────────┼────────┼───────────────────────────────────────────────────────┤                                    
  │ BLADE checkpoint  │ 2.4GB  │ https://github.com/NVlabs/BLADE/releases              │                                    
  ├───────────────────┼────────┼───────────────────────────────────────────────────────┤                                    
  │ Sapiens Pose 1B   │ 4.4GB  │ https://huggingface.co/facebook/sapiens-pose-1b       │                                    
  ├───────────────────┼────────┼───────────────────────────────────────────────────────┤                                    
  │ Sapiens Pose 0.3B │ 1.3GB  │ https://huggingface.co/facebook/sapiens-pose-0.3b     │                                    
  ├───────────────────┼────────┼───────────────────────────────────────────────────────┤                                    
  │ SMPL Models       │ ~750MB │ https://smpl.is.tue.mpg.de/ (Registration Required)   │                                    
  ├───────────────────┼────────┼───────────────────────────────────────────────────────┤                                    
  │ SMPL-X Models     │ ~1.6GB │ https://smpl-x.is.tue.mpg.de/ (Registration Required) │                                    
  └───────────────────┴────────┴───────────────────────────────────────────────────────┘                                    
  ---                                                                                                                       
  Challenges & Discussion                                                                                                   
                                                                                                                            
  1. Critical: mmcv 1.x vs 2.x API Incompatibility                                                                          
                                                                                                                            
  BLADE depends on libraries with conflicting requirements:                                                                 
                                                                                                                            
  mmhuman3d ──────► requires mmcv 1.x API                                                                                   
       │                                                                                                                    
       ▼                                                                                                                    
    CONFLICT                                                                                                                
       ▲                                                                                                                    
       │                                                                                                                    
  sapiens (mmdet, mmpose) ──────► requires mmcv 2.x (mmengine) API                                                          
                                                                                                                            
  Attempted Solution: Manual patching of mmcv 2.x to provide 1.x compatibility layer                                        
                                                                                                                            
  Files Modified:                                                                                                           
  - mmcv/__init__.py                                                                                                        
  - mmcv/utils/__init__.py                                                                                                  
  - mmcv/cnn/__init__.py                                                                                                    
  - mmcv/cnn/bricks/__init__.py                                                                                             
                                                                                                                            
  Limitation: System package modifications are lost on updates and cannot be shared.                                        
                                                                                                                            
  ---                                                                                                                       
  2. NumPy 2.x Compatibility                                                                                                
                                                                                                                            
  NumPy 2.0 removed deprecated aliases (np.int, np.float, etc.)                                                             
                                                                                                                            
  # Error                                                                                                                   
  AttributeError: module 'numpy' has no attribute 'int'                                                                     
                                                                                                                            
  # Fix: Change np.int to np.int64                                                                                          
  array.astype(np.int64)                                                                                                    
                                                                                                                            
  Affected Files:                                                                                                           
  - blade/models/architectures/blade.py                                                                                     
  - blade/datasets/pipelines/transforms.py                                                                                  
  - Multiple transform files in mmhuman3d and aios_repo                                                                     
                                                                                                                            
  ---                                                                                                                       
  3. Dependency Hell                                                                                                        
                                                                                                                            
  Package Conflicts:                                                                                                        
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                           
  mmcv 2.1.0      ←→  mmcv-full 1.5.3 (both installed)                                                                      
  numpy 2.x       ←→  numpy < 2 (different packages require different versions)                                             
  opencv 4.13     →   requires numpy >= 2                                                                                   
  mediapipe 0.10+ →   removed solutions API                                                                                 
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                                                           
                                                                                                                            
  ---                                                                                                                       
  4. GPU Memory Constraints (Primary Blocker)                                                                               
                                                                                                                            
  Test System: NVIDIA RTX 2070 (8GB VRAM)                                                                                   
                                                                                                                            
  BLADE Memory Requirements:                                                                                                
  ┌───────────────────────┬──────────┐                                                                                      
  │       Component       │   VRAM   │                                                                                      
  ├───────────────────────┼──────────┤                                                                                      
  │ BLADE Checkpoint      │ ~2.5GB   │                                                                                      
  ├───────────────────────┼──────────┤                                                                                      
  │ Sapiens 1B Pose Model │ ~4.4GB   │                                                                                      
  ├───────────────────────┼──────────┤                                                                                      
  │ SMPL-X Models         │ ~1.5GB   │                                                                                      
  ├───────────────────────┼──────────┤                                                                                      
  │ RTMDet Detector       │ ~100MB   │                                                                                      
  ├───────────────────────┼──────────┤                                                                                      
  │ Total Required        │ 10-15GB+ │                                                                                      
  └───────────────────────┴──────────┘                                                                                      
  Result: System freeze during model loading, requiring forced restart.                                                     
                                                                                                                            
  Mitigation Attempts:                                                                                                      
  - Switched to smaller model: sapiens_1b → sapiens_0.3b (4.4GB → 1.3GB)                                                    
  - CPU keypoint detection: kpt_device='cpu'                                                                                
  - Minimum batch size: samples_per_gpu=1                                                                                   
  - Reduced workers: workers_per_gpu=0                                                                                      
                                                                                                                            
  ---                                                                                                                       
  5. Missing/Placeholder Files                                                                                              
                                                                                                                            
  Git LFS files were not properly downloaded:                                                                               
                                                                                                                            
  demo_images/*.jpg  →  ASCII text (Git LFS pointers), not actual images                                                    
                                                                                                                            
  Required Manual Downloads:                                                                                                
  - SMPLX_to_J14.pkl from OpenMMLab                                                                                         
  - J_regressor_extra.npy (generated from SMPL data)                                                                        
  - J_regressor_h36m.npy (generated from SMPL data)                                                                         
                                                                                                                            
  ---                                                                                                                       
  6. Config Parsing Issues                                                                                                  
                                                                                                                            
  mmengine Config Problems:                                                                                                 
  - __file__ variable undefined during config parsing                                                                       
  - _file2dict returns 3 values (mmengine) vs 2 values (mmcv 1.x)                                                           
  - Missing .mim/model-index.yml for mmdet                                                                                  
                                                                                                                            
  ---                                                                                                                       
  Alternative Models                                                                                                        
                                                                                                                            
  For systems with limited GPU memory, consider these lighter alternatives:                                                 
  ┌───────────────────────────────────────────┬──────┬──────────────┬──────────────┬────────────┐                           
  │                   Model                   │ VRAM │ Installation │ Multi-Person │ Hands/Face │                           
  ├───────────────────────────────────────────┼──────┼──────────────┼──────────────┼────────────┤                           
  │ https://github.com/shubham-goel/4D-Humans │ ~4GB │ Easy         │ Yes          │ No         │                           
  ├───────────────────────────────────────────┼──────┼──────────────┼──────────────┼────────────┤                           
  │ https://github.com/Jeff-sjtu/HybrIK       │ ~3GB │ Medium       │ Yes          │ No         │                           
  ├───────────────────────────────────────────┼──────┼──────────────┼──────────────┼────────────┤                           
  │ https://github.com/Arthur151/ROMP         │ ~2GB │ Very Easy    │ Yes          │ No         │                           
  ├───────────────────────────────────────────┼──────┼──────────────┼──────────────┼────────────┤                           
  │ https://github.com/caizhongang/SMPLer-X   │ ~6GB │ Medium       │ Yes          │ Yes        │                           
  └───────────────────────────────────────────┴──────┴──────────────┴──────────────┴────────────┘                           
  Recommendation for Soccer Analysis: 4D-Humans or ROMP                                                                     
                                                                                                                            
  ---                                                                                                                       
  Project Structure                                                                                                         
                                                                                                                            
  .                                                                                                                         
  ├── blade/                  # BLADE model code                                                                            
  │   ├── configs/           # Configuration files                                                                          
  │   ├── models/            # Model architectures                                                                          
  │   └── datasets/          # Data loading                                                                                 
  ├── body_models/           # SMPL/SMPL-X models                                                                           
  │   ├── smpl/                                                                                                             
  │   └── smplx/                                                                                                            
  ├── pretrained/            # Model checkpoints                                                                            
  ├── sapiens/               # Sapiens pose estimation                                                                      
  ├── mmhuman3d/             # Human mesh recovery library                                                                  
  ├── api/                   # Inference API                                                                                
  └── results/               # Output directory                                                                             
                                                                                                                            
  ---                                                                                                                       
  References                                                                                                                
                                                                                                                            
  BLADE                                                                                                                     
                                                                                                                            
  @inproceedings{blade2025,                                                                                                 
    title={BLADE: Single-view Body Mesh Learning through Accurate Depth Estimation},                                        
    author={NVIDIA Research},                                                                                               
    booktitle={CVPR},                                                                                                       
    year={2025}                                                                                                             
  }                                                                                                                         
                                                                                                                            
  Related Works                                                                                                             
                                                                                                                            
  - https://smpl.is.tue.mpg.de/                                                                                             
  - https://smpl-x.is.tue.mpg.de/                                                                                           
  - https://github.com/facebookresearch/sapiens                                                                             
  - https://github.com/open-mmlab/mmhuman3d                                                                                 
                                                                                                                            
  ---                                                                                                                       
  Lessons Learned                                                                                                           
                                                                                                                            
  1. Check Hardware Requirements First - Large models need significant GPU memory                                           
  2. OpenMMLab Version Compatibility - mmcv versions are not backward compatible                                            
  3. Isolate Environments - Use separate conda environments per project                                                     
  4. Document Everything - Record all modifications for reproducibility                                                     
                                                                                                                            
  ---                                                                                                                       
  License                                                                                                                   
                                                                                                                            
  This project references code from:                                                                                        
  - BLADE: NVIDIA Proprietary License                                                                                       
  - SMPL/SMPL-X: Academic Use Only                                                                                          
  - Sapiens: Meta License                                                                                                   
                                                                                                                            
  ---                                                                                                                       
  Acknowledgments                                                                                                           
                                                                                                                            
  - https://github.com/NVlabs for BLADE                                                                                     
  - https://github.com/facebookresearch for Sapiens                                                                         
  - https://github.com/open-mmlab for mmhuman3d                                                                             
                                                                                                                            
  ---                                                                                                                       
  Contact                                                                                                                   
                                                                                                                            
  For questions about this setup attempt, please open an issue.                                                             
                                                                                                                            
  For BLADE-specific questions, refer to the https://github.com/NVlabs/BLADE.    
