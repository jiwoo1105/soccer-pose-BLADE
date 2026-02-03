# core/blade_wrapper.py
"""
=============================================================================
BLADE API 래퍼 - SMPL-X 54개 관절 3D 포즈 추출
=============================================================================

이 모듈은 BLADE (Body Landmarks Anatomical Dense Estimation)를 사용하여
비디오에서 정확한 3D 인체 포즈를 추출합니다.

BLADE의 장점:
1. SMPL-X 기반 54개 관절 추출 (MediaPipe 33개보다 상세)
2. 미터 단위의 정확한 3D 좌표
3. 골반 중심 좌표계 (카메라 뷰 불변)
4. 시간적 일관성 보장

SMPL-X 핵심 관절 매핑:
    0: pelvis (골반) - 기준점
    1: left_hip (왼쪽 엉덩이)
    2: right_hip (오른쪽 엉덩이)
    3: spine1 (척추 하단)
    4: left_knee (왼쪽 무릎)
    5: right_knee (오른쪽 무릎)
    6: spine2 (척추 중단)
    7: left_ankle (왼쪽 발목)
    8: right_ankle (오른쪽 발목)
    9: spine3 (척추 상단) - 상체 각도 계산용
    10: left_foot (왼발)
    11: right_foot (오른발)
    12: neck (목)
    13: left_collar (왼쪽 쇄골)
    14: right_collar (오른쪽 쇄골)
    15: head (머리)
    16: left_shoulder (왼쪽 어깨)
    17: right_shoulder (오른쪽 어깨)
    18: left_elbow (왼쪽 팔꿈치)
    19: right_elbow (오른쪽 팔꿈치)
    20: left_wrist (왼쪽 손목)
    21: right_wrist (오른쪽 손목)
    22: jaw (턱)
    23: left_eye (왼쪽 눈)
    24: right_eye (오른쪽 눈)
    25-54: 손가락 관절들
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import smplx
from tqdm import tqdm

# SMPL-X 관절 인덱스 상수
SMPLX_JOINT_INDICES = {
    'pelvis': 0,
    'left_hip': 1,
    'right_hip': 2,
    'spine1': 3,
    'left_knee': 4,
    'right_knee': 5,
    'spine2': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'spine3': 9,
    'left_foot': 10,
    'right_foot': 11,
    'neck': 12,
    'left_collar': 13,
    'right_collar': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
    'jaw': 22,
    'left_eye': 23,
    'right_eye': 24,
}


@dataclass
class PoseFrame3D:
    """
    BLADE에서 추출한 단일 프레임의 3D 포즈 데이터

    Attributes:
        frame_number: 프레임 번호 (0부터 시작)
        timestamp: 비디오 내 시간 (초)
        joints_3d: (54, 3) SMPL-X 관절의 3D 좌표 (미터 단위, 골반 중심)
                   - joints_3d[i][0]: x 좌표 (좌우, 오른쪽이 +)
                   - joints_3d[i][1]: y 좌표 (상하, 위쪽이 +)
                   - joints_3d[i][2]: z 좌표 (앞뒤, 앞이 +)
        joints_2d: (54, 2) 이미지 좌표 (0-1 정규화)
        confidence: (54,) 각 관절의 신뢰도 점수 [0-1]
        body_pose: SMPL-X body pose 파라미터 (옵션)
        global_orient: 전역 회전 (옵션)
        ball_position: (x, y) 공의 2D 픽셀 좌표, 탐지 실패 시 None
        ball_position_3d: (x, y, z) 공의 3D 좌표 (추정값), None 가능
        ball_bbox: (x1, y1, x2, y2) 공의 바운딩 박스
        frame_width: 프레임 너비 (픽셀)
        frame_height: 프레임 높이 (픽셀)

    사용 예시:
        >>> pose = pose_frames[0]
        >>> pelvis = pose.joints_3d[0]  # 골반 위치 (기준점)
        >>> left_knee = pose.joints_3d[4]  # 왼쪽 무릎
        >>> spine3 = pose.joints_3d[9]  # 상체 상단
    """
    frame_number: int
    timestamp: float
    joints_3d: np.ndarray  # (54, 3) - real world coordinates in meters
    joints_2d: np.ndarray  # (54, 2) - normalized image coordinates
    confidence: np.ndarray  # (54,) - confidence scores
    body_pose: Optional[np.ndarray] = None  # SMPL-X pose parameters
    global_orient: Optional[np.ndarray] = None  # global rotation
    ball_position: Optional[Tuple[float, float]] = None
    ball_position_3d: Optional[Tuple[float, float, float]] = None
    ball_bbox: Optional[Tuple[int, int, int, int]] = None
    frame_width: int = 0
    frame_height: int = 0

    def get_joint(self, joint_name: str) -> np.ndarray:
        """관절 이름으로 3D 좌표 조회"""
        if joint_name not in SMPLX_JOINT_INDICES:
            raise ValueError(f"Unknown joint name: {joint_name}")
        return self.joints_3d[SMPLX_JOINT_INDICES[joint_name]]

    def get_joint_confidence(self, joint_name: str) -> float:
        """관절 이름으로 신뢰도 조회"""
        if joint_name not in SMPLX_JOINT_INDICES:
            raise ValueError(f"Unknown joint name: {joint_name}")
        return self.confidence[SMPLX_JOINT_INDICES[joint_name]]


class BladeWrapper:
    """
    BLADE API 래퍼 - SMPL-X 54개 관절 추출

    BLADE (Body Landmarks Anatomical Dense Estimation)는
    단일 이미지에서 정확한 3D 인체 포즈를 추출하는 모델입니다.

    주요 기능:
    1. 비디오에서 프레임별 3D 포즈 추출
    2. SMPL-X 54개 관절 좌표 (미터 단위)
    3. 골반 중심 좌표계 출력 (카메라 뷰 불변)
    4. 시간적 스무딩 옵션

    사용 예시:
        >>> wrapper = BladeWrapper()
        >>> pose_frames = wrapper.extract_from_video("soccer.mp4")
        >>> print(f"추출된 프레임: {len(pose_frames)}")
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 smplx_path: Optional[str] = None,
                 device: str = 'cuda',
                 temporal_smoothing: bool = True,
                 smoothing_window: int = 5,
                 batch_size: int = 1,
                 workers_per_gpu: int = 2,
                 temp_output_dir: Optional[str] = None,
                 blade_repo_path: Optional[str] = None,
                 cfg_path: Optional[str] = None,
                 max_frames: Optional[int] = None,
                 resize_width: Optional[int] = None,
                 resize_height: Optional[int] = None):
        """
        BladeWrapper 초기화

        Args:
            model_path: BLADE 모델 가중치 경로
            smplx_path: SMPL-X 모델 경로
            device: 연산 장치 ('cuda' 또는 'cpu')
            temporal_smoothing: 시간적 스무딩 적용 여부
            smoothing_window: 스무딩 윈도우 크기 (홀수)
            batch_size: BLADE 처리 배치 크기 (1 권장)
            workers_per_gpu: 데이터 로더 워커 수
            temp_output_dir: BLADE 중간 결과 저장 경로
            blade_repo_path: BLADE 저장소 루트 경로
            cfg_path: BLADE 설정 파일 경로
            max_frames: 최대 처리 프레임 수 (스모크 테스트용, None이면 전체)
            resize_width: 입력 리사이즈 너비 (None이면 원본 유지)
            resize_height: 입력 리사이즈 높이 (None이면 원본 유지)
        """
        self.device = device
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_window = smoothing_window
        self.batch_size = max(1, int(batch_size))
        self.workers_per_gpu = max(0, int(workers_per_gpu))
        self.max_frames = max_frames if (max_frames is None or max_frames > 0) else None
        self.resize_width = resize_width if (resize_width is None or resize_width > 0) else None
        self.resize_height = resize_height if (resize_height is None or resize_height > 0) else None

        if str(self.device).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 장치를 요청했지만 사용할 수 없습니다. "
                "BLADE는 CPU에서 동작하지 않으니 GPU 환경에서 실행해주세요."
            )

        # 경로 설정
        self.project_root = Path(__file__).resolve().parents[1]
        self.blade_repo = Path(blade_repo_path).resolve() if blade_repo_path else (self.project_root / 'blade')
        self.cfg_path = cfg_path

        # BLADE 모델 초기화
        self.model_path = model_path
        self.smplx_path = smplx_path
        self.smplx_model = None
        self._blade_api_cls = None
        self._project_points_fn = None

        # 임시 출력 경로
        self.temp_output_root = Path(temp_output_dir).resolve() if temp_output_dir else (self.project_root / 'output' / 'blade_results')
        self.temp_output_root.mkdir(parents=True, exist_ok=True)

        self._initialize_blade(model_path, smplx_path)

    def _initialize_blade(self, model_path: Optional[str], smplx_path: Optional[str]):
        """
        BLADE 모델 초기화

        Note:
            이 프로젝트는 MediaPipe 폴백 없이 BLADE만 사용합니다.
        """
        if not self.blade_repo.exists():
            raise FileNotFoundError(f"BLADE 저장소 경로를 찾을 수 없습니다: {self.blade_repo}")

        # 모델 경로 해석
        if model_path is None:
            default_model = self.blade_repo / 'pretrained' / 'epoch_2.pth'
            if default_model.exists():
                model_path = str(default_model)
        model_path = self._resolve_path(model_path) if model_path else None
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"BLADE 모델 가중치가 없습니다: {model_path}")
        self.model_path = model_path

        # SMPL-X 경로 해석
        if smplx_path is None:
            default_smplx = self.blade_repo / 'body_models' / 'smplx'
            if default_smplx.exists():
                smplx_path = str(default_smplx)
        smplx_path = self._resolve_path(smplx_path) if smplx_path else None
        if smplx_path is None or not os.path.exists(smplx_path):
            raise FileNotFoundError(f"SMPL-X 모델 경로가 없습니다: {smplx_path}")
        self.smplx_path = smplx_path

        # BLADE API 로드
        if str(self.blade_repo) not in sys.path:
            sys.path.insert(0, str(self.blade_repo))

        try:
            from api.BLADE_API import BLADE_API
            from blade.cameras.utils import project_points_focal_length_pixel
        except Exception as e:
            raise RuntimeError(f"BLADE API 로드 실패: {e}") from e

        self._blade_api_cls = BLADE_API
        self._project_points_fn = project_points_focal_length_pixel

        # BLADE 설정 파일
        if self.cfg_path is None:
            cfg_default = self.blade_repo / 'blade' / 'configs' / 'blade_inthewild.py'
            self.cfg_path = str(cfg_default)
        self.cfg_path = self._resolve_path(self.cfg_path)
        if not os.path.exists(self.cfg_path):
            raise FileNotFoundError(f"BLADE 설정 파일이 없습니다: {self.cfg_path}")

        # SMPL-X 모델 로드
        self._load_smplx_model()

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        return str((self.project_root / path).resolve())

    def _load_smplx_model(self):
        """SMPL-X 모델 로드"""
        device = torch.device(self.device if torch.cuda.is_available() and 'cuda' in self.device else 'cpu')
        smplx_path = self.smplx_path
        if smplx_path is not None and os.path.basename(smplx_path) == 'smplx':
            smplx_path = os.path.dirname(smplx_path)
        self.smplx_model = smplx.create(
            smplx_path,
            model_type='smplx',
            gender='neutral',
            use_pca=False
        ).to(device)

    def _process_blade_batch(self,
                              batch_frames: List[Tuple[int, np.ndarray]],
                              batch_index: int,
                              fps: float,
                              frame_width: int,
                              frame_height: int,
                              ball_detector=None) -> List[PoseFrame3D]:
        """BLADE API로 배치 처리"""
        if not batch_frames:
            return []

        batch_dir = Path(tempfile.mkdtemp(prefix=f'blade_batch_{batch_index}_', dir=str(self.temp_output_root)))
        output_dir = batch_dir / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_list: Dict[str, Dict[str, str]] = {}
        frame_id_map: Dict[int, str] = {}

        for frame_number, frame in batch_frames:
            frame_id = f"frame_{frame_number:06d}"
            frame_id_map[frame_number] = frame_id
            img_path = batch_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(img_path), frame)
            batch_list[frame_id] = {'rgb_file': str(img_path)}

        blade = self._blade_api_cls(
            batch_list=batch_list,
            device=self._format_device(self.device),
            samples_per_gpu=1,
            workers_per_gpu=self.workers_per_gpu,
            temp_output_folder=str(output_dir),
            render_and_save_imgs=False,
            cfg=self.cfg_path,
            checkpoint=self.model_path
        )
        blade.process()

        pose_frames: List[PoseFrame3D] = []
        for frame_number, frame in batch_frames:
            frame_id = frame_id_map[frame_number]
            result_path = output_dir / f"{frame_id}.pth"
            if not result_path.exists():
                continue
            pose_frame = self._pose_from_blade_result(
                result_path=str(result_path),
                frame_number=frame_number,
                fps=fps,
                frame_width=frame_width,
                frame_height=frame_height,
                frame=frame,
                ball_detector=ball_detector
            )
            if pose_frame is not None:
                pose_frames.append(pose_frame)

        # 임시 파일 정리
        shutil.rmtree(batch_dir, ignore_errors=True)

        return pose_frames

    def _pose_from_blade_result(self,
                                result_path: str,
                                frame_number: int,
                                fps: float,
                                frame_width: int,
                                frame_height: int,
                                frame: np.ndarray,
                                ball_detector=None) -> Optional[PoseFrame3D]:
        """BLADE 결과(.pth)에서 PoseFrame3D 생성"""
        try:
            data = torch.load(result_path, map_location='cpu')
        except Exception:
            return None

        betas = self._reshape_pose(self._to_tensor(data.get('smplx_betas')), target_dim=10)
        body_pose = self._reshape_pose(self._to_tensor(data.get('smplx_body_pose')), target_dim=63)
        global_orient = self._reshape_pose(self._to_tensor(data.get('smplx_global_orientation')), target_dim=3)
        transl = self._reshape_pose(self._to_tensor(data.get('smplx_translation')), target_dim=3)
        left_hand_pose = self._reshape_pose(self._maybe_tensor(data.get('smplx_left_hand_pose')), target_dim=45)
        right_hand_pose = self._reshape_pose(self._maybe_tensor(data.get('smplx_right_hand_pose')), target_dim=45)

        camera_translation = self._to_numpy(data.get('camera_translation'))
        camera_focal_length = self._to_numpy(data.get('camera_focal_length'))
        camera_hw = self._to_numpy(data.get('camera_hw'))

        device = next(self.smplx_model.parameters()).device
        betas = betas.to(device)
        body_pose = body_pose.to(device)
        global_orient = global_orient.to(device)
        transl = transl.to(device)

        if left_hand_pose is not None:
            left_hand_pose = left_hand_pose.to(device)
        if right_hand_pose is not None:
            right_hand_pose = right_hand_pose.to(device)

        smplx_out = self.smplx_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose
        )

        joints = smplx_out.joints[0].detach().cpu().numpy()
        if joints.shape[0] < 54:
            padded = np.zeros((54, 3), dtype=joints.dtype)
            padded[:joints.shape[0]] = joints
            joints = padded
        joints = joints[:54]

        pelvis = joints[0].copy()
        joints_3d = joints - pelvis

        joints_2d = self._project_joints_to_2d(
            joints=joints,
            smplx_translation=self._to_numpy(data.get('smplx_translation')),
            camera_translation=camera_translation,
            camera_focal_length=camera_focal_length,
            camera_hw=camera_hw,
            frame_width=frame_width,
            frame_height=frame_height
        )

        confidence = np.ones(54, dtype=np.float32)
        timestamp = frame_number / fps if fps > 0 else 0.0

        pose_frame = PoseFrame3D(
            frame_number=frame_number,
            timestamp=timestamp,
            joints_3d=joints_3d.astype(np.float32),
            joints_2d=joints_2d.astype(np.float32),
            confidence=confidence,
            body_pose=body_pose.detach().cpu().numpy(),
            global_orient=global_orient.detach().cpu().numpy(),
            frame_width=frame_width,
            frame_height=frame_height
        )

        if ball_detector is not None:
            ball_result = ball_detector.detect_ball_with_box(frame, pose_frame)
            if ball_result:
                pose_frame.ball_position = (ball_result[0], ball_result[1])
                pose_frame.ball_bbox = (ball_result[2], ball_result[3], ball_result[4], ball_result[5])

        return pose_frame

    def _to_tensor(self, value: Optional[object]) -> torch.Tensor:
        if value is None:
            return torch.zeros((1, 1), dtype=torch.float32)
        if isinstance(value, torch.Tensor):
            return value.float()
        return torch.tensor(value, dtype=torch.float32)

    def _maybe_tensor(self, value: Optional[object]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.float()
        return torch.tensor(value, dtype=torch.float32)

    def _reshape_pose(self, value: Optional[torch.Tensor], target_dim: Optional[int] = None) -> Optional[torch.Tensor]:
        if value is None:
            return None
        flat = value.view(1, -1)
        if target_dim is not None and flat.shape[1] >= target_dim:
            flat = flat[:, :target_dim]
        return flat

    def _to_numpy(self, value: Optional[object]) -> Optional[np.ndarray]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.array(value)

    def _project_joints_to_2d(self,
                               joints: np.ndarray,
                               smplx_translation: Optional[np.ndarray],
                               camera_translation: Optional[np.ndarray],
                               camera_focal_length: Optional[np.ndarray],
                               camera_hw: Optional[np.ndarray],
                               frame_width: int,
                               frame_height: int) -> np.ndarray:
        """SMPL-X 3D 관절을 2D 정규화 좌표로 투영"""
        if joints.shape[0] == 0:
            return np.zeros((54, 2), dtype=np.float32)

        cam_h, cam_w = frame_height, frame_width
        if camera_hw is not None and camera_hw.size >= 2:
            cam_h, cam_w = int(camera_hw.reshape(-1)[0]), int(camera_hw.reshape(-1)[1])

        fx = fy = None
        if camera_focal_length is not None:
            focal = camera_focal_length.reshape(-1)
            if focal.size >= 2:
                fx, fy = float(focal[0]), float(focal[1])
            elif focal.size == 1:
                fx = fy = float(focal[0])

        if fx is None or fy is None or fx == 0 or fy == 0:
            # fallback: normalize by bounding box in 3D (approx)
            xy = joints[:, :2]
            min_xy = xy.min(axis=0)
            max_xy = xy.max(axis=0)
            denom = np.where((max_xy - min_xy) == 0, 1.0, (max_xy - min_xy))
            norm_xy = (xy - min_xy) / denom
            return np.clip(norm_xy, 0.0, 1.0)

        cx = cam_w / 2.0
        cy = cam_h / 2.0

        joints_cam = joints.copy()
        joints_no_transl = joints_cam.copy()
        if smplx_translation is not None and smplx_translation.size >= 3:
            t = smplx_translation.reshape(-1)[:3]
            joints_no_transl = joints_cam - t[None]

        proj_a = self._project_simple(joints_cam, None, fx, fy, cx, cy)
        proj_b = None
        if camera_translation is not None and camera_translation.size >= 3:
            cam_t = camera_translation.reshape(-1)[:3]
            proj_b = self._project_simple(joints_no_transl, cam_t, fx, fy, cx, cy)

        if proj_b is None:
            chosen = proj_a
        else:
            inside_a = self._count_points_inside(proj_a, cam_w, cam_h)
            inside_b = self._count_points_inside(proj_b, cam_w, cam_h)
            chosen = proj_b if inside_b >= inside_a else proj_a

        # 원본 프레임 크기에 맞게 스케일 보정
        if cam_w > 0 and cam_h > 0 and (cam_w != frame_width or cam_h != frame_height):
            scale_x = frame_width / cam_w
            scale_y = frame_height / cam_h
            chosen = chosen.copy()
            chosen[:, 0] *= scale_x
            chosen[:, 1] *= scale_y

        norm = np.zeros((chosen.shape[0], 2), dtype=np.float32)
        if frame_width > 0 and frame_height > 0:
            norm[:, 0] = chosen[:, 0] / frame_width
            norm[:, 1] = chosen[:, 1] / frame_height
        return np.clip(norm, 0.0, 1.0)

    def _project_simple(self,
                         points: np.ndarray,
                         translation: Optional[np.ndarray],
                         fx: float,
                         fy: float,
                         cx: float,
                         cy: float) -> np.ndarray:
        pts = points.copy()
        if translation is not None:
            pts = pts + translation[None]
        z = pts[:, 2].copy()
        z[z == 0] = 1e-6
        x = pts[:, 0]
        y = pts[:, 1]
        u = fx * x / z + cx
        v = fy * y / z + cy
        return np.stack([u, v], axis=-1)

    def _count_points_inside(self, points: np.ndarray, w: int, h: int) -> int:
        if points.size == 0:
            return 0
        inside = (points[:, 0] >= 0) & (points[:, 0] <= w) & (points[:, 1] >= 0) & (points[:, 1] <= h)
        return int(inside.sum())

    def _format_device(self, device: str) -> str:
        if device is None:
            return 'cpu'
        dev = str(device)
        if dev.startswith('cuda') and ':' not in dev:
            return 'cuda:0'
        return dev

    def extract_from_video(self, video_path: str,
                           ball_detector=None) -> List[PoseFrame3D]:
        """
        비디오에서 모든 프레임의 3D 포즈 추출

        Args:
            video_path: 비디오 파일 경로
            ball_detector: 공 탐지기 객체 (선택)

        Returns:
            List[PoseFrame3D]: 포즈가 감지된 프레임들의 리스트
        """
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.resize_width and self.resize_height:
            frame_width = int(self.resize_width)
            frame_height = int(self.resize_height)

        pose_frames: List[PoseFrame3D] = []
        frame_number = 0
        batch_frames: List[Tuple[int, np.ndarray]] = []
        batch_index = 0

        print(f"BLADE로 포즈 추출 중: {video_path}")
        print(f"총 프레임: {total_frames}, FPS: {fps}, 크기: {frame_width}x{frame_height}")

        with tqdm(total=total_frames, desc="3D 포즈 추출") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if self.max_frames is not None and frame_number >= self.max_frames:
                    break

                if self.resize_width and self.resize_height:
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)

                batch_frames.append((frame_number, frame))
                if len(batch_frames) >= self.batch_size:
                    pose_frames.extend(
                        self._process_blade_batch(
                            batch_frames,
                            batch_index,
                            fps,
                            frame_width,
                            frame_height,
                            ball_detector
                        )
                    )
                    batch_frames = []
                    batch_index += 1

                frame_number += 1
                pbar.update(1)

            if batch_frames:
                pose_frames.extend(
                    self._process_blade_batch(
                        batch_frames,
                        batch_index,
                        fps,
                        frame_width,
                        frame_height,
                        ball_detector
                    )
                )

        cap.release()

        # 시간적 스무딩 적용
        if self.temporal_smoothing and len(pose_frames) > 0:
            pose_frames = self._apply_temporal_smoothing(pose_frames)

        print(f"포즈 추출 완료: {len(pose_frames)}개 프레임")

        if ball_detector and len(pose_frames) > 0:
            ball_count = sum(1 for pf in pose_frames if pf.ball_position is not None)
            print(f"공 탐지: {ball_count}/{len(pose_frames)} 프레임 ({100*ball_count/len(pose_frames):.1f}%)")

        return pose_frames

    def _extract_single_frame(self, frame: np.ndarray, frame_number: int,
                               fps: float, frame_width: int, frame_height: int,
                               ball_detector=None) -> Optional[PoseFrame3D]:
        """단일 프레임에서 3D 포즈 추출"""
        return self._extract_blade(
            frame, frame_number, fps,
            frame_width, frame_height,
            ball_detector
        )

    def _extract_blade(self, frame: np.ndarray, frame_number: int,
                        fps: float, frame_width: int, frame_height: int,
                        ball_detector=None) -> Optional[PoseFrame3D]:
        """BLADE를 사용한 3D 포즈 추출 (단일 프레임)"""
        batch_frames = [(frame_number, frame)]
        pose_frames = self._process_blade_batch(
            batch_frames,
            batch_index=0,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            ball_detector=ball_detector
        )
        return pose_frames[0] if pose_frames else None

    def _extract_fallback(self, frame: np.ndarray, frame_number: int,
                           fps: float, frame_width: int, frame_height: int,
                           ball_detector=None) -> Optional[PoseFrame3D]:
        """
        폴백: MediaPipe를 사용하여 SMPL-X 형태로 변환

        MediaPipe 33개 관절을 SMPL-X 54개 관절로 매핑합니다.
        누락된 관절은 보간으로 추정합니다.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks or not results.pose_world_landmarks:
            return None

        timestamp = frame_number / fps

        # MediaPipe world_landmarks 추출 (미터 단위)
        mp_world = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.pose_world_landmarks.landmark
        ])

        # MediaPipe 정규화 좌표 추출
        mp_normalized = np.array([
            [lm.x, lm.y]
            for lm in results.pose_landmarks.landmark
        ])

        # visibility 추출
        mp_visibility = np.array([
            lm.visibility
            for lm in results.pose_landmarks.landmark
        ])

        # MediaPipe → SMPL-X 매핑
        joints_3d = self._mediapipe_to_smplx_3d(mp_world)
        joints_2d = self._mediapipe_to_smplx_2d(mp_normalized)
        confidence = self._mediapipe_to_smplx_confidence(mp_visibility)

        # 공 탐지
        ball_position = None
        ball_bbox = None
        if ball_detector:
            ball_result = ball_detector.detect_ball_with_box(frame)
            if ball_result:
                ball_position = (ball_result[0], ball_result[1])
                ball_bbox = (ball_result[2], ball_result[3], ball_result[4], ball_result[5])

        return PoseFrame3D(
            frame_number=frame_number,
            timestamp=timestamp,
            joints_3d=joints_3d,
            joints_2d=joints_2d,
            confidence=confidence,
            ball_position=ball_position,
            ball_bbox=ball_bbox,
            frame_width=frame_width,
            frame_height=frame_height
        )

    def _mediapipe_to_smplx_3d(self, mp_world: np.ndarray) -> np.ndarray:
        """
        MediaPipe 33개 관절을 SMPL-X 54개 관절로 변환 (3D)

        MediaPipe 인덱스:
            0: nose, 11: left_shoulder, 12: right_shoulder
            23: left_hip, 24: right_hip, 25: left_knee, 26: right_knee
            27: left_ankle, 28: right_ankle, 2: left_eye, 5: right_eye

        SMPL-X 인덱스:
            0: pelvis, 1: left_hip, 2: right_hip, 4: left_knee, 5: right_knee
            7: left_ankle, 8: right_ankle, 9: spine3, 15: head
            16: left_shoulder, 17: right_shoulder, 23: left_eye, 24: right_eye
        """
        joints_3d = np.zeros((54, 3))

        # 골반 = 양 엉덩이 중심
        joints_3d[0] = (mp_world[23] + mp_world[24]) / 2  # pelvis

        # 하체
        joints_3d[1] = mp_world[23]  # left_hip
        joints_3d[2] = mp_world[24]  # right_hip
        joints_3d[4] = mp_world[25]  # left_knee
        joints_3d[5] = mp_world[26]  # right_knee
        joints_3d[7] = mp_world[27]  # left_ankle
        joints_3d[8] = mp_world[28]  # right_ankle

        # 발 (발목 + 방향 추정)
        joints_3d[10] = mp_world[31]  # left_foot (LEFT_FOOT_INDEX)
        joints_3d[11] = mp_world[32]  # right_foot (RIGHT_FOOT_INDEX)

        # 상체
        shoulder_center = (mp_world[11] + mp_world[12]) / 2
        hip_center = (mp_world[23] + mp_world[24]) / 2

        # 척추 보간
        joints_3d[3] = hip_center + (shoulder_center - hip_center) * 0.25  # spine1
        joints_3d[6] = hip_center + (shoulder_center - hip_center) * 0.5   # spine2
        joints_3d[9] = hip_center + (shoulder_center - hip_center) * 0.75  # spine3

        # 목
        joints_3d[12] = mp_world[0] + (shoulder_center - mp_world[0]) * 0.5  # neck

        # 어깨
        joints_3d[13] = shoulder_center + (mp_world[11] - shoulder_center) * 0.3  # left_collar
        joints_3d[14] = shoulder_center + (mp_world[12] - shoulder_center) * 0.3  # right_collar
        joints_3d[16] = mp_world[11]  # left_shoulder
        joints_3d[17] = mp_world[12]  # right_shoulder

        # 팔
        joints_3d[18] = mp_world[13]  # left_elbow
        joints_3d[19] = mp_world[14]  # right_elbow
        joints_3d[20] = mp_world[15]  # left_wrist
        joints_3d[21] = mp_world[16]  # right_wrist

        # 머리
        joints_3d[15] = mp_world[0]  # head (nose 사용)
        joints_3d[22] = mp_world[0] + np.array([0, 0.02, 0])  # jaw
        joints_3d[23] = mp_world[2]  # left_eye
        joints_3d[24] = mp_world[5]  # right_eye

        # 골반 중심으로 좌표 변환 (뷰 불변성)
        pelvis = joints_3d[0].copy()
        joints_3d = joints_3d - pelvis

        return joints_3d

    def _mediapipe_to_smplx_2d(self, mp_normalized: np.ndarray) -> np.ndarray:
        """MediaPipe 정규화 좌표를 SMPL-X 형태로 변환"""
        joints_2d = np.zeros((54, 2))

        # 동일한 매핑 적용 (2D)
        joints_2d[0] = (mp_normalized[23] + mp_normalized[24]) / 2
        joints_2d[1] = mp_normalized[23]
        joints_2d[2] = mp_normalized[24]
        joints_2d[4] = mp_normalized[25]
        joints_2d[5] = mp_normalized[26]
        joints_2d[7] = mp_normalized[27]
        joints_2d[8] = mp_normalized[28]
        joints_2d[10] = mp_normalized[31]
        joints_2d[11] = mp_normalized[32]

        shoulder_center = (mp_normalized[11] + mp_normalized[12]) / 2
        hip_center = (mp_normalized[23] + mp_normalized[24]) / 2

        joints_2d[3] = hip_center + (shoulder_center - hip_center) * 0.25
        joints_2d[6] = hip_center + (shoulder_center - hip_center) * 0.5
        joints_2d[9] = hip_center + (shoulder_center - hip_center) * 0.75
        joints_2d[12] = mp_normalized[0] + (shoulder_center - mp_normalized[0]) * 0.5

        joints_2d[13] = shoulder_center + (mp_normalized[11] - shoulder_center) * 0.3
        joints_2d[14] = shoulder_center + (mp_normalized[12] - shoulder_center) * 0.3
        joints_2d[16] = mp_normalized[11]
        joints_2d[17] = mp_normalized[12]

        joints_2d[18] = mp_normalized[13]
        joints_2d[19] = mp_normalized[14]
        joints_2d[20] = mp_normalized[15]
        joints_2d[21] = mp_normalized[16]

        joints_2d[15] = mp_normalized[0]
        joints_2d[22] = mp_normalized[0]
        joints_2d[23] = mp_normalized[2]
        joints_2d[24] = mp_normalized[5]

        return joints_2d

    def _mediapipe_to_smplx_confidence(self, mp_visibility: np.ndarray) -> np.ndarray:
        """MediaPipe visibility를 SMPL-X 형태로 변환"""
        confidence = np.zeros(54)

        # 매핑된 관절의 신뢰도 복사
        confidence[0] = (mp_visibility[23] + mp_visibility[24]) / 2
        confidence[1] = mp_visibility[23]
        confidence[2] = mp_visibility[24]
        confidence[4] = mp_visibility[25]
        confidence[5] = mp_visibility[26]
        confidence[7] = mp_visibility[27]
        confidence[8] = mp_visibility[28]
        confidence[10] = mp_visibility[31]
        confidence[11] = mp_visibility[32]

        # 보간된 관절은 인접 관절의 평균
        shoulder_conf = (mp_visibility[11] + mp_visibility[12]) / 2
        hip_conf = (mp_visibility[23] + mp_visibility[24]) / 2

        confidence[3] = (hip_conf + shoulder_conf) / 2
        confidence[6] = (hip_conf + shoulder_conf) / 2
        confidence[9] = shoulder_conf
        confidence[12] = mp_visibility[0]

        confidence[13] = shoulder_conf
        confidence[14] = shoulder_conf
        confidence[16] = mp_visibility[11]
        confidence[17] = mp_visibility[12]

        confidence[18] = mp_visibility[13]
        confidence[19] = mp_visibility[14]
        confidence[20] = mp_visibility[15]
        confidence[21] = mp_visibility[16]

        confidence[15] = mp_visibility[0]
        confidence[22] = mp_visibility[0]
        confidence[23] = mp_visibility[2]
        confidence[24] = mp_visibility[5]

        return confidence

    def _apply_temporal_smoothing(self, pose_frames: List[PoseFrame3D]) -> List[PoseFrame3D]:
        """
        시간적 스무딩 적용

        이동 평균 필터를 사용하여 관절 위치의 떨림을 줄입니다.
        """
        if len(pose_frames) < self.smoothing_window:
            return pose_frames

        # 3D 관절 좌표 스택
        all_joints = np.array([pf.joints_3d for pf in pose_frames])  # (N, 54, 3)

        # 이동 평균 필터
        from scipy.ndimage import uniform_filter1d
        smoothed_joints = uniform_filter1d(all_joints, size=self.smoothing_window, axis=0)

        # 스무딩된 값으로 업데이트
        for i, pf in enumerate(pose_frames):
            pf.joints_3d = smoothed_joints[i]

        return pose_frames

    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()
