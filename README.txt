# ImageAI Inpaint Lab

AI 기반 이미지 편집 도구 (옷 제거, 상의/소매 변경 등)

Stable Diffusion XL (Juggernaut XL Ragnarok) + ControlNet + SAM + MediaPipe를 활용한 인페인팅/마스크 기반 이미지 편집 앱입니다.  
특히 **bare torso** (옷 제거) 결과에 초점을 맞춰 자연스러운 피부 톤, 포어, 조명 매칭을 목표로 합니다.


## 주요 기능
- **옷 제거 / 상의 변경** (Remove Clothes / Wear / Change Clothes 모드)
- **Manual 마스크**: SAM (vit_b/vit_h) 클릭으로 정밀 마스크 생성
- **Auto 마스크**: MediaPipe Tasks API 기반 자동 semantic 마스크 (상의/소매/바지/머리/배경)
- **ControlNet 지원**: depth / openpose (포즈/깊이감 유지, 로컬 로딩)
- **Refine Pass**: img2img로 톤/디테일 통일 (옵션, VRAM 절약 모드 추가)
- **Preview Prompt**: enrich된 프롬프트 미리보기 버튼 (auto-enrich 체크 시 적용)
- **Seed 제어**: 고정/랜덤 지원, 생성 후 로그 + 파일 저장으로 재현성 보장
- **GPU/CPU 자동 전환**: RTX 3080 Ti (GPU) 또는 CPU 모드 지원
- **VRAM 관리**: 생성 후 자동 청소 + xformers 지원 (속도 2~3배 향상)


## 요구사항
- **GPU 추천**: NVIDIA RTX 30xx 이상 (VRAM 12GB+)
- **CPU**: 가능하지만 느림 (30분 이상 소요 가능)
- **RAM**: 16GB 이상 (ControlNet + Refine 시 10~14GB 사용)
- **Python 3.11**
- **CUDA 12.1** (GPU 사용 시)


### 설치
1. **venv 생성 & 활성화**
   ```bash
   python -m venv venv311
   .\venv311\Scripts\activate

### 필수 패키지 설치
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git


### 모델 파일 다운로드
Juggernaut XL: models/stable-diffusion-xl/juggernautXL_ragnarokBy.safetensors
ControlNet: models/ControlNet/controlnet-depth-sdxl-1.0 & controlnet-openpose-sdxl-1.0
SAM: weights/sam_vit_b_01ec64.pth & sam_vit_h_4b8939.pth
MediaPipe: 자동 다운로드 (첫 실행 시)

### 실행
python app.py→ http://127.0.0.1:7860 열기

### 사용 방법
1. 이미지 업로드
2. Manual: SAM 클릭으로 마스크 그리기
3. Auto: 프롬프트 입력 → Auto Mask 생성 → 후보 선택
4. Positive / Negative 작성 (Auto-enrich 체크 시 자동 확장)
5. Preview Prompt 버튼으로 enrich된 프롬프트 미리 확인
6. ControlNet (depth/openpose) + Refine 옵션 선택
7. Seed 입력 (-1 랜덤, 고정값 재현)
8. Apply → 결과 확인 (VRAM 청소 버튼으로 메모리 관리)

### 알려진 이슈
CPU 모드: ControlNet 로드 시 RAM 부족 가능 (16GB 이상 추천)
Auto Mask (MediaPipe): Manual (SAM)에 비해 정밀도 낮음 (Tasks API 사용)
VRAM 누적: 생성 후 Clear VRAM 버튼으로 해결
토큰 초과: enrich_positive에서 BREAK로 multi-prompt 처리


### 향후 계획
xformers / torch.compile 추가 최적화 (속도 2~3배)
Multi-Prompt 지원 (토큰 77 제한 완전 우회)
Lora / IP-Adapter 지원 (스타일 강화)
Auto-enrich 한국어 지원 강화

### 라이선스
MIT License (개인/비상업 용도 자유)
Made with ❤️ by Hojun Hwang (January 16, 2026)