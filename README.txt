******************************************************************************************************
Using Local CPU

    # 1. venv 활성화
    .\venv311\Scripts\activate

    # 2. CUDA 버전 torch 설치 (RTX 3080 Ti = CUDA 12.1)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # 3. 나머지 패키지 설치
    pip install -r requirements.txt

    # 4. segment-anything은 git으로 설치 (pip에 없음)
    pip install git+https://github.com/facebookresearch/segment-anything.git


******************************************************************************************************