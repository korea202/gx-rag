# gx-rag

git clone https://github.com/korea202/gx-rag.git
cd gx-rag
(새로 구성할시) uv init / uv venv --python 3.11
(기존환경 셋팅시) uv sync
(테스트) uv run src/test.py

#가상환경 실행(optional) source .venv/bin/activate

#pytorch 설치(auto gpu 환경 인식, uv sync 실패시)
UV_TORCH_BACKEND=auto uv pip install torch torchvision torchaudio
#나머지 라이브러리 설치 uv pip install -r requirements.txt


실행
uv run streamlit run front_app/holmes_ui.py

uv run src/apps/persona/holmes.py