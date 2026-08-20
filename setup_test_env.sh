#!/usr/bin/env bash
# =============================================================================
#  K-MELLODDY 표준데이터 전처리 SW v1.0 — 시험환경 원클릭 구축
#  성적서 신청서 ONY-26-K027 4p 선언 환경을 그대로 재현한다.
#
#  사용법:  bash setup_test_env.sh        (제품 루트에서 실행)
#  결과:    ./.venv  → 신청서 절차 3) 의 `source .venv/bin/activate` 가 그대로 동작
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PRODUCT_HOME="$PWD"
echo "[*] PRODUCT_HOME = $PRODUCT_HOME"

# ---------------------------------------------------------------- 1. Python 3.12.3 확보
# venv 를 실제로 만들 수 있는 3.12.3 인터프리터를 찾는다.
#   - venv 내부 python(sys.prefix != sys.base_prefix)은 ensurepip 이 없어 제외
#   - Ubuntu 시스템 python 은 python3.12-venv 미설치인 경우가 많아 실제 생성으로 검증
can_make_venv() {
  local py="$1" probe; probe="$(mktemp -d)"
  "$py" -c 'import sys; sys.exit(0 if sys.prefix == sys.base_prefix else 1)' 2>/dev/null || { rm -rf "$probe"; return 1; }
  "$py" -m venv "$probe/v" >/dev/null 2>&1 \
    && "$probe/v/bin/python" -m pip --version >/dev/null 2>&1 \
    && { rm -rf "$probe"; return 0; }
  rm -rf "$probe"; return 1
}

PY=""
for cand in python3.12 python3 python; do
  command -v "$cand" >/dev/null 2>&1 || continue
  [ "$("$cand" -V 2>&1 | awk '{print $2}')" = "3.12.3" ] || continue
  if can_make_venv "$cand"; then PY="$(command -v "$cand")"; break; fi
done

if [ -z "$PY" ]; then
  echo "[*] 사용 가능한 Python 3.12.3 없음 → conda 로 확보"
  command -v conda >/dev/null 2>&1 \
    || { echo "[!] conda 없음. miniconda 설치 후 재실행:"; \
         echo "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
         echo "    bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3"; exit 1; }
  conda create -y -p "$PRODUCT_HOME/.py3123" python=3.12.3
  PY="$PRODUCT_HOME/.py3123/bin/python"
  can_make_venv "$PY" || { echo "[!] conda python 으로도 venv 생성 실패"; exit 1; }
fi
echo "[*] Python: $PY ($("$PY" -V 2>&1))"

# ---------------------------------------------------------------- 2. venv 생성
rm -rf "$PRODUCT_HOME/.venv"
"$PY" -m venv "$PRODUCT_HOME/.venv"
VPY="$PRODUCT_HOME/.venv/bin/python"
"$VPY" -m pip install -q -U pip

# ---------------------------------------------------------------- 3. 패키지 설치
# torch 는 CPU 전용 휠로 먼저 설치한다.
#   - 신청서 4p "GPU 불필요 (CPU 연산만 사용)" 선언과 일치
#   - 기본(CUDA) 휠은 nvidia-* 포함 약 6GB, CPU 휠은 약 200MB
echo "[*] torch (CPU 전용) 설치"
"$VPY" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
echo "[*] 고정 버전 패키지 설치"
"$VPY" -m pip install -r "$PRODUCT_HOME/requirements-test.txt"

# ---------------------------------------------------------------- 4. 선언 버전 대조
echo
echo "[*] 신청서 4p 선언 버전 대조"
"$VPY" - <<'PYEOF'
import importlib.metadata as m, sys
want = {"pandas":"3.0.3","numpy":"2.4.6","rdkit":"2025.3.2","scikit-learn":"1.6.1",
        "scipy":"1.15.3","pint":"0.24.4","omegaconf":"2.3.1","openpyxl":"3.1.5"}
ok = sys.version.split()[0] == "3.12.3"
print(f"  {'python':14} 선언=3.12.3    실제={sys.version.split()[0]}  {'OK' if ok else '<<< 불일치'}")
for k, v in want.items():
    try: got = m.version(k)
    except Exception: got = "MISSING"
    if got != v: ok = False
    print(f"  {k:14} 선언={v:10} 실제={got}  {'OK' if got == v else '<<< 불일치'}")
sys.exit(0 if ok else 1)
PYEOF

# ---------------------------------------------------------------- 5. 기능검증 스모크
echo
echo "[*] 기능검증 스크립트 스모크 테스트"
"$VPY" tests/verify_features.py > /tmp/kmelloddy_verify.log 2>&1
grep -q "ALL CHECKS COMPLETED" /tmp/kmelloddy_verify.log \
  && echo "  OK — ALL CHECKS COMPLETED" \
  || { echo "  <<< 실패. /tmp/kmelloddy_verify.log 확인"; exit 1; }

# ---------------------------------------------------------------- 6. TC8 산출물 사전 생성
# 신청서 TC8 절차 7) 은 processed_data/data_sample_unmapped.csv 를 읽지만,
# 이 파일은 절차 4) 의 verify_features.py 가 아니라 절차 8) 의 CLI 가 만든다.
# 절차대로 진행해도 실패하지 않도록 미리 한 번 생성해 둔다.
echo
echo "[*] TC8 산출물 사전 생성 (절차 7) 이 읽을 파일)"
"$VPY" hits-preprocess.py --input_path input_data/data_sample.csv \
       --to-gist-matrix --endpoint_mapper manual >/dev/null 2>&1
"$VPY" -c "
import pandas as pd
print('  gist_matrix shape :', pd.read_csv('processed_data/data_sample_gist_matrix.csv').shape, '(기대 (81, 87))')
print('  unmapped 행수     :', len(pd.read_csv('processed_data/data_sample_unmapped.csv')), '(기대 18)')
"

echo
echo "============================================================"
echo " 구축 완료.  시험 당일:"
echo "   cd $PRODUCT_HOME"
echo "   source .venv/bin/activate"
echo "   python tests/verify_features.py"
echo "============================================================"
