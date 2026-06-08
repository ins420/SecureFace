# 공식 가중치 다운로드 가이드

## 파일 정보
- 파일명: hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth
- 저장위치: Z:\캡스톤디자인\SecureFace-RX\checkpoints\
- 출처: BaiduDisk https://pan.baidu.com/s/1q-s1G4aqSzcXEofDOEfeHg  (비밀번호: 3cvh)

## 방법 1 — 브라우저 직접 다운로드
1. https://pan.baidu.com/s/1q-s1G4aqSzcXEofDOEfeHg 접속
2. 비밀번호 입력: 3cvh
3. 파일 선택 후 다운로드
4. checkpoints/ 폴더에 이동

## 방법 2 — bypy (Python BaiduDisk 클라이언트)
pip install bypy
bypy info
# 브라우저에서 인증 후
bypy downfile /apps/bypy/hybridAll_inv3_...pth checkpoints/

## 방법 3 — BaiduPCS-Go (Go 기반 CLI)
# https://github.com/qjfoidnh/BaiduPCS-Go/releases 에서 Windows 실행파일 다운로드
BaiduPCS-Go login -bduss=<your_bduss_token>
BaiduPCS-Go download /path/to/weights.pth

## 다운로드 후 확인
python -c "
import torch
w = torch.load('checkpoints/hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000.pth', map_location='cpu')
print('타입:', type(w))
print('키 수:', len(w) if isinstance(w, dict) else 'N/A')
if isinstance(w, dict):
    keys = list(w.keys())[:3]
    print('키 예시:', keys)
"
