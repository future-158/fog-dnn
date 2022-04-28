# .env 파일 생성
ftp_user
ftp_password
ftp_host
가 모두 있어야함.

# 환경 설치
- make venv
venv 폴더에 anaconda 환경 설치

- conda activate venv
terminal에서 conda 환경 활성화

- 각자 컴퓨터(wsl)에서 돌릴 경우



python src/download_data.py
python train_dnn.py
python post_train_dnn.py 

순서대로 실행


conf/config.yaml이 설정파일입니다.

모델 돌아가는 지 테스트하려고
test_run: true로 되어있습니다.
이로인해 각 epoch마다 처음 10배치만 돌립니다.
test_run: false로 바꾸시면 정상적으로 돌아갑니다

optuna_config:
  n_trials: 1
n_trials를 10~50으로 바꾸시면 하이퍼파라미터 튜닝이 됩니다



