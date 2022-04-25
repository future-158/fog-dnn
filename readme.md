# .env 파일 생성
먼저 .env 파일을 만드시고, 
ftp_user=myid
ftp_password=mypassword
ftp_host=*.*.*.246
을 입력


# 환경 설치
- make venv
venv 폴더에 anaconda 환경 설치

- conda activate venv
terminal에서 conda 환경 활성화

# 실행
python src/download_data.py
python train_dnn.py
python post_train_dnn.py 

순서대로 실행

# 설명
conf/config.yaml이 설정파일입니다.

모델 돌아가는 지 테스트하려고
test_run: true
optuna_config:
  n_trials: 1

로 되어있습니다.
이로인해 각 epoch마다 처음 10 배치만 돌리며, trial은 1회만 돌립니다


test_run: false
optuna_config:
  n_trials: 50

으로 바꾸시면 정상적으로 돌아갑니다
