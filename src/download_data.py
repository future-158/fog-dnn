from genericpath import exists
import requests
import wget
from omegaconf import OmegaConf
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
cfg = OmegaConf.load("conf/config.yaml")

"""
서버에서는 연구소 ftp 자료를 다운 못받음. 수동으로 upload해야함.
"""

station_codes = ['SF_0001',
 'SF_0002',
 'SF_0003',
 'SF_0004',
 'SF_0005',
 'SF_0006',
 'SF_0007',
 'SF_0008',
 'SF_0009',
 'SF_0010',
 'SF_0011']



for station_code in station_codes:
    try:
        url = cfg.template.data.format(
            **{
                'ftp_user': os.environ['ftp_user'],
                'ftp_password': os.environ['ftp_password'],
                'ftp_host': os.environ['ftp_host'],
                'station_code': station_code,
            }
        )
        out = Path('data') / 'clean' / os.path.basename(url)
        out.parent.mkdir(parents=True, exist_ok=True)
        wget.download(url, out.as_posix())
    except Exception:
        pass

for station_code in station_codes:
    try:
        url = cfg.template.dataset.format(
            **{
                'ftp_user': os.environ['ftp_user'],
                'ftp_password': os.environ['ftp_password'],
                'ftp_host': os.environ['ftp_host'],
                'station_code': station_code,
            }
        )
        out = Path('data') / 'processed' / os.path.basename(url)
        out.parent.mkdir(parents=True, exist_ok=True)
        wget.download(url, out.as_posix())
    except Exception:
        pass
