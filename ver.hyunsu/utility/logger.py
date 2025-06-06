# utils/logger.py

import logging
import sys
from pathlib import Path

def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO,
    fmt: str = "[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S"
) -> logging.Logger:
    """
    이름(name)으로 로거를 생성하고,
    콘솔 핸들러와 (선택적으로) 파일 핸들러를 설정합니다.

    :param name: 로거 이름 (보통 __name__ 사용)
    :param log_file: 로그를 저장할 파일 경로 (None이면 파일 기록 안 함)
    :param level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR 등)
    :return: 설정된 Logger 인스턴스
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 포맷터 설정
    formatter = logging.Formatter(fmt, datefmt)

    # 콘솔 핸들러
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 파일 핸들러 (옵션)
    if log_file:
        # 로그 디렉토리 자동 생성
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # 중복 핸들러 추가 방지
    logger.propagate = False
    return logger
