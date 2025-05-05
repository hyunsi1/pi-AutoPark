import logging
import sys

from interface.user_io import UserIO
from fsm.state_machine import StateMachine


def main():
    # 로그 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    logger.info("AutoPark 시작")

    # 사용자 I/O
    ui = UserIO()
    ui.prompt_start()  # Enter로 시작, q로 종료

    # 상태 머신 실행
    try: 
        sm = StateMachine(config_path="config/config.yaml")
        sm.run()
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"예기치 못한 오류 발생: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
