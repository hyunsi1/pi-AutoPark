import sys
import threading
import time
import logging
import select
from datetime import datetime

class UserIO:
    """
    터미널 기반 사용자 I/O
    - 주차 시작/취소 입력
    - 상태별 진행 메시지 출력
    - 완료 시 알림 및 사운드(시스템 벨)
    """
    def __init__(self):
        self._stop_event = threading.Event()
        self.requested = False

    def prompt_start(self):
        """주차 시작 요청 대기"""
        print("자동 주차 시스템을 시작하려면 Enter 키를 누르세요. 종료: q + Enter")
        while True:
            s = sys.stdin.readline().strip().lower()
            if s == 'q':
                print("프로그램을 종료합니다.")
                sys.exit(0)
            # Enter (빈 문자열)
            break
        self.requested = True
        logging.info("User requested start.")

    def show_status(self, msg: str):
        """터미널에 상태 메시지 출력"""
        now = datetime.now()
        timestamp = now.strftime('%H:%M:%S.') + f"{int(now.microsecond/1000):03d}"
        print(f"[{timestamp}] {msg}")
        logging.debug(f"Status: {msg}")

    def notify_complete(self):
        """주차 완료 알림"""
        print("\n===== 주차 완료! =====")
        # 시스템 벨을 울려 사용자에게 알림
        print("\a")  # BEL
        logging.info("User notified of completion.")

    def wait_cancel(self, timeout: float):
        """
        취소 요청 대기 (timeout 초 내에 q 입력 시 True 반환)
        """
        if timeout > 0:
            print(f"{timeout:.1f}초 내에 주차를 취소하려면 q + Enter를 누르세요.")
        start = time.time()
        while time.time() - start < timeout:
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                s = sys.stdin.readline().strip().lower()
                if s == 'q':
                    print("주차를 취소합니다.")
                    return True

if __name__ == '__main__':
    ui = UserIO()
    ui.prompt_start()
    ui.show_status("SEARCH 단계 진입")
    time.sleep(1)
    ui.show_status("NAVIGATE 단계 진입")
    time.sleep(1)
    ui.show_status("FINAL_APPROACH 단계 진입")
    time.sleep(1)
    ui.notify_complete()
