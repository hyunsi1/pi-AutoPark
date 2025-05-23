## 1. 초기화 단계

1. **`main.py`**
    - 애플리케이션 시작
    - `config/config.yaml` 로부터 Geofence, 카메라 파라미터, 사전 정의된 슬롯(좌표·폭·높이) 로드
    - 로거(`utils/logger.py`) 초기화
    - 카메라 캘리브레이션 데이터(`camera/calibration.py`) 불러오기
2. **YOLO 모델 로드**
    - **YOLOv5n** (`models/yolo/config.yaml`, `best_weights.pt`) 사용
    - `vision/yolo_detector.py` 내부에서 모델 초기화

---

## 2. 주 모드 전환(상태 머신)

`main.py`에서 크게 두 모드를 순회합니다:

- **Calibration 모드**
- **Auto-Parking 모드**

필요에 따라 `scripts/calibrate_camera.py`로 별도 캘리브레이션을 수행합니다.

---

## 3. Auto-Parking 모드 흐름

```mermaid
flowchart TD
    A["run_parking.py 호출"] --> B["main.py 초기화 완료"]
    B --> C{모드 선택}
    C -->|Calibration| CAL["Calibration 모드"]
    C -->|Auto-Parking| D["UserIO.prompt_start()"]
    D --> E["SEARCH 상태"]

    E --> F["FrameCapture로 프레임 획득"]
    F --> G["YOLOv5n.detect()"]
    G --> H["거리 추정 (MonoDepthEstimator)"]
    H --> I{SlotAllocator로 빈 슬롯 탐색}
    I -->|빈 슬롯 발견| J["NAVIGATE로 전이"]
    I -->|없음| E

    J --> K["NAVIGATE 상태"]
    K --> L["현재 위치 계산 (카메라 뷰 기반)"]
    L --> M["PathPlanner.plan()"]
    M --> N{각 waypoint 순회}
    N -->|장애물 없음| O["Controller.navigate_to() 이동"]
    O --> N
    N -->|장애물 있음| P["OBSTACLE_AVOID로 전이"]

    P --> Q["Controller.stop() 정지"]
    Q --> R["PathPlanner.replan_around()"]
    R --> S["Controller.navigate_to() 이동"]
    S --> T["NAVIGATE 재진입"]

    T --> U["FINAL_APPROACH 상태"]
    U --> V["PanTiltController.tilt 아래로"]
    V --> W{depth ≤ threshold?}
    W -->|아직| V
    W -->|충분| X["COMPLETE로 전이"]

    X --> Y["UserIO.notify_complete()"]
    Y --> E
```

### 3.1. SEARCH 단계

1. **프레임 획득**
    - `camera/capture.py`: 웹캠 스트림 또는 이미지 파일
2. **슬롯 정보 로드**
    - `config/config.yaml`에 정의된 슬롯별 **화면 좌표, 폭(width), 높이(height)** 읽어오기
3. **장애물(킥보드) 검출**
    - `vision/yolo_detector.py` ▶ **YOLOv5n**를 통해 바운딩박스 반환
4. **거리 추정**
    - `vision/monodepth_estimator.py` ▶ 기하학 모델로 각 바운딩박스의 거리 계산
5. **빈 슬롯 후보 생성**
    - 사전 정의된 슬롯 중, 검출된 장애물이 없는 슬롯 리스트 필터링
6. **목표 슬롯 할당**
    - 첫 번째 빈 슬롯을 목표로 설정 ▶ state = NAVIGATE

### 3.2. NAVIGATION 단계

1. **경로 계획**
    - `navigation/path_planner.py` ▶ 장애물 회피 경로 생성
2. **주행 제어**
    - `navigation/controller.py` ▶ PID 제어로 스티어링·속도 제어
3. **전방 관찰**
    - `camera/pan_tilt_control.py` ▶ Pan 유지, Tilt = 0°

### 3.3. FINAL_APPROACH 단계

1. **근접 임계치 도달 감지**
    - `utils/distance.py` ▶ 현재 위치와 목표점 거리 확인
2. **정밀 모드 전환**
    - `camera/pan_tilt_control.py` ▶ Tilt → 아래 보기 (예: -30°)
3. **깊이 보정**
    - `vision/monodepth_estimator.py` ▶ 단안 기하학 계산으로 칸 중앙 보정
4. **Fine-Control**
    - `navigation/controller.py` ▶ 짧은 거리 PID 조향/속도 제어

### 3.4. COMPLETE 단계

1. **주차 완료 로깅**
    - `utils/logger.py` ▶ 최종 상태 로컬/서버 전송
2. **상태 초기화**
    - 다음 반납 요청을 위해 `state = SEARCH`
