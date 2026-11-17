
import cv2
from ultralytics import YOLO
from stable_baselines3 import DQN

detector = YOLO("yolo26n.pt")
dqn = DQN.load("traffic_dqn")

cap = cv2.VideoCapture("traffic.mp4")
if not cap.isOpened():
    raise SystemExit("Cannot open traffic.mp4")

vehicle_classes = {2, 3, 5, 7}

# Choose "DQN" or "TRAFFIC_AWARE"
CONTROLLER = "DQN"

phase = 0
total_frames = 0
dqn_differences = 0

print("Press Q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    height, width = frame.shape[:2]
    middle = width // 2

    results = detector(frame, conf=0.15, verbose=False)

    left_count = 0
    right_count = 0

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            center_x = (x1 + x2) // 2

            if center_x < middle:
                left_count += 1
            else:
                right_count += 1

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 255, 0), 2
            )

    # DQN action
    obs = [
        min(left_count, 50) / 50,
        min(right_count, 50) / 50,
        float(phase)
    ]
    dqn_action, _ = dqn.predict(obs, deterministic=True)
    dqn_action = int(dqn_action)

    # Simple traffic-aware baseline
    if left_count > right_count:
        traffic_action = 0
    elif right_count > left_count:
        traffic_action = 1
    else:
        traffic_action = phase  # Keep current choice on a tie

    if CONTROLLER == "DQN":
        action = dqn_action
    else:
        action = traffic_action

    if dqn_action != traffic_action:
        dqn_differences += 1

    total_frames += 1
    phase = action

    signal = (
        "LEFT REGION GREEN" if action == 0
        else "RIGHT REGION GREEN"
    )

    cv2.line(frame, (middle, 0), (middle, height), (255, 255, 0), 2)
    cv2.rectangle(frame, (10, 10), (510, 175), (0, 0, 0), -1)

    lines = [
        f"Left vehicles: {left_count}",
        f"Right vehicles: {right_count}",
        f"Controller: {CONTROLLER}",
        f"Decision: {signal}",
        f"DQN vs baseline differences: {dqn_differences}/{total_frames}"
    ]

    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (20, 38 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0) if i == 3 else (255, 255, 255), 2
        )

    cv2.imshow("Intelligent Traffic Management", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Frames evaluated: {total_frames}")
print(f"DQN/baseline decision differences: {dqn_differences}")
