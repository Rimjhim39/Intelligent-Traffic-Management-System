
import cv2
from ultralytics import YOLO

# Load pretrained vehicle detection model
model = YOLO("yolo26n.pt")

# Read recorded traffic video
cap = cv2.VideoCapture("traffic.mp4")

if not cap.isOpened():
    print("Error: Could not open traffic.mp4")
    raise SystemExit

vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

while True:
    success, frame = cap.read()

    if not success:
        # Restart video when it reaches the end
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model(
        frame,
        conf=0.15,
        imgsz=960,
        verbose=False
    )

    counts = {
        "Car": 0,
        "Motorcycle": 0,
        "Bus": 0,
        "Truck": 0
    }

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])

            if class_id not in vehicle_classes:
                continue

            name = vehicle_classes[class_id]
            counts[name] += 1

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            cv2.putText(
                frame, name, (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

    # Count vehicles visible in this frame
    total = sum(counts.values())

    if total < 5:
        density = "LOW"
    elif total < 12:
        density = "MEDIUM"
    else:
        density = "HIGH"

    # Display traffic statistics
    cv2.rectangle(frame, (10, 10), (330, 155), (0, 0, 0), -1)

    lines = [
        f"Visible vehicles: {total}",
        f"Cars: {counts['Car']}",
        f"Motorcycles: {counts['Motorcycle']}",
        f"Buses: {counts['Bus']}",
        f"Trucks: {counts['Truck']}",
        f"Traffic density: {density}"
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (20, 32 + i * 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 255), 1
        )

    cv2.imshow("AI Traffic Density Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
