yolo_output = original_img.copy()

results = model(test_image_path)
detections = results[0].boxes

vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck
vehicle_count = 0

for box in detections:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    
    if cls in vehicle_classes and conf > 0.35:
        vehicle_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(yolo_output, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.rectangle(yolo_output, (x1, y1-20), (x1+100, y1), (0,0,0), -1)
        cv2.putText(yolo_output, f"Vehicle {conf:.2f}", (x1+5, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

show_img(f"YOLOv8 Detection (Count: {vehicle_count})", yolo_output)
