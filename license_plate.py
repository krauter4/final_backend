import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in plates:
        plate_roi = gray[y:y+h, x:x+w]
        plate_text = pytesseract.image_to_string(plate_roi, config='--psm 6')
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, plate_text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print("License Plate:", plate_text.strip())

    cv2.imshow("License Plate Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
