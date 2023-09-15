import cv2
import numpy as np

# Video dosyasının yolunu belirtin
#video_path = "c:/vidi.mp4"
cap = cv2.VideoCapture(0)

# Takip edilecek nesnenin başlangıç koordinatlarını ve boyutunu sıfırlayın
x, y, w, h = 0, 0, 0, 0
selection = None
drag_start = None
tracking_started = False

# Kalman filtresini oluşturun
kalman_filter = cv2.KalmanFilter(4, 2)
kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]], np.float32)
kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], np.float32)
kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], np.float32)
kalman_filter.measurementNoiseCov = np.array([[1, 0],
                                              [0, 1]], np.float32)
kalman_filter.errorCovPost = np.array([[0.1, 0],
                                       [0, 0.1]], np.float32)

# Eklemek istediğiniz metin
top_left_text = "Threat Image Projection"
top_right_text = "MILITEK-SOLJECTION SYSTEMS"
bottom_left_text = "UMT-4 CAM/2"
bottom_right_text = "NOMINAL"
center_text="";

# Efekt uygulamayı kontrol etmek için bir bayrak
apply_effect = False

# Fare olaylarına yanıt vermek için işlevi tanımlayın
def on_mouse(event, x_pos, y_pos, flags, param):
    global x, y, w, h, selection, drag_start, tracking_started

    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x_pos, y_pos)
        tracking_started = False
        x, y, w, h = x_pos, y_pos, 0, 0

    if drag_start:
        x, y, w, h = drag_start[0], drag_start[1], x_pos - drag_start[0], y_pos - drag_start[1]
        tracking_started = False

        if w < 0:
            x, w = x + w, -w
        if h < 0:
            y, h = y + h, -h

        if w > 0 and h > 0:
            selection = (x, y, w, h)

    if event == cv2.EVENT_LBUTTONUP:
        drag_start = None
        if w > 0 and h > 0:
            tracking_started = True

cv2.namedWindow("Object Tracking", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Object Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.setMouseCallback("Object Tracking", on_mouse)

termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if tracking_started:
        x, y, w, h = selection

        roi = frame[y:y+h, x:x+w]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        _, track_window = cv2.meanShift(dst, (x, y, w, h), termination_criteria)
        x, y, w, h = track_window

        result_frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        label_position = (x, y - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.1
        font_color = (120, 120, 255)
        font_thickness = 2
        cv2.putText(result_frame, "PROB. TIP", label_position, font, font_scale, font_color, font_thickness)

        # Kalman filtresini güncelle
        measurement = np.array([[x + w / 2], [y + h / 2]], np.float32)
        kalman_filter.correct(measurement)
        prediction = kalman_filter.predict()
        x, y = prediction[0][0], prediction[1][0]
        if apply_effect:
            center_text = "Adaptive Vision Mode"
            font_scale = 2
            (text_width, text_height), _ = cv2.getTextSize(center_text, font, font_scale, font_thickness)
            text_x = (frame.shape[1] - text_width) // 2
            text_y = text_height + 10  # 10 piksel üstte görünsün
            cv2.putText(result_frame, center_text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    else:
        result_frame = frame.copy()
        if selection:
            x, y, w, h = selection
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Köşelere metin ekleyin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    font_color = (0, 255, 0)
    font_thickness = 2

    cv2.putText(result_frame, top_left_text, (10, 30), font, font_scale, font_color, font_thickness)
    cv2.putText(result_frame, top_right_text, (result_frame.shape[1] - 150, 30), font, font_scale, font_color, font_thickness)
    cv2.putText(result_frame, bottom_left_text, (10, result_frame.shape[0] - 10), font, font_scale, font_color, font_thickness)
    cv2.putText(result_frame, bottom_right_text, (result_frame.shape[1] - 150, result_frame.shape[0] - 10), font, font_scale, font_color, font_thickness)

    # Efekt uygulamayı kontrol etmek için "e" tuşunu kullanın
    if apply_effect:
        # Gri tonları artırma efekti uygulayın
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2GRAY)
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Object Tracking", result_frame)

    # Tuşlara basılmasını kontrol etmek için tuşları dinleyin
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        apply_effect = True
    elif key == ord('r'):
        apply_effect = False

cap.release()
cv2.destroyAllWindows()
