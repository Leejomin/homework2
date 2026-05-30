import cv2
import numpy as np
import math
import os

# 앞선 step1, step2 결과 이어받기 가정
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'images', 'apple_side_A.png')
img_array = np.fromfile(image_path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
original_display = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 가장 큰 윤곽선인 사과만 선택하여 자나 글자 윤곽선 필터링
if len(contours) > 0:
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area >= 500:
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            # A. 원형도 (Circularity) 산출
            circularity = (4 * math.pi * area) / (perimeter ** 2)
            
            # B. 구형도용 (간접 2D Bounding Box 치수 기반)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            dim1, dim2 = rect[1]
            pixel_L = max(dim1, dim2)
            pixel_W = min(dim1, dim2)
            
            pixel_T = 304.1
            GMD = (pixel_L * pixel_W * pixel_T) ** (1/3)
            sphericity = (GMD / pixel_L) * 100
            
            # 화면 표출 텍스트 및 윤곽선
            cv2.drawContours(original_display, [cnt], -1, (0, 255, 0), 2)
            cv2.drawContours(original_display, [box], 0, (255, 0, 0), 2)
            
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                cv2.putText(original_display, f"Circularity: {circularity:.3f}", (cx - 60, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(original_display, f"Sphericity: {sphericity:.1f}%", (cx - 60, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imshow("Step 3: Final Shape Analysis", original_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
