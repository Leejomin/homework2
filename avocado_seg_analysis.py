import cv2
import numpy as np
import matplotlib.pyplot as plt

path = {
    "base": "C:/boimaterial-teamproject/image/avocado_non_shadow.jpg",
    "test": "C:/boimaterial-teamproject/image/avocado_shadow.jpg"
}

config = {
    "size": (1920, 1440),
    "font_scale": 1.8,
    "thickness": 5,
    "text_color": (255, 255, 0),  
    "hsv_low_base": np.array([20, 10, 5]),
    "hsv_low_test": np.array([20, 30, 0]),
    "hsv_upp": np.array([120, 255, 255])
}

def add_styled_text(img, lines):
    out_img = img.copy()
    for i, line in enumerate(lines):
        pos = (50, 100 + (i * 90))
        cv2.putText(out_img, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 
                    config["font_scale"], (0, 0, 0),  config["thickness"] + 6)
        cv2.putText(out_img, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 
                    config["font_scale"], config["text_color"], config["thickness"])
    return out_img

def process_segmentation(path, mode="base"):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")
    
    img = cv2.resize(img, config["size"], interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Adaptive
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    ada = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 21, 7)
    
    # HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    low = config["hsv_low_base"] if mode == "base" else config["hsv_low_test"]
    hsv_mask = cv2.inRange(hsv, low, config["hsv_upp"])
    
    return img_rgb, ada, hsv_mask

def get_report_set(path, is_base=True, base_masks=None):
    img_rgb, ada, hsv_mask = process_segmentation(path, mode="base" if is_base else "test")
    
    ada_rgb = cv2.cvtColor(ada, cv2.COLOR_GRAY2RGB)
    hsv_rgb = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2RGB)
    
    if is_base:
        res = [
            add_styled_text(img_rgb, ["Base (Non-shadow)"]),
            add_styled_text(ada_rgb, ["Adaptive", f"Area: {cv2.countNonZero(ada)}px"]),
            add_styled_text(hsv_rgb, ["HSV", f"Area: {cv2.countNonZero(hsv_mask)}px"])
        ]
        return res, (ada, hsv_mask)
    else:
        ada_b, hsv_b = base_masks
        def get_stats(t_m, b_m):
            ta, ba = cv2.countNonZero(t_m), cv2.countNonZero(b_m)
            return (1 - abs(ta-ba)/ba)*100, ((ta-ba)/ba)*100

        acc_a, e_a = get_stats(ada, ada_b)
        acc_h, e_h = get_stats(hsv_mask, hsv_b)
        
        res = [
            add_styled_text(img_rgb, ["Test (Shadow)"]),
            add_styled_text(ada_rgb, ["Adaptive", f"Acc: {acc_a:.1f}%", f"Bias: {e_a:.1f}%"]),
            add_styled_text(hsv_rgb, ["HSV", f"Acc: {acc_h:.1f}%", f"Bias: {e_h:.1f}%"])
        ]
        return res


try:
 
    base_imgs, base_masks = get_report_set(path["base"], is_base=True)
    test_imgs = get_report_set(path["test"], is_base=False, base_masks=base_masks)

    all_imgs = base_imgs + test_imgs
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(all_imgs[i])
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"오류 발생: {e}")