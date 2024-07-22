import numpy as np
import cv2


def cp_diff(bg1_img, ob_img):
    return np.abs(bg1_img - ob_img).mean(axis=2).astype(np.uint8)


def cp_bin_mask(diff_img, threshold):
    result = np.where(diff_img <= threshold, 0, 255)
    result = np.stack((result,)*3, axis=-1)
    return result


def replace_bg(bg1_img, bg2_img, ob_img):
    diff_img = cp_diff(bg1_img, ob_img)
    bin_mask = cp_bin_mask(diff_img, threshold=5)
    output = np.where(bin_mask == 255, ob_img, bg2_img)
    return output


def main():
    bg1_img = cv2.imread("GreenBackground.png", 1)
    bg1_img = cv2.resize(bg1_img, (640, 480))
    ob_img = cv2.imread("Object.png", 1)
    ob_img = cv2.resize(ob_img, (640, 480))
    bg2_img = cv2.imread("NewBackground.jpg", 1)
    bg2_img = cv2.resize(bg2_img, (640, 480))

    output = replace_bg(bg1_img, bg2_img, ob_img)

    cv2.imshow('img Window', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
