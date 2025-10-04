# if __name__ == '__main__':
#    print_hi('PyCharm')


import cv2
import numpy as np

cam = cv2.VideoCapture('4695859-uhd_3840_2160_30fps.mp4')

w = 400
h = 200

mask_size = np.zeros((h, w), dtype=np.uint8)

upper_right = (w * 0.55, h * 0.75)
upper_left = (w * 0.45, h * 0.75)
lower_left = (0, h)
lower_right = (w, h)

shape = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)

mask = cv2.fillConvexPoly(mask_size, shape, 1)

mask_bounds = np.float32(shape)
frame_bounds = np.float32(np.array([(w, 0), (0, 0), (0, h), (w, h)]))
perspective = cv2.getPerspectiveTransform(mask_bounds, frame_bounds)

sobel_vertical = np.float32([
    [-1, -2, -1],
    [0, 0, 0],
    [+1, +2, +1]
])
sobel_orizontal = np.transpose(sobel_vertical)

while True:

    ret, frame = cam.read()
    if ret is False or frame is None:
        break

    frame = cv2.resize(frame, (w, h))
    cv2.imshow('Original', frame)
    ori_frame = frame.copy()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = frame * mask
    cv2.imshow('Trapez', frame)

    frame = cv2.warpPerspective(frame, perspective, (w, h))
    cv2.imshow('Top Down', frame)

    frame = cv2.blur(frame, ksize=(6, 6))
    cv2.imshow('Blur', frame)

    filtered_frame1 = cv2.filter2D(np.float32(frame), -1, sobel_vertical)
    filtered_frame2 = cv2.filter2D(np.float32(frame), -1, sobel_orizontal)
    frame = np.sqrt(filtered_frame1 ** 2 + filtered_frame2 ** 2)
    cv2.imshow('Filter', cv2.convertScaleAbs(frame))

    ret, frame = cv2.threshold(frame, 95, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binarized', frame)

    frame_copy = frame.copy()

    frame_copy[0: w // 40] = 0
    frame_copy[:, 0:w // 40] = 0
    frame_copy[h - h // 20:h, :] = 0
    frame_copy[:, w - w // 40:w] = 0

    cv2.imshow('copy no edges', frame_copy)

    left_frame, right_frame = np.hsplit(frame_copy, 2)

    left_values = np.argwhere(left_frame > 0)
    right_values = np.argwhere(right_frame > 0)

    left_xs = left_values[:, 1]
    left_ys = left_values[:, 0]

    right_xs = right_values[:, 1]+ w/2
    right_ys = right_values[:, 0]

    # cv2.imshow('huh', left_frame)
    if len(left_xs) and len(left_ys):
        b, a = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)

        left_top_y = 0
        t_left_top_x = (left_top_y - b) // a
        if -10 ** 8 < t_left_top_x < 10 ** 8:
            left_top_x = t_left_top_x

        left_bottom_y = h
        t_left_bottom_x = (left_bottom_y - b) // a
        if -10 ** 8 < t_left_bottom_x < 10 ** 8:
            left_bottom_x = t_left_bottom_x
        left_bottom_x = (left_bottom_y - b) / a

        left_top = int(left_top_x), int(left_top_y)
        left_bottom = int(left_bottom_x), int(left_bottom_y)

        left_frame = cv2.line(left_frame, left_top, left_bottom, (200, 0, 0), 3)
        cv2.imshow('line', left_frame)

    if len(right_ys) and len(right_ys):
        b, a = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

        right_top_y = 0
        t_right_top_x = (right_top_y - b) // a
        if -10 ** 8 < t_right_top_x < 10 ** 8:
            right_top_x = t_right_top_x

        right_bottom_y = h
        t_right_bottom_x = (right_bottom_y - b) // a
        if -10 ** 8 < t_right_bottom_x < 10 ** 8:
            right_bottom_x = t_right_bottom_x
        right_bottom_x = (right_bottom_y - b) / a

        right_top = int(right_top_x), int(right_top_y)
        right_bottom = int(right_bottom_x), int(right_bottom_y)

        right_frame = cv2.line(right_frame, right_top, right_bottom, (200, 0, 0), 3)
        cv2.imshow('line2', right_frame)

        blank_frame1 = np.zeros((h, w), dtype=np.uint8)
        blank_frame1 = cv2.line(blank_frame1, left_top, left_bottom, (255, 0, 0), 3)
        perspective_left_line = cv2.getPerspectiveTransform(frame_bounds, mask_bounds)
        perspective_left_line = cv2.warpPerspective(blank_frame1, perspective_left_line, (w, h))

        blank_frame2 = np.zeros((h, w), dtype=np.uint8)
        blank_frame2 = cv2.line(blank_frame2, right_top, right_bottom, (255, 0, 0), 3)
        perspective_right_line = cv2.getPerspectiveTransform(frame_bounds, mask_bounds)
        perspective_right_line = cv2.warpPerspective(blank_frame2, perspective_right_line, (w, h))

        #cv2.imshow('blank line', perspective_left_line)
        #cv2.imshow('blank line2', perspective_right_line)

        left_points = np.argwhere(perspective_left_line > 0)
        right_points = np.argwhere(perspective_right_line > 0)

        for x, y in left_points:
            ori_frame[x, y] = (50, 50, 250)
        for x, y in right_points:
            ori_frame[x, y] = (50, 250, 50)

        cv2.imshow('result', ori_frame)

    if cv2.waitKeyEx(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
