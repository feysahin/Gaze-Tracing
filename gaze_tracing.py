import cv2
import dlib
import numpy as np
import os.path
import time

from serial import Serial

video_capture = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()

dir_path = os.path.dirname(os.path.realpath(__file__))
landmark_file_path = os.path.realpath(os.path.join(dir_path, 'shape_predictor_68_face_landmarks.dat'))
predictor = dlib.shape_predictor(landmark_file_path)


def rect_to_bounding_box(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def scale_rect(rect, scale):
    (x, y, w, h) = rect
    return int(x / scale), int(y / scale), int(w / scale), int(h / scale)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def scale_point(point, scale):
    (x, y) = point
    return int(x / scale), int(y / scale)


def trace_face(frame):
    scale = 300 / min(frame.shape[1], frame.shape[0])
    thumb = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    triangles = []
    down_y = up_y = 1

    if faces:

        for face_rect in faces:
            shape = predictor(gray, face_rect)
            shape = shape_to_np(shape)

            # left = [36, 37, 38, 39, 40, 41]
            # right = [42, 43, 44, 45, 46, 47]

            eye_marks = np.asarray(shape[36:48,:2])

            # Display the landmarks
            for x, y in eye_marks:
                cv2.circle(gray, (x, y), 1, (0, 0, 255), -1)

            _, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

            ll_tri = [eye_marks[0], eye_marks[1], eye_marks[5]]     # 36-37-41
            lr_tri = [eye_marks[2], eye_marks[3], eye_marks[4]]     # 38-39-40

            l_middle = np.asarray([(eye_marks[1][0] + eye_marks[2][0]) // 2, (eye_marks[1][1] + eye_marks[5][1]) // 2])
            lu_tri = [eye_marks[1], eye_marks[2], l_middle]     # 37-38-l_middle
            ld_tri = [eye_marks[4], eye_marks[5], l_middle]     # 40-41-l_middle

            up_y = eye_marks[1][1] - eye_marks[0][1]
            down_y = eye_marks[0][1] - eye_marks[5][1]

            contours = [ll_tri, lr_tri, lu_tri, ld_tri]
            triangles = []

            for cnt in contours:
                rect1 = cv2.boundingRect(np.array(cnt))
                (x, y, w, h) = rect1
                triangle = gray[y: y + h, x: x + w]
                cropped_tr_mask = np.zeros((h, w), np.uint8)
                points = np.array([[cnt[0][0] - x, cnt[0][1] - y],
                                   [cnt[1][0] - x, cnt[1][1] - y],
                                   [cnt[2][0] - x, cnt[2][1] - y]], np.int32)

                cv2.fillConvexPoly(cropped_tr_mask, points, 255)
                triangles.append(cv2.bitwise_and(triangle, triangle, mask=cropped_tr_mask))
                gray = cv2.polylines(gray, np.array([cnt]), True, (0, 0, 255), 1)

    else:
        print("no face")
    return gray, triangles, (up_y - down_y)


#       0
#       |
#   1 - x - 3
#       |
#       2

# CONNECTION
print("Start")
port = "COM6"
bluetooth=Serial(port, 9600)
print("Connected")
bluetooth.flushInput()
print("Connected2")

du_ratio = 0.75

movements = ["FORWARD", "LEFT", "BACK", "RIGHT", "CENTER"]
buffer = []
start_time = time.time()

while True:

    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    face_trace_frame, triangles_f, is_down = trace_face(frame)
    cv2.imshow('Video', face_trace_frame)

    #cv2.imshow("left", triangles_f[0])
    #cv2.imshow("right", triangles_f[1])
    #cv2.imshow("up", triangles_f[2])
    #cv2.imshow("down", triangles_f[3])

    if triangles_f and (time.time() - start_time < 2):
        # ratio of 1's to 0's in each frame
        ratio_l = np.sum(triangles_f[0] == 255) / np.sum(triangles_f[0] == 0)
        ratio_r = np.sum(triangles_f[1] == 255) / np.sum(triangles_f[1] == 0)
        ratio_u = np.sum(triangles_f[2] == 255) / np.sum(triangles_f[2] == 0)
        ratio_d = np.sum(triangles_f[3] == 255) / np.sum(triangles_f[3] == 0)

        ratio_l_r = ratio_l / (ratio_r + 0.000001)
        dif_d_u = ratio_d - ratio_u

        # looking UP if there are more 1's in down-frame than up-frame
        if (0.2 < ratio_l_r < 5) and (dif_d_u > du_ratio):
            buffer.append(0)

        # looking BACK if there are more 1's in up-frame than down-frame
        elif is_down > 0:
            buffer.append(2)

        # looking NOWHERE
        elif is_down <= 0 and (0.3 < ratio_l_r < 5):
            buffer.append(4)

        # looking LEFT if there are more 1's in right-frame than left-frame
        elif ratio_l_r < 0.3:
            buffer.append(1)

        # looking RIGHT if there are more 1's in left-frame than right-frame
        elif ratio_l_r > 5:
            buffer.append(3)

        if cv2.waitKey(1) == 27:
            break

    # make a vote-system to evaluate the frames collected in 2 seconds 
    elif buffer:
        move_idx = np.bincount(buffer).argmax()
        print("MOVE -> ", movements[move_idx])

        if move_idx == 0:   bluetooth.write(b"FORWARD")
        elif move_idx == 1:   bluetooth.write(b"LEFT")
        elif move_idx == 2:   bluetooth.write(b"BACK")
        elif move_idx == 3:   bluetooth.write(b"RIGHT")

        time.sleep(3)

        buffer = []
        start_time = time.time()

    else:
        start_time = time.time()

video_capture.release()
cv2.destroyAllWindows()
bluetooth.close()
print("Done")
