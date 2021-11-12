import cv2 as cv

from face_recognition_commented import detect


def faceDetector(gray_image, original_frame, face_cascade, eye_cascade):

    # image, scale factor, minimum number of neighbour zones
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    for(x, y, w, h) in faces:
        cv.rectangle(original_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = original_frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for(ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

    return original_frame


def main():
    face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv.CascadeClassifier("haarcascade_eye.xml")

    video_feed = cv.VideoCapture(0)

    while True:
        _, frame = video_feed.read()
        gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        detection = faceDetector(gray_image, frame, face_cascade, eye_cascade)
        cv.imshow("Video Feed", detection)

        if cv.waitKey(1) & 0xFF is ord("q"):
            break

    video_feed.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
