#/usr/bin/python3
"""
    Test RateMe on video stream
"""
import cv2
from RateMe.utils import RateMe, CAM
import argparse


parser = argparse.ArgumentParser(description="Like/dislike detector")
parser.add_argument("-c", "--camera", dest="cam",
                    help="Webcam index (0 by default),\
                          there could be video stream address",
                    metavar="CAM_ID",
                    default=0,
                    type=int)
args = parser.parse_args()


# load net
net = RateMe()
# access to webcam
cap = CAM(args.cam)

while cap.video.isOpened():
    # get new frame
    frame = cap.next()
    # get net predictions
    label = net.predict(frame)
    # plot detections on frame
    tmp = frame.copy()
    cv2.putText(tmp, "{}".format(str(label)),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)
    # show frame series
    cv2.imshow('FPS', tmp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
