#  --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --input videos/People_30.mp4 \
#  --output output/output_video.avi

from Tracker.centroidtracker import CentroidTracker
from Tracker.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.20,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=10,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

print("Loading model & video")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
vs = cv2.VideoCapture(args["input"])
writer = None

# initialize the frame dimensions (we'll set them as soon as we read the first frame from the video)
W = None
H = None

# instantiate our centroid tracker
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# for counting people
totalFrames = 0
totalDown = 0
totalUp = 0
total = 0

# start the frames per second throughput estimator
#fps = FPS().start()

# loop over frames from the video
while True:
	frame = vs.read()
	frame = frame[1]

	# if we did not grab a frame then we have reached the end of the video
	if args["input"] is not None and frame is None:
		break

	# resize the frame to have a maximum width of 500 pixels
	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# initialize the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(W, H), True)

	rects = []

	if totalFrames % args["skip_frames"] == 0:
		#initialize our new set of object trackers
		trackers = []

		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum confidence
			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])

				if CLASSES[idx] != "person":
					continue

				# compute the (x, y)-coordinates of the bounding box
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")
				cv2.rectangle(frame, (startX, startY), (endX, endY),
							  (255, 0, 0), 2)

				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)
				trackers.append(tracker)

	else:
		# loop over the trackers
		for tracker in trackers:
			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))

	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		to = trackableObjects.get(objectID, None)

		if to is None:
			to = TrackableObject(objectID, centroid)

		else:
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			if not to.counted:
				if direction < 0 and centroid[1] < H // 2: # if the direction is moving up, count the object
					totalUp += 1
					to.counted = True

				# if the direction is moving down), count the object
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.counted = True
				total = totalUp+totalDown

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

	# construct a tuple of information we will be displaying on the frame
	info = [
		("Up", totalUp),
		("Down", totalDown),
		("Total", total)
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 34, 25), 2)

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	#exit
	if key == ord("q"):
		break

	totalFrames += 1

cv2.destroyAllWindows()