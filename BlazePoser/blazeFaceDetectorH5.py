import time
import cv2
import numpy as np
import tensorflow as tf
import math
from blazeFaceUtils import gen_anchors, SsdAnchorsCalculatorOptions

KEY_POINT_SIZE = 6
MAX_FACE_NUM = 100
INPUT_FRONT = 128
INPUT_BACK = 256

###################################
#####################################

class EMAFilter:
    def __init__(self, alpha: float, initial_value: float = 0.0):
        """
        alpha: smoothing factor in (0,1)
        initial_value: starting state for the filter
        """
        assert 0.0 < alpha <= 1.0, "alpha must be in (0,1]"
        self.alpha = alpha
        self.state = initial_value
        self.initialized = False

    def update(self, measurement: float) -> float:
        """Ingest a new measurement and return the smoothed value."""
        if not self.initialized:
            # On the first call, just set the state to the measurement
            self.state = measurement
            self.initialized = True
        else:
            self.state = self.alpha * measurement + (1.0 - self.alpha) * self.state
        return self.state


###################################
###################################
def EulerToMatrix(roll, yaw, pitch):
    # roll - z axis
    # yaw  - y axis
    # pitch - x axis
    roll = roll / 180 * np.pi
    yaw = yaw / 180 * np.pi
    pitch = pitch / 180 * np.pi
 
    Rz = [[math.cos(roll), -math.sin(roll), 0],
          [math.sin(roll), math.cos(roll), 0],
          [0, 0, 1]]
 
    Ry = [[math.cos(yaw), 0, math.sin(yaw)],
          [0, 1, 0],
          [-math.sin(yaw), 0, math.cos(yaw)]]
 
    Rx = [[1, 0, 0],
          [0, math.cos(pitch), -math.sin(pitch)],
          [0, math.sin(pitch), math.cos(pitch)]]
 
    matrix = np.matmul(Rx, Ry)
    matrix = np.matmul(matrix, Rz)
    return matrix

def drawAxis_simo(img, headpose, tdx , tdy,  size=100):
    roll, yaw, pitch = headpose[0], headpose[1], headpose[2]
    
    matrix = EulerToMatrix(-roll, -yaw, -pitch)
 
    Xaxis = np.array([matrix[0, 0], matrix[1, 0], matrix[2, 0]]) * size
    Yaxis = np.array([matrix[0, 1], matrix[1, 1], matrix[2, 1]]) * size
    Zaxis = np.array([matrix[0, 2], matrix[1, 2], matrix[2, 2]]) * size
 
    cv2.line(img, (int(tdx), int(tdy)), (int(Xaxis[0]+tdx), int(-Xaxis[1]+tdy)), (0, 255, 0), 3)  # pitch
    cv2.line(img, (int(tdx), int(tdy)), (int(-Yaxis[0]+tdx), int(Yaxis[1]+tdy)), (0, 0, 255), 3)  # yaw
    cv2.line(img, (int(tdx), int(tdy)), (int(Zaxis[0]+tdx), int(-Zaxis[1]+tdy)), (255, 0, 0), 2)  # roll
 
    return img


class blazeFaceDetector():

	def __init__(self, scoreThreshold = 0.4, iouThreshold = 0.3):
		self.scoreThreshold = scoreThreshold
		self.iouThreshold = iouThreshold
		self.sigmoidScoreThreshold = np.log(self.scoreThreshold/(1-self.scoreThreshold))
		self.fps = 0
		self.timeLastPrediction = time.time()
		self.frameCounter = 0

		# Initialize model based on model type
		self.initializeModel()

		# Generate anchors for model
		self.generateAnchors()

	def initializeModel(self):
		
		#self.interpreter = tf.keras.models.load_model("UnifiedModels/reg1-c1a88c64-reg2-cl4obelj.h5") # the largest model by far, and the best model on test_mae for reg1
		#self.interpreter = tf.keras.models.load_model("UnifiedModels/reg1-stoqa9pt-reg2-cl4obelj.h5") #best model selected based on my reasoning for reg1 
		#self.interpreter = tf.keras.models.load_model("UnifiedModels/reg1-9w31h50k-reg2-cl4obelj.h5") #this is a bit worse than the stoqa9pt but the size is reduced from around 5.8k to 3.2k for reg1 
		#self.interpreter = tf.keras.models.load_model("UnifiedModels/reg1-4121t6zb-reg2-cl4obelj.h5") #this is the most trivial model with 0.2k params for reg1
		self.interpreter = tf.keras.models.load_model("UnifiedModels/reg1-stoqa9pt-reg2-hrchr82r-selected.h5") #the best selected model considering elbow method for both reg1 and reg2
			

		# Get model info
		self.getModelInputDetails()
		#self.getModelOutputDetails()

	def detectFaces(self, image):

		# Prepare image for inference
		input_tensor = self.prepareInputForInference(image)

		# Perform inference on the image
		loc_concat, cls_concat, pose_front, pose_back = self.inference(input_tensor)

		# Filter scores based on the detection scores
		scores, goodDetectionsIndices = self.filterDetections(cls_concat)
		boxes, keypoints = self.extractDetections(loc_concat, goodDetectionsIndices)
		detectionResults = self.filterWithNonMaxSupression(
			boxes, keypoints, scores, goodDetectionsIndices, pose_front, pose_back)

		# Update fps calculator
		self.updateFps()

		return detectionResults

	def updateFps(self):
		updateRate = 1
		self.frameCounter += 1

		# Every updateRate frames calculate the fps based on the ellapsed time
		if self.frameCounter == updateRate:
			timeNow = time.time()
			ellapsedTime = timeNow - self.timeLastPrediction

			self.fps = int(updateRate/(ellapsedTime+0.0001))
			self.frameCounter = 0
			self.timeLastPrediction = timeNow


	def draw_axis(self, img, yaw, pitch, roll, tdx, tdy, size=50, thickness=2):
		"""
		Draws three arrows indicating head rotations:
		- Red (Yaw): rotation around vertical axis
		- Green (Pitch): rotation around lateral axis
		- Blue (Roll): rotation around longitudinal axis
		yaw, pitch, roll are expected in degrees.
		tdx, tdy are the center point on the image.
		"""
		# Convert angles from degrees to radians
		yaw_r   = - math.radians(yaw)
		pitch_r = math.radians(pitch)
		roll_r  = math.radians(roll)

		# Center point
		cx, cy = int(tdx), int(tdy)

		# 1) Yaw arrow (red): rotate forward vector [0, -1] by yaw in image plane
		end_x = cx + size * math.sin(yaw_r)
		end_y = cy - size * math.cos(yaw_r)
		cv2.line(img, (cx, cy), (int(end_x), int(end_y)), (0, 0, 255), thickness)

		# 2) Pitch arrow (green): vertical nod, positive pitch lifts up
		end_x = cx
		end_y = cy - size * math.sin(pitch_r)
		cv2.line(img, (cx, cy), (int(end_x), int(end_y)), (0, 255, 0), thickness)

		# 3) Roll arrow (blue): rotate right vector [1, 0] by roll
		end_x = cx + size * math.cos(roll_r)
		end_y = cy + size * math.sin(roll_r)
		cv2.line(img, (cx, cy), (int(end_x), int(end_y)), (255, 0, 0), thickness)
  
  
	def drawDetections(self, img, results):

		poses = results.poses
		boundingBoxes = results.boxes
		# Base Y position for stable angle display
		y_base = self.img_height - 80
		keypoints = results.keypoints
		scores = results.scores

		# Add bounding boxes and keypoints
		for i, (boundingBox, keypoints, score) in enumerate(zip(boundingBoxes, keypoints, scores)):
			x1 = (self.img_width * boundingBox[0]).astype(int)
			x2 = (self.img_width * boundingBox[2]).astype(int)
			y1 = (self.img_height * boundingBox[1]).astype(int)
			y2 = (self.img_height * boundingBox[3]).astype(int)
			cv2.rectangle(img, (x1, y1), (x2, y2), (22, 22, 250), 2)
			cv2.putText(img, '{:.2f}'.format(score), (x1, y1 - 6)
								, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (22, 22, 250), 2)

			# Add keypoints for the current face
			for keypoint in keypoints:
				xKeypoint = (keypoint[0] * self.img_width).astype(int)
				yKeypoint = (keypoint[1] * self.img_height).astype(int)
				cv2.circle(img,(xKeypoint,yKeypoint), 4, (214, 202, 18), -1)

			# Draw 3D axis for head pose
			yaw, pitch, roll = poses[i]
			# compute center of bounding box
			tdx = (x1 + x2) / 2
			tdy = (y1 + y2) / 2
			#self.draw_axis(img, yaw, pitch, roll, tdx, tdy, size=int(min(x2-x1, y2-y1)/2), thickness=2)
			img = drawAxis_simo(img, [roll, yaw, pitch], tdx,tdy , size=int(min(x2-x1, y2-y1)/2))
			# Print angles with colored text
			# cv2.putText(img, f'Yaw: {yaw:.2f}', (10, y_base), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
			# cv2.putText(img, f'Pitch: {pitch:.2f}', (10, y_base + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
			# cv2.putText(img, f'Roll: {roll:.2f}', (10, y_base + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)
			cv2.putText(img, f'Yaw: {yaw:.2f}', (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
			cv2.putText(img, f'Pitch: {pitch:.2f}', (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			cv2.putText(img, f'Roll: {roll:.2f}', (x1, y2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

		# Add fps counter
		cv2.putText(img, f'FPS: {self.fps}', (40, 40)
						,cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 250, 22), 2)

		return img

	def getModelInputDetails(self):

		self.inputHeight = INPUT_FRONT
		self.inputWidth = INPUT_FRONT
			

			
		self.channels = 3
	
	#def getModelOutputDetails(self):
	#	self.output_details = self.interpreter.get_output_details()

	def generateAnchors(self):
	
		# Options to generate anchors for SSD object detection models.
		ssd_anchors_calculator_options = SsdAnchorsCalculatorOptions(input_size_width=128, input_size_height=128, min_scale=0.1484375, max_scale=0.75
				, anchor_offset_x=0.5, anchor_offset_y=0.5, num_layers=4
				, feature_map_width=[], feature_map_height=[]
				, strides=[8, 16, 16, 16], aspect_ratios=[1.0]
				, reduce_boxes_in_lowest_layer=False, interpolated_scale_aspect_ratio=1.0
				, fixed_anchor_size=True)



		self.anchors = gen_anchors(ssd_anchors_calculator_options)

	def prepareInputForInference(self, image):
		
		img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		self.img_height, self.img_width, self.img_channels = img.shape

		# Input values should be from -1 to 1 with a size of 128 x 128 pixels for the fornt model
		# and 256 x 256 pixels for the back model
		img = img / 255.0
		img_resized = tf.image.resize(img, [self.inputHeight,self.inputWidth], 
									method='bicubic', preserve_aspect_ratio=False)
		#img_resized = tf.image.resize_with_crop_or_pad (img, self.inputHeight, self.inputWidth )
		#boxes = tf.constant([0, 0.21, 1, 0.79])
		#box_indices = tf.constant([0])
		#CROP_SIZE = tf.constant([self.inputHeight, self.inputWidth])
		#img_resized = tf.image.crop_and_resize(img[None,:,:,:], np.asarray([[0, 0.21, 1, 0.79]]), [0], [self.inputHeight, self.inputWidth])
		img_input = img_resized.numpy()
		img_input = (img_input - 0.5) / 0.5

		# Adjust matrix dimenstions
		reshape_img = img_input.reshape(1,self.inputHeight,self.inputWidth,self.channels)
		#tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)

		return reshape_img

	def inference(self, input_tensor):
		raw_outputs = self.interpreter(input_tensor)
		cls_front = np.squeeze(raw_outputs[0])
		cls_back = np.squeeze(raw_outputs[1])
		loc_front = np.squeeze(raw_outputs[2])
		loc_back = np.squeeze(raw_outputs[3])
		pose_front = np.squeeze(raw_outputs[4])
		pose_back = np.squeeze(raw_outputs[5])
		# concatenate location and classification outputs
		loc_concat = np.concatenate((loc_front, loc_back))
		cls_concat = np.concatenate((cls_front, cls_back))
		return loc_concat, cls_concat, pose_front, pose_back

	def extractDetections(self, output0, goodDetectionsIndices):

		numGoodDetections = goodDetectionsIndices.shape[0]

		keypoints = np.zeros((numGoodDetections, KEY_POINT_SIZE, 2))
		boxes = np.zeros((numGoodDetections, 4))
		for idx, detectionIdx in enumerate(goodDetectionsIndices):
			anchor = self.anchors[detectionIdx]

			sx = output0[detectionIdx, 0]
			sy = output0[detectionIdx, 1]
			w = output0[detectionIdx, 2]
			h = output0[detectionIdx, 3]

			cx = sx + anchor.x_center * self.inputWidth
			cy = sy + anchor.y_center * self.inputHeight

			cx /= self.inputWidth
			cy /= self.inputHeight
			w /= self.inputWidth
			h /= self.inputHeight

			for j in range(KEY_POINT_SIZE):
				lx = output0[detectionIdx, 4 + (2 * j) + 0]
				ly = output0[detectionIdx, 4 + (2 * j) + 1]
				lx += anchor.x_center * self.inputWidth
				ly += anchor.y_center * self.inputHeight
				lx /= self.inputWidth
				ly /= self.inputHeight
				keypoints[idx,j,:] = np.array([lx, ly])

			boxes[idx,:] = np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5])

		return boxes, keypoints

	def filterDetections(self, output1):

		# Filter based on the score threshold before applying sigmoid function
		goodDetections = np.where(output1 > self.sigmoidScoreThreshold)[0]

		# Convert scores back from sigmoid values
		scores = 1.0 /(1.0 + np.exp(-output1[goodDetections]))

		return scores, goodDetections

	def filterWithNonMaxSupression(self, boxes, keypoints, scores,
									 detection_indices, pose_front, pose_back):
		# Filter based on non max suppression
		selected_indices = tf.image.non_max_suppression(boxes, scores, MAX_FACE_NUM, self.iouThreshold).numpy()
		filtered_boxes = tf.gather(boxes, selected_indices).numpy()
		filtered_keypoints = tf.gather(keypoints, selected_indices).numpy()
		filtered_scores = tf.gather(scores, selected_indices).numpy()
		# If no faces detected, return empty results with zero poses
		if selected_indices.size == 0:
			empty_poses = np.zeros((0, 3), dtype=np.float32)
			return Results(filtered_boxes, filtered_keypoints, filtered_scores, empty_poses)
		# compute poses per selected detection
		poses = []
		for det_idx in detection_indices[selected_indices]:
			if det_idx < 512:
				spatial_idx = det_idx // 2
				row = spatial_idx // 16
				col = spatial_idx % 16
				poses.append(pose_front[row, col])
			else:
				idx = det_idx - 512
				spatial_idx = idx // 6
				row = spatial_idx // 8
				col = spatial_idx % 8
				poses.append(pose_back[row, col])
		filtered_poses = np.stack(poses, axis=0)
		detectionResults = Results(filtered_boxes, filtered_keypoints,
									filtered_scores, filtered_poses)
		return detectionResults

class Results:
	def __init__(self, boxes, keypoints, scores, poses):
		self.boxes = boxes
		self.keypoints = keypoints
		self.scores = scores
		self.poses = poses

if __name__ == "__main__":
    # Initialize detector for front camera images
    detector = blazeFaceDetector()
    # Open default camera (0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open front camera")
        exit(1)

    # Flag to enable video recording
    record_video = True
    video_writer = None
    alpha = 0.15

    # Flag to toggle EMA usage
    use_ema = True

    # Initialize EMA filters for yaw, pitch, roll, bounding boxes, and keypoints if EMA is enabled
    if use_ema:
        yaw_filter = EMAFilter(alpha)
        pitch_filter = EMAFilter(alpha)
        roll_filter = EMAFilter(alpha)
        bbox_filters = [EMAFilter(alpha) for _ in range(4)]  # 4 values for bounding box (x1, y1, x2, y2)
        keypoint_filters = [[EMAFilter(alpha) for _ in range(2)] for _ in range(KEY_POINT_SIZE)]  # x, y for each keypoint

    try:
        while True:
            ret, frame = cap.read()
            # Crop to square frame
            h, w = frame.shape[:2]
            side = min(h, w)
            x = (w - side) // 2
            y = (h - side) // 2
            frame = frame[y:y+side, x:x+side]
            if not ret:
                print("Error: Failed to read frame")
                break
            # Run face detection and pose estimation
            results = detector.detectFaces(frame)

            if use_ema:
                # Apply EMA to smooth yaw, pitch, roll, bounding boxes, and keypoints
                for i, (pose, bbox, keypoints) in enumerate(zip(results.poses, results.boxes, results.keypoints)):
                    # Smooth yaw, pitch, and roll
                    raw_yaw, raw_pitch, raw_roll = pose
                    smooth_yaw = yaw_filter.update(raw_yaw)
                    smooth_pitch = pitch_filter.update(raw_pitch)
                    smooth_roll = roll_filter.update(raw_roll)
                    results.poses[i] = [smooth_yaw, smooth_pitch, smooth_roll]

                    # Smooth bounding box
                    for j in range(4):
                        bbox[j] = bbox_filters[j].update(bbox[j])
                    results.boxes[i] = bbox

                    # Smooth keypoints
                    for k, (x, y) in enumerate(keypoints):
                        keypoints[k][0] = keypoint_filters[k][0].update(x)
                        keypoints[k][1] = keypoint_filters[k][1].update(y)
                    results.keypoints[i] = keypoints

            # Draw the detections and pose axes
            output_frame = detector.drawDetections(frame, results)

            # Initialize video writer if recording is enabled
            if record_video and video_writer is None:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
                video_writer = cv2.VideoWriter(f"{timestamp}.mp4", fourcc, 20.0, (output_frame.shape[1], output_frame.shape[0]))

            # Write frame to video file if recording
            if record_video and video_writer is not None:
                video_writer.write(output_frame)

            # Display the result
            cv2.imshow("BlazeFace Head Pose Estimation", output_frame)
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()