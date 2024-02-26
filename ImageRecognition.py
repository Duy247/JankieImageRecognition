import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QLineEdit, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import requests
from io import BytesIO
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import requests
import cv2
from matplotlib import pyplot as plt

class ImageRecognitionApp(QMainWindow):
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.setWindowTitle('Image Recognition App')
		self.setGeometry(300, 300, 600, 500)

		# Main layout
		self.central_widget = QWidget()
		self.setCentralWidget(self.central_widget)
		self.layout = QVBoxLayout(self.central_widget)

		# Label for URL input
		self.url_input_label = QLabel('Enter Image URL:', self)
		self.layout.addWidget(self.url_input_label)

		# Text field for URL input
		self.url_input_field = QLineEdit(self)
		self.layout.addWidget(self.url_input_field)

		# Button to load image from URL
		self.btn_load_image = QPushButton('Load Image from URL', self)
		self.btn_load_image.clicked.connect(self.loadImageFromURL)
		self.layout.addWidget(self.btn_load_image)

		# Label to display the image
		self.image_label = QLabel(self)
		self.image_label.setAlignment(Qt.AlignCenter)
		self.image_label.setText('No image loaded.')
		self.image_label.setFixedSize(1920, 1080)
		self.layout.addWidget(self.image_label)

		# Button to recognize image (disabled until an image is loaded)
		self.btn_recognize = QPushButton('Recognize Image', self)
		self.btn_recognize.setEnabled(False)
		self.url = self.url_input_field.text()
		self.btn_recognize.clicked.connect(self.recognizeImage)
		self.layout.addWidget(self.btn_recognize)

		self.current_image_data = None
		self.image_data = None

	def loadImageFromURL(self):
		url = self.url_input_field.text()
		try:
			response = requests.get(url)
			response.raise_for_status()
			self.image_data = BytesIO(response.content)
			pixmap = QPixmap()
			pixmap.loadFromData(self.image_data.getvalue())
			self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
			self.btn_recognize.setEnabled(True)
			self.current_image_data = self.image_data
		except requests.RequestException as e:
			QMessageBox.critical(self, 'Error', f'Failed to load image from URL.\n{e}')
	def load_image_from_url(self,url):
		response = requests.get(url)
		image = Image.open(BytesIO(response.content))
		image = image.convert('RGB')
		image_np = np.array(image)
		return image_np
	def detect_objects(self,image_np):
		detector = tf.saved_model.load("model2").signatures['default']
		# Convert the image to a tensor
		input_tensor = tf.convert_to_tensor(image_np)
		input_tensor = input_tensor[tf.newaxis, ...]
	
		# Assuming 'input_tensor' is your uint8 tensor with shape (1, 1067, 1600, 3)
		input_tensor = tf.cast(input_tensor, dtype=tf.float32)  # Cast to float32
		input_tensor = input_tensor / 255.0  # Normalize to range [0, 1]
	
		# Now you can pass this normalized tensor to your function/model
		# Run object detection
		results = detector(input_tensor)
	
		# Convert results to numpy arrays
		result = {key:value.numpy() for key,value in results.items()}
	
		# Extract boxes, scores, and classes
		boxes = result['detection_boxes']
		scores = result['detection_scores']
		classes = result['detection_class_entities']
	
		return boxes, scores, classes
		print(boxes, scores, classes)

	def draw_boxes_on_image(self, image_np, boxes, classes, scores, max_boxes=5):
		# Draw bounding boxes and labels on the image
		for i in range(min(max_boxes, len(boxes))):
			box = boxes[i]
			class_name = classes[i]
			score = scores[i]

			# Convert box coordinates to integers
			ymin, xmin, ymax, xmax = box
			(left, right, top, bottom) = (xmin * image_np.shape[1], xmax * image_np.shape[1],
										  ymin * image_np.shape[0], ymax * image_np.shape[0])
			left, right, top, bottom = int(left), int(right), int(top), int(bottom)

			# Draw the bounding box
			cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)

			# Draw the label
			label = f'{class_name}: {score:.2f}'
			cv2.putText(image_np, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		return image_np
	    
	def recognizeImage(self):
		image_url = self.url_input_field.text()
		if self.current_image_data:
			# Example of using the above functions
			image_np = self.load_image_from_url(image_url)
			boxes, scores, classes = self.detect_objects(image_np)
			
			# Sort the detections by scores in descending order and select the top 5
			top_indices = np.argsort(scores)[::-1][:5]
			top_boxes = [boxes[i] for i in top_indices]
			top_scores = [scores[i] for i in top_indices]
			top_classes = [classes[i] for i in top_indices]
			
			# Draw the boxes on the image
			image_with_boxes = self.draw_boxes_on_image(image_np.copy(), top_boxes, top_classes, top_scores)
			
			# Convert color from BGR to RGB
			image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
			
			q_image = QImage(image_with_boxes.data, image_with_boxes.shape[1], image_with_boxes.shape[0], QImage.Format_RGB888)
			pixmap = QPixmap.fromImage(q_image)
			self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
        
			self.btn_recognize.setEnabled(True)
			self.current_image_data = self.image_data
		else:
			QMessageBox.warning(self, 'Warning', 'No image loaded.')

if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = ImageRecognitionApp()
	ex.show()
	sys.exit(app.exec_())