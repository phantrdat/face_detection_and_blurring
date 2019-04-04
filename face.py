# Import the necessary libraries
import multiprocessing
import cv2
import argparse
import os
import csv
BLUR_DIR  = 'blur/'
LOCATE_DIR = 'location/'
INFO_DIR = 'info/'


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

def detect_faces(cascade, test_image, scaleFactor = 1.01):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    
    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors = 5)
    # print('Faces found: ', len(faces_rect))

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image_copy, faces_rect

def blur_face(cascade, image, faces_rect, scaleFactor = 1.01, ):
	result_image = image.copy()
	for (x, y, w, h) in faces_rect:
		sub_face = image[y:y + h, x:x + w]
		# apply a gaussian blur on this new recangle image
		sub_face = cv2.GaussianBlur(sub_face,(51, 51), 75)
		# merge this blurry rectangle to our final image
		result_image[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
	return result_image, faces_rect

def detect_and_blur(img, input_path, output_path):
	print(img)
	name = img[:img.rfind('.')]
	image = cv2.imread(input_path + img)
	# call the function to detect faces
	faces, faces_rect  = detect_faces(haar_cascade_face, image)
	blur_faces, faces_rect = blur_face(haar_cascade_face, image, faces_rect)
	# convert to RGB and display image
	convertToRGB(faces)
	convertToRGB(blur_faces)
	# Saving the final image
	cv2.imwrite(output_path + LOCATE_DIR + img.strip('.jpg') + '_locate.jpg', faces)
	cv2.imwrite(output_path + BLUR_DIR + img.strip('.jpg') + '_blur.jpg', blur_faces)
	# Saving infomation
	# with open(output_path + INFO_DIR + name + '.csv', 'w', newline='', encoding="utf-8") as csvfile:
	# 	fieldnames = ['id', 'left_top_x', 'left_top_y', 'width', 'height']
	# 	writer = csv.writer(csvfile)
	# 	writer.writerow(fieldnames)
	# 	for (idx, loc) in enumerate(faces_rect):
	# 		x, y, w, h = loc
	# 		writer.writerow(['id_' + str(idx), str(x), str(y), str(w), str(h)])

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--i', help='Input Path')
	parser.add_argument('--o', help='Output Path')
	parser.add_argument('--th', help='Number of Threads')
	args = parser.parse_args()


	# parse --i flag to get input path
	input_path = args.i
	if (input_path[len(input_path)-1] != '/'):
		input_path = input_path + "/"

	# parse --o flag to get output path
	output_path = args.o
	if (output_path[len(output_path)-1] != '/'):
		output_path = output_path + "/"

	# parse --o flag to get output path
	if (args.th==None):
		threads_num = 1

	else:
		threads_num = int(args.th)
	if (os.path.isdir(output_path+BLUR_DIR)==False):
		os.mkdir(output_path+BLUR_DIR,0o755)
	if (os.path.isdir(output_path+LOCATE_DIR)==False):
		os.mkdir(output_path+LOCATE_DIR,0o755)
	if (os.path.isdir(output_path+INFO_DIR)==False):
		os.mkdir(output_path+INFO_DIR,0o755)
	img_list = os.listdir(args.i)
	while (len(img_list)!=0):
		if len(img_list)<=threads_num:
			tmp_list = img_list
			img_list = []
		else:
			tmp_list = img_list[:threads_num]
			img_list = img_list[threads_num:]
		jobs = []
		for img in tmp_list:
			# detect_and_blur(img, input_path, output_path)
			process = multiprocessing.Process(target=detect_and_blur, args=(img, input_path, output_path,))
			jobs.append(process)
		for t in jobs:
			t.start()
		for t in jobs:
			t.join()
if __name__ == '__main__':
	import time
	start = time.time()
	main()
	print ("\n\n\nExecute in: " + str(time.time()-start) + " seconds\n\n\n")

# python face.py --i ./images --o ./result --th 4