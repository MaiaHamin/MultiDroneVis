import csv
from PIL import Image, ImageDraw
import sys
import numpy as np
import cv2
import os
from pathlib import Path
import math

def convert_gps_to_box(gps, n_boxes=9):
    """Convert gps coordinates to one-hot label of box index.

    Hard-coded currently to have 9 total boxes to cover space of
    coordinates.
    Args:
        gps: (x,y) coordinates
    Returns:
        one-hot-vector indicating which box (x,y) is in

    x \in [0, 0.98]; y \in [0.58, 0.749] (y range smaller because
    it corresponds to vertical area where water is)
    This area is divided into 9 boxes
    """
    x, y = gps
    if x > 0.98:
        x = 0.98
    if y > 0.749:
        y = 0.749
    if y < 0.58:
        y = 0.58

    n_each_coord = int(np.sqrt(n_boxes))
    x_box_idx = int(x * 100) // int(100 / n_each_coord)
    y_box_idx = int((y - 0.58) * 100) // int(19 / n_each_coord)

    box_idx = n_each_coord * y_box_idx + x_box_idx
    if box_idx > n_boxes - 1 or box_idx < 0:
        raise ValueError(
            "Box Index {} is incorrect; gps: {}".format(box_idx, gps))

    return box_idx



def project(viewProjectionMatrix, orig_x, orig_y, orig_z):
	testx = viewProjectionMatrix[0][0] * orig_x + viewProjectionMatrix[1][0] * orig_y + viewProjectionMatrix[2][0] * orig_z + viewProjectionMatrix[3][0]
	testy = viewProjectionMatrix[0][1] * orig_x + viewProjectionMatrix[1][1] * orig_y + viewProjectionMatrix[2][1] * orig_z + viewProjectionMatrix[3][1]
	testw = viewProjectionMatrix[0][3] * orig_x + viewProjectionMatrix[1][3] * orig_y + viewProjectionMatrix[2][3] * orig_z + viewProjectionMatrix[3][3]

	pred_x = ((1 + (testx / testw)) / 2.)
	pred_y = ((1 - (testy / testw)) / 2.)
	return pred_x, pred_y

def draw_arc(im, draw):
	zero_x, zero_y = project(viewProjectionMatrix, camera_x, camera_y, camera_z)
	left_corner_x, left_corner_y = project(viewProjectionMatrix, xLim[0], yLim[1])
	right_corner_x, right_corner_y = project(viewProjectionMatrix, xLim[2], yLim[1])

	draw.line(((zero_x * width, zero_y - y_start * height), (left_corner_x*width, height - left_corner_y*height)), fill="red")
	draw.line(((zero_x * width, zero_y - y_start * height), (right_corner_x*width, height - right_corner_y*height)), fill="red")

	im = Image.open("HighDrone/HighresScreenshot" + str(count).zfill(5) + ".png")
	draw = ImageDraw.Draw(im)

def draw_target(im, draw, real_x, real_y):
	width = im.size[0]
	height = im.size[1]

	for interval in [0, 1, 2, 3, -1, -2, -3]:
		target = Image.open("target.png")
		target2 = target.convert("RGBA")
		bound_elipse = [int(real_x*width - 16), int(real_y*height - 16), int(real_x*width + 16), int(real_y*height + 16)]
		im.paste(target2, bound_elipse, target2)

def draw_point(im, draw, x, y, color="red"):
	width = im.size[0]
	height = im.size[1]
	print(str(x) + ", " + str(y))

	draw.ellipse(((x * width, y * height), (x * width + 4, y * height + 4)), fill=color)




def draw_grid(im, draw, viewProjectionMatrix, color="black"):
	width = im.size[0]
	height = im.size[1]

	must_project = False
	# xLim = [240, 2830]
	# yLim = [-740, 1190]
	# zLim = [100, 100]

	xLim = [0,0.98]
	yLim = [0.58, 0.749]

	xw = xLim[1] - xLim[0]
	yw = yLim[1] - yLim[0]

	xpoints = []
	ypoints = []

	for interval in [0., 1., 2., 3.]:
		xpoints.append(xLim[0] + interval * xw / 3.)
		ypoints.append(yLim[0] + interval * yw / 3.)
	if must_project:


		for x in xpoints:
			x_start, y_start = project(viewProjectionMatrix, x, ypoints[0], 775)
			x_end, y_end = project(viewProjectionMatrix, x, ypoints[3], 775)
			print("from (" + str(x_start) + ", " + str(y_start) + ") to ( " + str(x_end) + ", " + str(y_end) + ")")
			draw.line(((x_start * width, height - y_start * height), (x_end*width, height - y_end*height)), fill="red")
			draw.point((x_start * width, height - y_start * height), fill="black")
			draw.point((x_end * width, height - y_end * height), fill="black")

		for y in ypoints:
			x_start, y_start = project(viewProjectionMatrix, xpoints[0], y, 775)
			x_end, y_end = project(viewProjectionMatrix, xpoints[3], y, 775)
			print("from (" + str(x_start) + ", " + str(y_start) + ") to ( " + str(x_end) + ", " + str(y_end) + ")")
			draw.line(((x_start * width, height - y_start * height), (x_end*width, height - y_end*height)), fill="red")
			draw.point((x_start * width, height - y_start * height), fill="black")
			draw.point((x_end * width, height - y_end * height), fill="black")

	else:
		for i in range(4):
			draw.line(((xpoints[i]*width, ypoints[0] * height), (xpoints[i]*width,  ypoints[3]*height)), fill=color)
		for i in range(4):
			draw.line(((xpoints[0], ypoints[i] * height), (xpoints[3]*width, ypoints[i]*height)), fill=color)


def draw_box(im, draw, ind, color):
	width = im.size[0]
	height = im.size[1]

	must_project = False
	# xLim = [240, 2830]
	# yLim = [-740, 1190]
	# zLim = [100, 100]

	xLim = [0,0.98]
	yLim = [0.58, 0.749]

	xw = xLim[1] - xLim[0]
	yw = yLim[1] - yLim[0]

	xpoints = []
	ypoints = []

	for interval in [0., 1., 2., 3.]:
		xpoints.append(xLim[0] + interval * xw / 3.)
		ypoints.append(yLim[0] + interval * yw / 3.)

	xind = 3 - int(ind / 3)
	yind = 3 - int(ind % 3)

	draw.rectangle(((xpoints[xind]*width,ypoints[yind] * height), (xpoints[xind - 1]*width, ypoints[yind - 1]*height)), fill=color)




def make_vid(image_folder):
		video_name = 'video.avi'
		images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
		frame = cv2.imread(os.path.join(image_folder, images[0]))
		height, width, layers = frame.shape
		video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 1, (width,height))
		for image in images:
		    video.write(cv2.imread(os.path.join(image_folder, image)))
		cv2.destroyAllWindows()
		video.release()


#with open('TrainingPatternsTransformedMatrix200.csv', "rt") as csvfile:
#with open("DroneVis/tformeddemocoords.csv", "rt") as csvfile:
def main():
	preds = open("ErrorVis/preds_single_inv_cor.csv", "rt")
	predreader = csv.reader(preds)
	indata = open("ErrorVis/tformedinverted.csv", "rt")
	datareader = csv.reader(indata)

	next(datareader)
	count = 0
	must_project = False
	if must_project:
		camera_x = float(row[3])
		camera_y = float(row[4])
		camera_z = float(row[5])

		o_x = float(row[0])
		o_y = float(row[1])
		o_z = float(row[2])
		orig_coords = np.array([o_x, o_y, o_z, 1.], np.float64)

		rel_x = float(row[6])
		rel_y = float(row[7])
		rel_z = float(row[8])

		t_x = camera_x - rel_x
		t_y = camera_y - rel_y
		t_z = 525

		matrix_str = "[0 0 0 1] [1 0 0 0] [0 1.77778 0 0] [112.5 -776.171 10 544]"
		matrix_rows = matrix_str.split("]")
		matrix_nums = []
		for r in matrix_rows:
			str_nums = r.strip()[1:]
			str_nums = str_nums.split()
			float_nums = []
			for num in str_nums:
				float_nums.append(float(num))
			matrix_nums.append(float_nums)
		viewProjectionMatrix = np.array(matrix_nums[:4], np.float64)
		calc_x, calc_y = project(viewProjectionMatrix, o_x, o_y, o_z)
		rel_cal_x, rel_cal_y = project(viewProjectionMatrix, t_x, t_y, t_z)
		draw_point(im, draw, rel_cal_x, rel_cal_y)
		draw_point(im, draw, calc_x, calc_y)
		im.save("ModTrans/HighresScreenshot" + str(count).zfill(5) + ".png", quality=100)
		count += 1
	else:
		count = 0
		subcount = 0
		for row in list(datareader)[:25]:
			im = Image.open("ErrorVis/images/HighresScreenshot" + str(count).zfill(5) + ".png")
			draw = ImageDraw.Draw(im)
			obj_x = float(row[2])
			obj_y = float(row[3])
			copy_x = float(row[8])
			copy_y = float(row[9])
			obj_box = convert_gps_to_box((obj_x, obj_y))
			copy_box = convert_gps_to_box((copy_x, copy_y))
			draw_grid(im, draw, [], "black")
			#draw_box(im, draw, obj_box, "red")
			#draw_box(im, draw, copy_box, "yellow")


			bpred = next(predreader)
			print(bpred)
			draw_box(im, draw, int(bpred[1]), "green")

			draw_point(im, draw, obj_x, obj_y, "red")
			draw_point(im, draw, copy_x, copy_y, "yellow")

			im.save("ErrorVis/pred_images/" + str(count).zfill(5) + "_" + str(count) + ".png", quality=100)
			count += 1
	image_folder = 'ErrorVis/pred_images'
	make_vid(image_folder)


main()
