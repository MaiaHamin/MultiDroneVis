import csv
import numpy as np

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
vpm = np.array(matrix_nums[:4], np.float64)

def project(viewProjectionMatrix, orig_x, orig_y, orig_z):
	testx = viewProjectionMatrix[0][0] * orig_x + viewProjectionMatrix[1][0] * orig_y + viewProjectionMatrix[2][0] * orig_z + viewProjectionMatrix[3][0]
	testy = viewProjectionMatrix[0][1] * orig_x + viewProjectionMatrix[1][1] * orig_y + viewProjectionMatrix[2][1] * orig_z + viewProjectionMatrix[3][1]
	testw = viewProjectionMatrix[0][3] * orig_x + viewProjectionMatrix[1][3] * orig_y + viewProjectionMatrix[2][3] * orig_z + viewProjectionMatrix[3][3]

	pred_x = ((1 + (testx / testw)) / 2.)
	pred_y = ((1 - (testy / testw)) / 2.)
	return pred_x, pred_y


outfile = open("tformeddemocoords.csv", "w")
with open("democoords.csv", "rt") as csvfile:
	reader = csv.reader(csvfile)
	writer = csv.writer(outfile)
	readlist = list(reader)
	writer.writerow(readlist[0])
	for row in readlist[1:]:
		print(row)
		x, y = project(vpm, float(row[2]), float(row[3]), float(row[4]))
		row[2] = x
		row[3] = y
		row[4] = ""
		writer.writerow(row)
