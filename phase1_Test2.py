import cv2
import dlib
import numpy as np 
import matplotlib.pyplot as plt
import sys


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords


def rect_to_np(rects):
	rect = []
	for i in rects:
		rect.append([int(i.left()),int(i.top()),int(i.right()),int(i.bottom())])
	return rect

def point_to_id(pt,fp):
	for i in range(0,fp.shape[0]):
		if (list(fp[i]) == list(pt)):
			return i

def face_triangles(triangles,fp1,c):
	
	extLeft =	tuple(c[c[:, :, 0].argmin()][0])
	extRight = 	tuple(c[c[:, :, 0].argmax()][0])
	extTop = 	tuple(c[c[:, :, 1].argmin()][0])
	extBot = 	tuple(c[c[:, :, 1].argmax()][0])
	
	delete_idx = []
	for i in range(0,len(triangles)):
		if np.any((triangles[i,0] < extLeft[0]) or (triangles[i,2] < extLeft[0]) or (triangles[i,4] < extLeft[0])):
			delete_idx.append(i)
		if np.any((triangles[i,0] > extRight[0]) or (triangles[i,2] > extRight[0]) or (triangles[i,4] > extRight[0])):
			delete_idx.append(i)
		if np.any((triangles[i,1] < extTop[1]) or (triangles[i,3] < extTop[1]) or (triangles[i,5] < extTop[1])):
			delete_idx.append(i)
		if np.any((triangles[i,1] > extBot[1]) or (triangles[i,3] > extBot[1]) or (triangles[i,5] > extBot[1])):
			delete_idx.append(i)
	delete_idx = np.unique(delete_idx)
	triangles = np.delete(triangles,delete_idx,axis = 0)
	return triangles

def get_points_in_triangles(img1, pt1,pt2,pt3):
	mask = np.zeros_like(img1)
	pts = np.array([pt1,pt2,pt3])
	mask = cv2.fillPoly(mask,[pts],255)
	mask_pts = np.argwhere(mask>0)
	mask_pts = np.flip(mask_pts,axis=1)
	return mask_pts

def main():

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
	
	cap = cv2.VideoCapture('./Data/TestSet_P2/Test2.mp4')
	# out = cv2.VideoWriter('Test2.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (854,480))


	while True:
		ret,img1 = cap.read()
		if ret==False:
			cap.release()
			cv2.destroyAllWindows()
			break

		img1_copy = np.copy(img1)

		img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
		rects1 = detector(img1_gray,1)
		# print("Rects1 :",rects1)
		rect1 = rect_to_np(rects1)

		if len(rect1) < 2  :
			# out.write(img1)
			continue
		pts1 = predictor(img1_gray,rects1[1])
		fp1 = shape_to_np(pts1)

		pts2 = predictor(img1_gray,rects1[0])
		fp2 = shape_to_np(pts2)
		

		src1 = np.zeros_like(img1)
		src2 = np.zeros_like(img1)
		poisson_mask1 = np.zeros_like(img1_gray)
		poisson_mask2 = np.zeros_like(img1_gray)

		
		hull1 = cv2.convexHull(fp2)
		face_rect2 = cv2.boundingRect(hull1)
		subdiv2 = cv2.Subdiv2D(face_rect2)
		subdiv2.insert(tuple(fp2))
		triangles2 = subdiv2.getTriangleList()
		triangles2 = triangles2.astype(int)
		triangles2 = face_triangles(triangles2,fp2,hull1)

		hull = cv2.convexHull(fp1)
		face_rect1 = cv2.boundingRect(hull)
		subdiv1 = cv2.Subdiv2D(face_rect1)
		subdiv1.insert(tuple(fp1))
		triangles1 = subdiv1.getTriangleList()
		triangles1 = triangles1.astype(int)
		triangles1 = face_triangles(triangles1,fp1,hull)

		for t in triangles1:
			# print("Triangle",t)

			pt1 = [t[0], t[1]]
			pt2 = [t[2], t[3]]
			pt3 = [t[4], t[5]]

			points = get_points_in_triangles(img1_gray,pt1,pt2,pt3)
			one_c = np.expand_dims(np.ones(points.shape[0],dtype=np.int64),axis=1)
			points = np.append(points,one_c,axis=1)
			# print("Points :",points)
			Bd = np.array([[t[0],t[2],t[4]],[t[1],t[3],t[5]],[1,1,1]],dtype=np.int64)
			bary_pt1 = np.matmul(np.linalg.inv(Bd),points.transpose().astype(np.int64))
			# print("Points :",points.shape)
			points = points[bary_pt1.min(axis=0)>=0,:]
			bary_pt1 = bary_pt1[:,bary_pt1.min(axis=0)>=0]
			# print("Bary1 ",bary_pt1.shape)

			tri_id1 = [point_to_id(pt1,fp1),point_to_id(pt2,fp1),point_to_id(pt3,fp1)]
			pts2 =  np.array([fp2[tri_id1[0]],fp2[tri_id1[1]],fp2[tri_id1[2]]])

			Ad = np.array([[fp2[tri_id1[0]][0],fp2[tri_id1[1]][0],fp2[tri_id1[2]][0]],
				[fp2[tri_id1[0]][1],fp2[tri_id1[1]][1],fp2[tri_id1[2]][1]],
				[1,1,1]])
			bary_pt2 = np.matmul(Ad,bary_pt1)
			bary_pt2 = (bary_pt2/bary_pt2[2]).astype(np.int64)


			cv2.fillPoly(poisson_mask1,[np.array([pt1,pt2,pt3])],255)
			# # Optimizable
			for i in range(0,points.shape[0]):
				src1[points[i,1],points[i,0]] = img1[bary_pt2.transpose()[i,1],bary_pt2.transpose()[i,0]]

		points = []
		for t in triangles2:
			# print("Triangle",t)

			pt1 = [t[0], t[1]]
			pt2 = [t[2], t[3]]
			pt3 = [t[4], t[5]]
		
			points = get_points_in_triangles(img1_gray,pt1,pt2,pt3)
			one_c = np.expand_dims(np.ones(points.shape[0],dtype=np.int64),axis=1)
			points = np.append(points,one_c,axis=1)
			# print("Points :",points)
			Bd = np.array([[t[0],t[2],t[4]],[t[1],t[3],t[5]],[1,1,1]],dtype=np.int64)
			bary_pt1 = np.matmul(np.linalg.inv(Bd),points.transpose().astype(np.int64))
			# print("Points :",points.shape)
			points = points[bary_pt1.min(axis=0)>=0,:]
			bary_pt1 = bary_pt1[:,bary_pt1.min(axis=0)>=0]
			# print("Bary1 ",bary_pt1.shape)

			tri_id1 = [point_to_id(pt1,fp2),point_to_id(pt2,fp2),point_to_id(pt3,fp2)]
			pts2 =  np.array([fp2[tri_id1[0]],fp2[tri_id1[1]],fp2[tri_id1[2]]])

			Ad = np.array([[fp1[tri_id1[0]][0],fp1[tri_id1[1]][0],fp1[tri_id1[2]][0]],
				[fp1[tri_id1[0]][1],fp1[tri_id1[1]][1],fp1[tri_id1[2]][1]],
				[1,1,1]])
			bary_pt2 = np.matmul(Ad,bary_pt1)
			bary_pt2 = (bary_pt2/bary_pt2[2]).astype(np.int64)


			cv2.fillPoly(poisson_mask2,[np.array([pt1,pt2,pt3])],255)
			# # Optimizable
			for i in range(0,points.shape[0]):
				src2[points[i,1],points[i,0]] = img1[bary_pt2.transpose()[i,1],bary_pt2.transpose()[i,0]]


		mixed_clone = cv2.seamlessClone(src1,img1, poisson_mask1, (int((2*face_rect1[0]+face_rect1[2])/2),int((2*face_rect1[1]+face_rect1[3])/2)), cv2.MIXED_CLONE)
		mixed_clone = cv2.seamlessClone(src2,mixed_clone, poisson_mask2, (int((2*face_rect2[0]+face_rect2[2])/2),int((2*face_rect2[1]+face_rect2[3])/2)), cv2.MIXED_CLONE)

		# out.write(mixed_clone)
		cv2.waitKey()
	

if __name__ == '__main__':
	main()
