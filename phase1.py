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

def face_triangles(triangles,fp1):
	delete_idx = []
	for i in range(0,len(triangles)):
		if np.any(triangles[i]<0)==True:
			delete_idx.append(i)
		if np.any((triangles[i,0] > 711) or (triangles[i,2] > 711) or (triangles[i,4] > 711)):
			delete_idx.append(i)
		if np.any((triangles[i,1] > 1050) or (triangles[i,3] > 1050) or (triangles[i,5] > 1050)):
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

	img1 = cv2.imread("./Data/bradley_cooper.jpg")
	img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

	rects1 = detector(img1_gray,1)
	pts1 = predictor(img1_gray,rects1[0])
	
	rect1 = rect_to_np(rects1)
	print("Rectangle 1: ",rect1)
	fp1 = shape_to_np(pts1)

	img2 = cv2.imread("./Data/jim_carrey.jpg")
	img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

	rects2 = detector(img2_gray,1)
	pts2 = predictor(img2_gray,rects2[0])
	
	rect2 = rect_to_np(rects2)
	print("Rectangle 2: ",rect2)
	fp2 = shape_to_np(pts2)

	hull1 = cv2.convexHull(fp1)

	
	# Make mask
	mask1 = np.zeros([img1_gray.shape[0],img1_gray.shape[1]], dtype=np.uint8)
	mask1=cv2.fillConvexPoly(mask1,hull1,255)
	mask_pts = np.argwhere(mask1>200)

	face_1 = cv2.bitwise_and(img1,img1,mask=mask1) 
	
	face_rect1 = cv2.boundingRect(hull1)

	subdiv1 = cv2.Subdiv2D(face_rect1)
	subdiv1.insert(tuple(fp1))
	triangles1 = subdiv1.getTriangleList()
	triangles1 = triangles1.astype(int)
	print("Image shape :",mask1.shape)
	triangles1 = face_triangles(triangles1,fp1)

	for t in triangles1:
		print("Triangle",t)
		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])
		# cv2.line(img1, pt1, pt2, (0, 0, 255), 1)
		# cv2.line(img1, pt2, pt3, (0, 0, 255), 1)
		# cv2.line(img1, pt1, pt3, (0, 0, 255), 1)
		points = get_points_in_triangles(img1_gray,pt1,pt2,pt3)
		print("Points :",points)

		for p in points:
			Bd = np.array([[t[0],t[2],t[4]],[t[1],t[3],t[5]],[1,1,1]])
			point = np.reshape(np.array([p[0],p[1],1]),(3,1))
			bary_pt1 = np.matmul(np.linalg.inv(Bd),point)
			if (bary_pt1>0).all()==True:
				pts = np.array([pt1,pt2,pt3])
				if (pts>0).all()==True:
					tri_id1 = [point_to_id(pts[0],fp1),point_to_id(pts[1],fp1),point_to_id(pts[2],fp1)]
					pts2 =  np.array([fp2[tri_id1[0]],fp2[tri_id1[1]],fp2[tri_id1[2]]])
					Ad = np.array([[fp2[tri_id1[0]][0],fp2[tri_id1[1]][0],fp2[tri_id1[2]][0]],
						[fp2[tri_id1[0]][1],fp2[tri_id1[1]][1],fp2[tri_id1[2]][1]],
						[1,1,1]])
					bary_pt2 = np.matmul(Ad,bary_pt1)
					print("Pt1 :",p[0],p[1])
					print("Pt2 :",int(bary_pt2[0]/bary_pt2[2]),int(bary_pt2[1]/bary_pt2[2]))
					img1[p[1],p[0]] = img2[int(bary_pt2[1]/bary_pt2[2]),int(bary_pt2[0]/bary_pt2[2])] 
	cv2.imshow("Triangulation 2:",img2)
	cv2.imshow("Triangulation :",img1)
	cv2.waitKey()

if __name__ == '__main__':
	main()
