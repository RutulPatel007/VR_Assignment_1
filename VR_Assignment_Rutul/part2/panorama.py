import numpy as np
import cv2
import imutils
from tqdm import tqdm
import os

#Function to visualize matches between keypoints of two images
def visualizeMatches(imageA,imageB,interestA,interestB,matches,status):
    """
    Visualize matches between keypoints of two images

    Args:
        imageA (numpy.ndarray): The first input image
        imageB (numpy.ndarray): The second input image
        interestA (numpy.ndarray): Keypoints from the first image
        interestB (numpy.ndarray): Keypoints from the second image
        matches (list): List of matched keypoints
        status (numpy.ndarray): Status of each match

    Returns:
        numpy.ndarray: Image with visualized matches
    """
    
    hA,wA=imageA.shape[:2]
    hB,wB=imageB.shape[:2]
    viz=np.zeros((max(hA,hB),wA+wB,3),dtype="uint8")# Create an empty canvas for visualization
    viz[0:hA,0:wA]=imageA
    viz[0:hB,wA:]=imageB
    for((trainIdx,queryIdx),s) in zip(matches,status):
        if s == 1:# If the match is good, draw a line
            ptA=(int(interestA[queryIdx][0]),int(interestA[queryIdx][1]))
            ptB=(int(interestB[trainIdx][0])+wA,int(interestB[trainIdx][1]))
            cv2.line(viz,ptA,ptB,(0,255,0),1)
    return viz



# Function to detect keypoints and compute descriptors using SIFT


# Function to match keypoints between two images using BFMatcher and ratio test
def interestPointMacher(interestA,interestB,xA,xB,ratio,re_proj):

    """
    Match keypoints between two images using BFMatcher and ratio test

    Args:
        interestA (numpy.ndarray): Keypoints from the first image
        interestB (numpy.ndarray): Keypoints from the second image
        xA (numpy.ndarray): Descriptors from the first image
        xB (numpy.ndarray): Descriptors from the second image
        ratio (float): Ratio for Lowe's ratio test
        re_proj (float): RANSAC reprojection threshold

    Returns:
        tuple: Matches, homography matrix, and status of each match
    """
    matcher=cv2.BFMatcher()
    rawMatches=matcher.knnMatch(xA,xB,2) #KNN matching
    matches=[]
    for m in rawMatches:
        if len(m) == 2 and m[0].distance<m[1].distance*ratio: #Lowe's ratio test
            matches.append((m[0].trainIdx,m[0].queryIdx))
    if len(matches)>4: #Ensure enough matches exist to compute homography
        ptsA=np.float32([interestA[i] for (_,i) in matches])
        ptsB=np.float32([interestB[i] for (i,_) in matches])
        H,status=cv2.findHomography(ptsA,ptsB,cv2.RANSAC,re_proj) #homography matrix
        return (matches,H,status)
    return None



def siftDetectDescriptor(image): 

    """
    Detect keypoints and compute descriptors using SIFT

    Args:
        image (numpy.ndarray): Input image

    Returns:
        tuple: Keypoints and descriptors
    """
    descriptor=cv2.SIFT_create() #SIFT
    kps,features=descriptor.detectAndCompute(image,None)
    kps=np.float32([kp.pt for kp in kps]) #keypoints to float32 array
    return (kps,features)



# Function to crop the extra black regions from the stitched panorama
def cropBlackRegion(image):
    """
    Crop the extra black regions from the stitched panorama

    Args:
        image (numpy.ndarray): Input stitched image

    Returns:
        numpy.ndarray: Cropped image
   
    """
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY) # Convert to binary image
    contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x,y,w,h=cv2.boundingRect(contours[0]) # Find bounding box of the content area
        return image[y:y+h-1,x:x+w-1] # the cropped img
    return image

# Function to perform image stitching
def stichImages(images,ratio=0.75,re_proj=5.0,show_overlay=False):
    """
    Stitch two images to create a panorama

    Args:
        images (list): List containing two images to be stitched
        ratio (float): Ratio for Lowe's ratio test
        re_proj (float): RANSAC reprojection threshold
        show_overlay (bool): Flag to show overlay of matches

    Returns:
        numpy.ndarray: Stitched panorama image
   
    """
    imageB,imageA=images
    interestA,xA=siftDetectDescriptor(imageA)
    interestB,xB=siftDetectDescriptor(imageB)
    M=interestPointMacher(interestA,interestB,xA,xB,ratio,re_proj)
    if M is None:
        print("Not enough matches found.")
        return None
    matches,H,status=M
    pano_img=cv2.warpPerspective(imageA,H,(imageA.shape[1]+imageB.shape[1],imageA.shape[0]))# Warp imageA
    pano_img[0:imageB.shape[0],0:imageB.shape[1]]=imageB # Overlay imageB on top
    pano_img = cropBlackRegion(pano_img)
    if show_overlay:
        visualization=visualizeMatches(imageA,imageB,interestA,interestB,matches,status)
        return (pano_img,visualization)
    return pano_img

def panorama(input_dir, output_dir):
    
    img_path = []
    
    # Collecting only image files with (.jpeg, .jpg, .png)
    valid_extensions = {".jpg", ".jpeg", ".png"}  # Add other extensions as per your need
    
    for i in os.listdir(input_dir):
        if os.path.splitext(i)[1].lower() in valid_extensions:  # Check for valid extensions defined above
            img_path.append(os.path.join(input_dir, i))
    
    assert len(img_path) > 0, "No image found in input folder"
    
    # Sorting images numerically if their names are numbers
    img_path.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))  
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    left_img = cv2.imread(img_path[0])
    left_img = imutils.resize(left_img, width=600)


    # Stitching images together
    for i in tqdm(range(1, len(img_path))):
        right_img = cv2.imread(img_path[i])
        right_img = imutils.resize(right_img, width=600)
        
        pano_img = stichImages([left_img, right_img], show_overlay=True)
        
        if pano_img is not None:
            left_img, viz = pano_img
            cv2.imwrite(os.path.join(output_dir, f"stitched_image_{i}.jpg"), viz)
    
    cv2.imwrite(os.path.join(output_dir, "panorama.jpg"), left_img)
    print("Panorama saved successfully.")



# if __name__=='__main__':

    # parser=argparse.ArgumentParser(description="Process input panorama.")
    # parser.add_argument("input_dir",help="Name of the input directory")
    # parser.add_argument("output_dir",help="Name of the output directory")
    # args=parser.parse_args()
    # panorama(args.input_dir,args.output_dir)
