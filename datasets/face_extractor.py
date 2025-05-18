import os
import cv2

# Set the source folder and destination folder
src_folder = "]"  # Folder with source images
dest_folder = ""  # Folder to save cropped faces

# Create the destination folder if it does not exist
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')

# Define the size of the cropped face image
crop_size = (512, 512)
surrounding_scale = 1.5  # Scale factor to determine the size of the surrounding region
bottom_reduction = 0.1  # Percentage of reduction from the bottom

# Iterate over the files in the source folder
for file in os.listdir(src_folder):
    try:
        # Set the source path
        src_path = os.path.join(src_folder, file)
        # Read the image
        img = cv2.imread(src_path)
        if img is None:
            print(f"Skipping file '{file}' - Unable to read image")
            continue
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # Iterate over the faces and crop them
        for (x, y, w, h) in faces:
            # Calculate the coordinates of the face region with surrounding area
            face_center_x = x + (w // 2)
            face_center_y = y + (h // 2)
            face_radius = max(w, h) // 2
            surrounding_radius = int(face_radius * surrounding_scale)
            x1 = max(face_center_x - surrounding_radius, 0)
            y1 = max(face_center_y - surrounding_radius, 0)
            x2 = min(face_center_x + surrounding_radius, img.shape[1])
            y2 = min(face_center_y + surrounding_radius - int(h * bottom_reduction), img.shape[0])  # Adjusted to reduce from the bottom
            # Crop the region around the face from the image
            face_img = img[y1:y2, x1:x2]
            # Resize the cropped face image to the desired size
            face_img = cv2.resize(face_img, crop_size)
            # Set the destination path
            dest_path = os.path.join(dest_folder, f"{file[:-4]}.jpg")
            # Save the cropped and resized face image to the destination folder
            cv2.imwrite(dest_path, face_img)
    except Exception as e:
        print(f"Skipping file '{file}' - Error: {str(e)}")
