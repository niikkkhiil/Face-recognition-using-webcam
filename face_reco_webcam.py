import numpy as np
import face_recognition as fr
import cv2

video_capture = cv2.VideoCapture(0)

sidds_image = fr.load_image_file("D:\Computer Vision\Face Recognition using webcam\Face\Siddhesh\WhatsApp Image 2022-02-08 at 12.44.42 AM (3).jpeg")
sidd_face_encoding = fr.face_encodings(sidds_image)[0]

jennifer_image = fr.load_image_file("D:\Computer Vision\Face Recognition using webcam\Face\Jennifer_Lawrence\LBY_210805_JENNIFER_LAWRENCE_VF_05G_Shot_06_091_QC_sRGB_LR.jpg")
jenni_face_encoding = fr.face_encodings(jennifer_image)[0]

emma_image = fr.load_image_file("D:\\Computer Vision\\Face Recognition using webcam\\Face\\Emma_Rossum\\5498a52541195_-_hbz-emmy-rossum-promo.jpg")
emma_face_encoding = fr.face_encodings(emma_image)[0]

virat_image = fr.load_image_file("D:\Computer Vision\Face Recognition using webcam\Face\Virat_Kohli\Virat_Kohli.jpg")
virat_face_encoding = fr.face_encodings(virat_image)[0]

lionel_image = fr.load_image_file("D:\\Computer Vision\\Face Recognition using webcam\\Face\\Lionel_Messi\\16421717588322.jpg")
lionel_face_encoding = fr.face_encodings(lionel_image)[0]

ab_image = fr.load_image_file("D:\\Computer Vision\\Face Recognition using webcam\\Face\\AB_Divillers\\4454.jpg")
ab_face_encoding = fr.face_encodings(ab_image)[0]

manu_image = fr.load_image_file("D:\Computer Vision\Face Recognition using webcam\Face\Manushi_Chhillar\Manushi-Chhillar-to-start-social-media-campaign-to-talk-to-people-about-nutrition.jpg")
manu_face_encoding = fr.face_encodings(manu_image)[0]

taps_image = fr.load_image_file("D:\Computer Vision\Face Recognition using webcam\Face\Taapsee_Pannu\images.jpg")
taps_face_encoding = fr.face_encodings(taps_image)[0]

known_face_encondings = [sidd_face_encoding, jenni_face_encoding, emma_face_encoding, virat_face_encoding, lionel_face_encoding, ab_face_encoding, manu_face_encoding, taps_face_encoding]
known_face_names = ["Sidd", "Jennifer", "Emma", "King Kohli", "Messi", "AB Diviller", "Manushi", "Taapsee"]

while True: 
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

