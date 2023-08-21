import cv2
import detect_with_API
import torch

if __name__ == '__main__':

    detector = detect_with_API.detectapi(weights='yolov7.pt')
    cap = cv2.VideoCapture("TRAJ-RIGHT-PITCHER_0818_1.mp4") 
    frame_count = 0
    with torch.no_grad():
        while True:
            output_file_name = 'output/frame_{}.txt'.format(frame_count) 
            content = []
            rec, img = cap.read()
            result, names = detector.detect([img])
            img = result[0][0]  # 每一帧图片的处理结果图片
            img_height, img_width = img.shape[:2] 
            # 每一帧图像的识别结果（可包含多个物体）
            for cls, (x1, y1, x2, y2), conf in result[0][1]:
                if names[cls]=='sports ball': # we only need ball
                    x_center = ((x1+x2)/2)/ img_width
                    y_center = ((y1+y2)/2)/ img_height
                    w = (x2-x1) / img_width
                    h = (y2-y1) / img_height
                    print(frame_count, names[cls],x_center, y_center, w, h) 
                    sub_content = '0 {} {} {} {}'.format(x_center, y_center, w, h) 
                    content.append(sub_content)

            # write in file
            with open(output_file_name, "w") as file:
                for line in content:
                    file.write(line + "\n")
            frame_count += 1