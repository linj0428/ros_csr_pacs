# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path
import random
import shutil
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
#when in pycharm,you should comment out the next line
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import time
#fileDir是总的测试集的文件夹，tarDir是这次用于测试的随机抽取出的图片存放的文件夹
#fileDir = 'J:/a2022/guosai/yolov5_guosai/renqun/images/test/'  # 源图片文件夹路径
#tarDir = 'J:/a2022/guosai/yolov5_guosai/renqun/images/test_random/'  # 移动到新的文件夹路径

#先删除test_random的文件夹中的图片，然后从测试集中随机复制20张到名为test_random的文件夹中，确保每次都是20张
'''def copyFile(fileDir):
    rootdir = "J:/a2022/guosai/yolov5_guosai/renqun/images/test_random"
    filelist = os.listdir(rootdir)
    for file in filelist:
        if '.jpg' in file:
            del_file = rootdir + '\\' + file  # 当代码和要删除的文件不在同一个文件夹时，必须使用绝对路径
            os.remove(del_file)  # 删除文件
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        shutil.copyfile(fileDir + name, tarDir + name)
    return
'''

@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path

            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                move_put=1
                people_num=0
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    cv2.putText(im0, f"{n} {names[int(c)]}{'s' * (n > 1)}", (5+125*move_put, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                    people_num=n+people_num
                    move_put=move_put+1
                    print(f"{n} {names[int(c)]}{'s' * (n > 1)}")
                    if names[int(c)]== 'red':
                         global max_reds_num
                         max_reds_num = max( max_reds_num,n)
                    if names[int(c)]=='blue':
                        global max_blues_num
                        max_blues_num = max(max_blues_num, n)
                    '''if names[int(c)] == 'grey':
                        max_greys_num = max(max_greys_num, n)
                    if names[int(c)] == 'black':
                        max_blacks_num = max(max_blacks_num, n)'''

                cv2.putText(im0, f"all={people_num}", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                print(f"总人数={people_num}结果一并在图片上标识了")
                print(p.name)
                global max_all_num
                if max_all_num<=people_num:
                    rootdir2 = '/home/eaibot/yolov5_guosai/renqun/images/result/'
                    fileDir2 = '/home/eaibot/yolov5_guosai/renqun/images/result/'
                    filelist = os.listdir(fileDir2)
                    for file in filelist:
                        if '.jpg' in file:
                            del_file = rootdir2 + '' + file  # 当代码和要删除的文件不在同一个文件夹时，必须使用绝对路径
                            os.remove(del_file)  # 删除文件
                    result_path='renqun/images/result/'+p.name
                    #cv2.imwrite(result_path, im0)
                    max_all_num=max(max_all_num,people_num)
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        cv2.imwrite(result_path, im0)
            # Stream results
            im0 = annotator.result()
            if   view_img:
                if p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

                cv2.imshow(str(p), im0)
                cv2.waitKey(0)  # 1 millisecond

            # Save results (image with detections)
            #不需要存其他的图片，这边保存注释掉，结果图片在上面保存到了result中
            '''
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)

                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
'''
        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/eaibot/yolov5_guosai/runs/train/exp8/weights/best.pt', help='model path(s)')  # 修改处 权重文件
    parser.add_argument('--source', type=str, default='/home/eaibot/yolov5_guosai/renqun/images/test_guosai', help='source')# 修改处 图像、视频或摄像头
    parser.add_argument('--data', type=str, default='/home/eaibot/yolov5_guosai/data/people.yaml', help='(optional) dataset.yaml path')  # 修改处
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')  # 修改处 高 宽
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')  # 置信度
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')# 非极大抑制
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # 修改处
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    #copyFile(fileDir)
    global save_flag#控制保存图片
    save_flag=0
    # 定义四个数字分别来存储红色蓝色灰色黑色以及总人数的最大值
    #192-195行也注释了因为还没有黑色和灰色，之后训练的模型里四个类型要与那边if判断的名称相同
    global max_reds_num
    max_reds_num=0
    global max_blues_num
    max_blues_num = 0
    #global max_greys_num
    #max_greys_num = 0
    #globa max_blacks_num
    #max_blacks_num = 0
    global max_all_num
    max_all_num = 0
    while (1):
        # 摄像头保存图像的位置为fileDir1
        fileDir1 = '/home/eaibot/yolov5_guosai/renqun/images/test_guosai/'  # 源图片文件夹路径
        rootdir = '/home/eaibot/yolov5_guosai/renqun/images/test_guosai/'
        filelist = os.listdir(fileDir1)
        if filelist != []:
            opt = parse_opt()
            main(opt)
            print(f"最终认定红色有{max_reds_num}人,蓝色有{max_blues_num}人,总共有{max_all_num}人")#以后也不需要添加，因为灰色和黑色不用输出
            print(f"Result saved to 'renqun/images/result/'")
            # 但是统计总人数的时候算进去了不用担心。
            #这种获得人数的方法需要确保黑色和灰色识别准确率,如果把干扰项分成两类训练那么识别时也必须两类精确得分出来否则会影响统计的总人数。
            for file in filelist:
                if '.jpg' in file:
                    del_file = rootdir + '/' + file  # 当代码和要删除的文件不在同一个文件夹时，必须使用绝对路径
                    os.remove(del_file)  # 删除文件







