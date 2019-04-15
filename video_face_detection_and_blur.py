# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse

import click
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import sys
import itertools
import cv2
import csv
import time

BLURRED_DIR = 'blur/'
# LOCATE_DIR = 'location/'
FRAMES_DIR = 'frames/'
INFO_DIR = 'info/'


# def print_result(filename, location):
#     top, right, bottom, left = location
#     print("{},{},{},{},{}".format(filename, top, right, bottom, left))

def video_detect_and_blur(img, input_path, output_path, model):
    if (os.stat(input_path + img).st_size != 0):
        name = img[:img.rfind('.')]
        unknown_image = face_recognition.load_image_file(input_path + img)
        face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=0, model=model)
        image = cv2.imread(input_path + img)
        for face_location in face_locations:
            top, right, bottom, left = face_location
            sub_face = image[top:bottom, left:right]
            # apply a gaussian blur on this new recangle image
            sub_face = cv2.GaussianBlur(sub_face, (51, 51), 75)
            # merge this blurry rectangle to our final image
            image[top:top + sub_face.shape[0], left:left + sub_face.shape[1]] = sub_face
        cv2.imwrite(output_path + BLURRED_DIR + img, image)
        with open(output_path + INFO_DIR + name + '.csv', 'w', newline='', encoding="utf-8") as csvfile:
            fieldnames = ['location_id', 'top', 'left', 'bottom', 'right']
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for (idx, loc) in enumerate(face_locations):
                top, right, bottom, left = loc
                writer.writerow(['id_' + str(idx), str(top), str(left), str(bottom), str(right)])


def merge_csv(output_path, name):
    # remove file name extension
    list_csv = sorted(os.listdir(output_path + name + "/" + INFO_DIR))
    for i in range(0, len(list_csv)):
        list_csv[i] = int(list_csv[i].strip('.csv'))
    list_csv = sorted(list_csv)
    list_csv = [str(l) for l in list_csv]
    rows = []
    for csv_file in list_csv:
        with open(output_path + name + "/" + INFO_DIR + csv_file + ".csv", 'r', newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                new_row = [csv_file] + row
                rows.append(new_row)
    with open(output_path + name + "/" + name + '.csv', 'w', newline='', encoding="utf-8") as csvfile:
        fieldnames = ['frame_id', 'location_id', 'top', 'left', 'bottom', 'right']
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        for r in rows:
            writer.writerow(r)


def process_images_in_process_pool(input_path, output_path, number_of_cpus, model):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    images_arr = os.listdir(input_path)
    input_path = [input_path] * len(images_arr)
    output_path = [output_path] * len(images_arr)
    function_parameters = zip(
        images_arr, input_path, output_path,
        itertools.repeat(model),
    )

    pool.starmap(video_detect_and_blur, function_parameters)


def write(output_path, name, ext, fps, size):
    files = sorted(os.listdir(output_path + BLURRED_DIR))
    for i in range(0, len(files)):
        files[i] = int(files[i].strip('.jpg'))
    files = sorted(files)
    out = cv2.VideoWriter(output_path.replace(BLURRED_DIR, '') + "blurred_" + name + ext.lower(), 0x00000020, fps, size)
    for filename in files:
        if os.path.isfile(output_path + BLURRED_DIR + str(filename) + ".jpg"):
            img = cv2.imread(output_path + BLURRED_DIR + str(filename) + ".jpg")
            out.write(img)
    out.release()

def extract_frames(input_path, output_path, name, ext):
    video = cv2.VideoCapture(input_path + name + ext)
    frame_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(0, frame_length):
        check, frame = video.read()
        cv2.imwrite(output_path + name + "/" + FRAMES_DIR + str(i + 1) + ".jpg", frame)


def main():
    # Multi-core processing only supported on Python 3.4 or greater
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', help='Video input path')
    parser.add_argument('--o', help='Output Path')
    parser.add_argument('--th', default=1, help='Number of Threads')
    parser.add_argument('--model', default="hog", help='Which face detection model to use. Options are "hog" or "cnn".')
    args = parser.parse_args()

    # parse --i flag to get input path
    input_path = args.i
    if (input_path[len(input_path) - 1] != '/'):
        input_path = input_path + "/"

    # parse --o flag to get output path
    output_path = args.o
    if (output_path[len(output_path) - 1] != '/'):
        output_path = output_path + "/"

    # parse --o flag to get output path

    threads_num = int(args.th)
    model = args.model

    video_names = os.listdir(input_path)

    for name in video_names:
        ext = name[name.rfind('.'):]
        name = name[:name.rfind('.')]
        fr_dir = output_path + name + "/" + FRAMES_DIR
        bl_dir = output_path + name + "/" + BLURRED_DIR
        info_dir = output_path + name + "/" + INFO_DIR
        if (os.path.isdir(output_path + name) == False):
            os.mkdir(output_path + name, 0o755)
        if (os.path.isdir(bl_dir) == False):
            os.mkdir(bl_dir, 0o755)
        if (os.path.isdir(fr_dir) == False):
            os.mkdir(fr_dir, 0o755)
        if (os.path.isdir(info_dir) == False):
            os.mkdir(info_dir, 0o755)

        print("Extracting frames from", name + ext)
        extract_frames(input_path, output_path, name, ext)

        if (sys.version_info < (3, 4)) and threads_num != 1:
            click.echo(
                "WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
            threads_num = 1
        
        print("Blurring frames from", name + ext)
        if os.path.isdir(fr_dir):
            if threads_num == 1:
                [video_detect_and_blur(image_file, fr_dir, output_path + name, model) for image_file in
                 os.listdir(fr_dir)]
            else:
                process_images_in_process_pool(fr_dir, output_path + name + "/", threads_num, model)
        print("Writing blurred frames to video")
        video = cv2.VideoCapture(input_path + name + ext)
        fps = float(video.get(cv2.CAP_PROP_FPS))
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))


        write(output_path=output_path + name + "/", name=name, ext=ext, fps=fps, size = size)

        merge_csv(output_path, name)

        # remove all temp file
        # import shutil
        # shutil.rmtree(bl_dir)
        # shutil.rmtree(fr_dir)
        # shutil.rmtree(info_dir)


if __name__ == "__main__":
    start = time.time()
    main()
    print("\n\n\nExecute in: " + str(time.time() - start) + " seconds\n\n\n")
