from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import pdb
import math
import os
import os.path as osp
import re
import shutil
import subprocess as sp
import numpy as np
import sys

# number of clips each video has
VIDEO = {
    'speed_1': 15, 'speed_2': 15, 'speed_5': 3,
    'speed_10': 5, 'speed_14': 5, 
    'speed_48': 5, 'speed_49': 4, 'speed_50': 10, 'speed_51': 6,'speed_52': 7, 
    'speed_54': 10, 'speed_56': 10,
    }

# radar gun's velocity in each clips
SPEED = {
    'speed_1': [49, 52, 54, 57, 54, 57, 54, 56, 56, 56, 55, 56, 57, 57, 57],
    'speed_2': [56, 58, 57, 58, 57, 60, 57, 58, 60, 57, 59, 60, 60, 61, 60],
    'speed_5': [50, 51, 51],
    'speed_10': [58, 61, 60, 63, 59],
    'speed_14': [53, 54, 53, 50, 48],
    'speed_48': [63, 61, 61, 51, 59],
    'speed_49': [51, 52, 54, 53],
    'speed_50': [60 ,62, 60, 51, 60, 58, 55, 60, 69, 63],
    'speed_51': [48, 47, 47, 44, 46, 50],
    'speed_52': [61, 58, 60, 51, 54, 50, 53],
    'speed_54': [60, 62, 61, 65, 65, 67, 68, 70, 68, 70],
    'speed_56': [69, 69, 53, 70, 72, 69, 70, 70, 71, 72],
}

# old videos are filmed under 30 fps
FPS_30 = {'speed_1', 'speed_2', 'speed_5', 'speed_52'}

def dis(pa, pb):
    '''
    distance between two points
    '''
    return math.sqrt((pa[0] - pb[0]) * (pa[0] - pb[0]) + (pa[1] - pb[1]) * (pa[1] - pb[1]))

def dis_x(pa, pb):
    '''
    distance at x-axis between two points
    '''
    return abs(pa[0] - pb[0])

def dis_y(pa, pb):
    """
    distance at y-axis between twn points
    """
    return abs(pa[1] - pb[1])

def d2v(d, rlpp, interval):
    '''
    give pixel distance, return velocity in real world(mph)
    d: pixel distance
    rlpp: real length per pixel: how long one pixel stands for in real word
    interval: time interval for dis distance
    '''
    return abs(d) * rlpp * 3600 / interval

def v2d(v, rlpp, interval):
    '''
    give real velocity in real world(mph), return pixel distance on an image
    '''
    return v * interval / (rlpp * 3600)

def get_detect_data(clip_name):
    '''
    change directory to yolov5
    input the clip path
    make the detection and return the label data
    '''
    img = 1280  # gcp3.pt are trained under 1280*1280
    weights = 'gcp3.pt'
    data = 'data/fast_baseball.yaml'
    source = osp.join('calculate_velocity', clip_name + '.mov')
    max_det = 1

    label_dir = 'runs/detect/exp/labels'
    result_dir = 'runs/detect/exp'

    dir_solution = os.getcwd()
    os.chdir('../yolov5')  # enter yolov5 dir

    if osp.isdir(result_dir):
        shutil.rmtree(result_dir)

    # detect: 1 detection every frame, write labels, write probabilities
    command = 'python detect.py --img ' + str(img) + ' --weights ' + weights + ' --data ' + data + ' --source ' + source + ' --max-det ' + str(max_det) + ' --save-txt --save-conf'
    sp.run(command.split(' '))
    dir_yolov5 = os.getcwd()

    os.chdir(label_dir)  # get label data
    label_data = []

    pattern = re.compile(r'(.*)_(.*)_(.*).txt')
    for file in os.listdir():
        number_of_frame = re.search(pattern, file).groups()[-1]
        with open (file, 'r') as f:
            line = f.readlines()[0].strip()
            line += ' ' + number_of_frame
            label_data.append(line)
    
    os.chdir(dir_yolov5)  # delete this run directory
    shutil.rmtree(result_dir)
    os.chdir(dir_solution)
    return label_data

def get_calibration_data(clip_name):
    """
    change directory to yolov5
    input the clip path
    copy the calibration frame beforehead
    make the detection on this frame and return calibration data
    """
    weights = 'yolov5m6.pt'
    source = osp.join('calculate_velocity',  'calibration_' + clip_name[:clip_name.rfind('_')] + '.jpg')
    class_filter = 0  # only detect person
    max_det = 2  # detect pitcher and catcher

    label_dir = 'runs/detect/exp/labels'
    result_dir = 'runs/detect/exp'

    dir_solution = os.getcwd()
    os.chdir('../yolov5')  # enter yolov5 dir

    if osp.isdir(result_dir):
        shutil.rmtree(result_dir)
        
    command = 'python detect.py' + ' --weights ' + weights + ' --source ' + source + ' --max-det ' + str(max_det) + ' --classes ' + str(class_filter) + ' --save-txt'
    sp.run(command.split(' '))
    dir_yolov5 = os.getcwd()

    os.chdir(label_dir)  # get label data
    calibration_data = []

    for file in os.listdir():
        with open(file, 'r') as f:
            for line in f.readlines():
                calibration_data.append(line.strip().split(' ')) 

    os.chdir(dir_yolov5)  # delete this run directory
    shutil.rmtree(result_dir)
    os.chdir(dir_solution)
    return calibration_data

def outlier_boxplot_2d(x,y, whis=1.5):
        xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
        ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

        ##the box
        box = Rectangle(
            (xlimits[0],ylimits[0]),
            (xlimits[2]-xlimits[0]),
            (ylimits[2]-ylimits[0]),
            ec = 'k',
            zorder=0
        )
        # ax.add_patch(box)

        ##the x median
        vline = Line2D(
            [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
            color='k',
            zorder=1
        )
        # ax.add_line(vline)

        ##the y median
        hline = Line2D(
            [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
            color='k',
            zorder=1
        )
        # ax.add_line(hline)

        # the central point
        # ax.plot([xlimits[1]],[ylimits[1]], color='k', marker='o')

        ##the x-whisker
        ##defined as in matplotlib boxplot:
        ##As a float, determines the reach of the whiskers to the beyond the
        ##first and third quartiles. In other words, where IQR is the
        ##interquartile range (Q3-Q1), the upper whisker will extend to
        ##last datum less than Q3 + whis*IQR). Similarly, the lower whisker
        ####will extend to the first datum greater than Q1 - whis*IQR. Beyond
        ##the whiskers, data are considered outliers and are plotted as
        ##individual points. Set this to an unreasonably high value to force
        ##the whiskers to show the min and max values. Alternatively, set this
        ##to an ascending sequence of percentile (e.g., [5, 95]) to set the
        ##whiskers at specific percentiles of the data. Finally, whis can
        ##be the string 'range' to force the whiskers to the min and max of
        ##the data.
        iqr = xlimits[2]-xlimits[0]

        ##left
        left = np.min(x[x > xlimits[0]-whis*iqr])
        whisker_line = Line2D(
            [left, xlimits[0]], [ylimits[1],ylimits[1]],
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_line)
        whisker_bar = Line2D(
            [left, left], [ylimits[0],ylimits[2]],
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_bar)

        ##right
        right = np.max(x[x < xlimits[2]+whis*iqr])
        whisker_line = Line2D(
            [right, xlimits[2]], [ylimits[1],ylimits[1]],
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_line)
        whisker_bar = Line2D(
            [right, right], [ylimits[0],ylimits[2]],
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_bar)

        ##the y-whisker
        iqr = ylimits[2]-ylimits[0]

        ##bottom
        bottom = np.min(y[y > ylimits[0]-whis*iqr])
        whisker_line = Line2D(
            [xlimits[1],xlimits[1]], [bottom, ylimits[0]], 
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_line)
        whisker_bar = Line2D(
            [xlimits[0],xlimits[2]], [bottom, bottom], 
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_bar)

        ##top
        top = np.max(y[y < ylimits[2]+whis*iqr])
        whisker_line = Line2D(
            [xlimits[1],xlimits[1]], [top, ylimits[2]], 
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_line)
        whisker_bar = Line2D(
            [xlimits[0],xlimits[2]], [top, top], 
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_bar)

        ##outliers
        mask = (x<left)|(x>right)|(y<bottom)|(y>top)
        # print(mask)
        # ax.scatter(
            # x[mask],y[mask],
            # facecolors='none', edgecolors='k'
        # )
        return mask

def plot_moment(plt, left, right, color, label):
    """
    plot two vertical lines to show where the players are, or when the velocity happens
    """
    plt.axvline(left, color=color, label=label)
    plt.axvline(right, color=color, label=label)

VELOCITY_DATA = 'velocity_data'
CALIBRATION = 'calibration'

class Velocity:
    def __init__(self, clip_name, reference='height', resolution_x=1920, resolution_y=1080):
        self.clip_name = clip_name
        self.save_path = osp.join(VELOCITY_DATA, f'label_data_{clip_name}.txt')
        self.calibration_path = osp.join(VELOCITY_DATA, CALIBRATION + '_' + clip_name[:clip_name.rfind('_')] + '.txt')

        self.draw_subplot = 1
        self.draw_row = 3
        self.draw_col = 3
        self.plt = plt.figure(figsize=(16, 9))

        self.img_x, self.img_y = resolution_x, resolution_y  # resolution
        self.fps = 30 if clip_name[:clip_name.rfind('_')] in FPS_30 else 60  # fps
        self.round = 2  # round digit in functions

        self.pos = []  # original trajectory data, [[position x, position y, number of frame]]
        self.distance = []  # calculated data, [[d, dx, dy, number of intervals, start frame, v, vx, vy]]
        self.frame_pos = {}  # use to plot velocity back on trajectory figure, {frame: [pos x, pos y]}

        self.reference = reference  # choose height or distance to calculate

    def setup(self):
        """
        if the clip has no calibration data, detect that frame first\n
        save calibration data into txt and load data\n
        do the setup for further use
        """
        self.calibration = []  # calibration data
        if not osp.isfile(self.calibration_path):
            calibration_data = get_calibration_data(self.clip_name)
            with open(self.calibration_path, 'w') as f:
                for line in calibration_data:
                    self.calibration.append(line)
                    line = ' '.join(line) + '\n'
                    f.write(line)
        else:
            # print(f'{self.clip_name} has calibration data')
            with open(self.calibration_path, 'r') as f:
                for line in f.readlines():
                    self.calibration.append(line.strip().split(' '))
        
        self.calibration_x = [float(i[1]) for i in self.calibration]  # use distance
        self.calibration_height = [float(i[4]) for i in self.calibration]  # use height

        # setup part
        self.left_player = min(self.calibration_x[0], self.calibration_x[1]) * self.img_x  # use distance
        self.right_player = max(self.calibration_x[0], self.calibration_x[1]) * self.img_x

        self.short_player = min(self.calibration_height[0], self.calibration_height[1]) * self.img_y  # use height
        self.tall_player = max(self.calibration_height[0], self.calibration_height[1]) * self.img_y

        pixel_length = abs(self.calibration_x[0] - self.calibration_x[1]) * self.img_x
        real_length = (60 * 12 + 6) / 63360  # miles, 1 mile = 1,760 yards = 5,280 feet = 63,360 inches

        self.rlpp_distance = real_length / pixel_length  # real length per pixel, use distance
        self.rlpp_height = ((6 * 12 + 1) / 63360) / (self.tall_player)  # real lenght per pixel, use height(6 feet 1 inch)

        self.rlpp = self.rlpp_height if self.reference == 'height' else self.rlpp_distance

        self.interval = 1 / self.fps
        self.limit_v1, self.limit_v2 = 30, 110  # max and min velocity to detect

    def save_and_load(self):
        """
        if the clip has no detection data, detect first
        save detection data into txt and load the data
        """
        if not osp.isfile(self.save_path):
            label_data = get_detect_data(self.clip_name)
            with open(self.save_path, 'w') as f:
                for line in label_data:
                    self.pos.append(line)
                    f.write(line + '\n')
        else:
            # print(f'{self.clip_name} is already detected')
            with open(self.save_path, 'r') as f:
                self.pos = f.readlines()

        
        self.pos = [[i.split(' ')[1]] + [i.split(' ')[2]] + [i.split(' ')[-1]] for i in self.pos]  # write x, y, frame number
        self.pos = [[float(i[0]), float(i[1]), int(i[2])] for i in self.pos]  # TODO: only use detection position now, not using probability
        self.pos.sort(key=lambda x: x[2])  # sort by frame number

    def plot_trajectory_normalized_pixel(self):
        """
        plot trajectory in normalized pixel
        """
        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title(f'trajectory in normalized pixel, {len(self.pos)} detections')
        plt.xlim([0, 1])
        plt.ylim([-1, 0])
        plt.axis('on')
        for x, y, frame in self.pos:
            plt.scatter(x, -y)

    def plot_trajectory_real_pixel(self):
        """
        plot trajectory in real pixel
        """
        self.pos = [[i * self.img_x, j * self.img_y, k] for i, j, k in self.pos]  # normalized pixel to the real pixel
        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title(f'trajectory in real pixel, {len(self.pos)} detections')
        plt.xlim([0, self.img_x])
        plt.ylim([-self.img_y, 0])
        plt.axis('on')
        for x, y, frame in self.pos:
            plt.scatter(x, -y)

        for x, y, frame in self.pos:
            self.frame_pos[frame] = [x, y]

    def delete_repeat_detection(self):
        """
        1. all point at the same/nearly same position, delete
        now position data are normalized pixel
        round them and note which ones are detected repeatly
        in position data, round them, delete repeated ones
        still keep the original data(not rounded for further use)
        """
        copy_pos = [[round(x, self.round), round(y, self.round)] for x, y, frame in self.pos]
        seen = defaultdict(int)
        for index, (x, y) in enumerate(copy_pos):
            seen[(x, y)] += 1
        self.pos = [(x, y, frame) for x, y, frame in self.pos if seen[(round(x, self.round), round(y, self.round))] == 1]  # change original data

        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title(f'trajectory after deleting repeated points, {len(self.pos)} detections')
        plt.xlim([0, self.img_x])
        plt.ylim([-self.img_y, 0])
        plt.axis('on')
        for x, y, frame in self.pos:
            plt.scatter(x, -y)

    def plot_velocity_between_detection(self):
        """
        self.distance: [[d, dx, dy, number of interval, start frame, v, vx, vy]]
        d: distance
        dx: distance at x-axis
        dyt: distance at y-axis
        number of interval: intervals between two consecutive detections: frame2 - frame1
        start frame: frame1
        v: velocity
        vx: velocity at x-axis
        vy: velocity ay y-axis
        """
        for index, points in enumerate(zip(self.pos, self.pos[1:])):
            (*p1, f1), (*p2, f2) = points
            d, dx, dy = dis(p1, p2), dis_x(p1, p2), dis_y(p1, p2)
            number_of_interval = f2 - f1
            start_frame = f1
            vx = d2v(dx, self.rlpp, number_of_interval * self.interval)
            vy = d2v(dy, self.rlpp, number_of_interval * self.interval)
            v = d2v(d, self.rlpp, number_of_interval * self.interval)
            self.distance.append([d, dx, dy, number_of_interval, start_frame, v, vx, vy])

        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title(f'velocity betweeen consecutive detections, {len(self.distance)} data')
        plt.axis('on')
        # self.distance.sort(key=lambda x: x[3])  # sort by the start frame
        for _, _, _, _, start_frame, v, _, _ in self.distance:
            plt.scatter(start_frame, v)

    def delete_outranged_velocity(self):
        '''
        # 2. all distances that are out of range, delete
        # set a limit range of distance beforehead
        # sort data
        # loop the data, use distance not distance x, if the distance is our of range, delete
        # delete corresponding distance x too 
        '''
        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.axis('on')
        new_d = []
        for index, (_, _, _, _, start_frame, v, _, _) in enumerate(self.distance):
            if self.limit_v1 <= v <= self.limit_v2:
                plt.scatter(start_frame, v)
                new_d.append(self.distance[index])
        self.distance = new_d
        plt.title(f'velocity, deleting unvalid ones, {len(self.distance)} data')

    def delete_outliers_1dbox(self):
        """
        # 3. all distances that are outliers in boxplot, delete
        # use 1-d boxplot to find outliers in distance, not distance x
        # round all distances, round all outliers
        # delete all outliers
        """
        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title('1-d boxplot of velocity')
        plt.axis('on')
        ret = plt.boxplot([v for _, _, _, _, _, v, _, _ in self.distance])
        outliers = {round(i, self.round) for i in ret['fliers'][0].get_ydata()}

        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.axis('on')
        new_d = []
        for index, (_, _, _, _, start_frame, v, _, _) in enumerate(self.distance):
            if round(v, self.round) not in outliers:
                plt.scatter(start_frame, v)
                new_d.append(self.distance[index])
        self.distance = new_d
        plt.title(f'velocity, deleting outliers in 1d-boxplot, {len(self.distance)}')

    def delete_outliers_2dbox(self):
        """
        4. all distances that are outliers in 2d-boxplot, delete
        similar to 1-d boxplot
        """
        data = [[start_frame, v] for _, _, _, _, start_frame, v, _, _ in self.distance]
        x, y = np.array([i[0] for i in data]), np.array([i[1] for i in data])
        mask = outlier_boxplot_2d(x, y)
        self.distance = [d for d, m in zip(self.distance, mask) if not m]

        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.axis('on')
        for index, (_, _, _, _, start_frame, v, _, _) in enumerate(self.distance):
            plt.scatter(start_frame, v)
            plt.title(f'velocity, deleting outliers in 2d-boxplot, {len(self.distance)}')

    def tricky_velocity(self):
        """
        find the maximum vx with the minimum vy
        """
        index = 0
        min_vy = float('inf')
        for i, (_, _, _, _, _, v, vx, vy) in enumerate(self.distance):
            if round(vy, self.round) < round(min_vy, self.round):
                index = i
                min_vy = vy
        tricky_v = self.distance[index][-3]
        return tricky_v

    def calculate_and_save(self):
        """
        
        """
        index = self.clip_name.rfind('_')
        video, clip = self.clip_name[:index], int(self.clip_name[index + 1:])
        self.real_v = real_v=  SPEED[video][clip - 1]

        self.max_v = max_v = max(v for _, _, _, _, _, v, _, _ in self.distance)
        self.tricky_v = tricky_v = self.tricky_velocity()
        self.average_v = average_v = sum(v for _, _, _, _, _, v, _, _ in self.distance) / len(self.distance)
        
        
        error_max = (max_v - real_v) / real_v * 100
        error_average =(average_v - real_v) / real_v * 100
        error_tricky = (tricky_v - real_v) / real_v * 100
        # print(f'{self.clip_name} max v: {max_v:.1f}({error_max:.1f}%), tricky v: {tricky_v:.1f}({error_tricky:.1f}%)')
        print(f'{self.clip_name} max v: {max_v:.1f}({error_max:.1f}%)')

        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title(f'final figure, with real velocity')

        for _, _, _, _,start_frame, v, _, _ in self.distance:
            plt.scatter(start_frame, v)
        plt.axhline(real_v, color='g', label='real velocity')  # horizontal line, real velocity
        plt.axhline(max_v, color='r', label='max velocity')  # horizontal line, max velocity
        # plt.axhline(average_v, color='y', label='average velocity')  # horizontal line, average velocity
        # plt.axhline(tricky_v, color='k', label='average velocity')  # horizontal line, tricky velocity
        # plt.legend()

    def plot_velocity_on_trajectory(self):
        """
        visualize where the velocity happens on the original trajectory
        """
        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title('trajectory with the max velocity')

        plt.xlim([0, self.img_x])
        plt.ylim([-self.img_y, 0])
        plt.axis('on')
        for x, y, frame in self.pos:
            plt.scatter(x, -y)

        # find where is the max velocity and real velocity and tricky velocity on trajectory
        max_frame = real_frame = tricky_frame = 0
        max_intv = real_intv = tricky_intv = 0
        
        min_real_error = min_tricky_error = float('inf')  # use error to find which velocity is the closest to the real velocity

        for _, _, _, number_of_interval, start_frame, v, _, _ in self.distance:
            if v == self.max_v:
                max_frame = start_frame
                max_intv = number_of_interval
            if abs(v - self.real_v) < min_real_error:
                min_real_error = abs(v - self.real_v)
                real_frame = start_frame
                real_intv = number_of_interval
            if abs(v - self.tricky_v) < min_tricky_error:
                min_tricky_error = abs(v - self.tricky_v)
                tricky_frame = start_frame
                tricky_intv = number_of_interval


        max_v_start, _ = self.frame_pos[max_frame]
        max_v_end, _ = self.frame_pos[max_frame + max_intv]
        real_v_start, _ = self.frame_pos[real_frame]
        real_v_end, _ = self.frame_pos[real_frame + real_intv]
        tricky_v_start, _ = self.frame_pos[tricky_frame]
        tricky_v_end, _ = self.frame_pos[tricky_frame + tricky_intv]
        plot_moment(plt, max_v_start, max_v_end, 'r', 'max velocity')  # range of max velocity
        plot_moment(plt, real_v_start, real_v_end, 'g', 'real velocity')  # range of real velocity
        # plot_moment(plt, tricky_v_start, tricky_v_end, 'k', 'tricky velocity')  # range of tricky velocity

        # find where is the players
        plot_moment(plt, self.left_player, self.right_player, 'b', 'players')

        # plt.legend()
        plt.savefig(osp.join(VELOCITY_DATA, self.clip_name + '.png'))
        plt.close()

 
    def main(self):
        self.setup()
        self.save_and_load()

        # self.plot_trajectory_normalized_pixel()  # 9 figures for now so don't plot this one.
        # if so or adding more figures, remember to revise number of row and col in class Velocity.
        self.plot_trajectory_real_pixel()
        self.delete_repeat_detection()
        self.plot_velocity_between_detection()
        self.delete_outranged_velocity()
        self.delete_outliers_1dbox() 
        self.delete_outliers_2dbox()

        self.calculate_and_save()
        self.plot_velocity_on_trajectory()

if __name__ == '__main__':
    def f(num):
        return [f'speed_{num}_{i}' for i in range(1, VIDEO[f'speed_{num}'] + 1)]

    # clips = f(48) + f(49) + f(50) + f(51)
    # clips += f(1) + f(2) + f(5) 
    # clips += f(10) + f(14)
    clips = f(54) + f(56)

    for c in clips:
        V = Velocity(c)  # if not 1080p or want to use different reference, input the keyword arguments
        V.main()