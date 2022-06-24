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

# number of clips each video has
VIDEO = {'speed_1': 15, 'speed_2': 15, 'speed_5': 3, 'speed_50': 10, 'speed_48': 5, 'speed_49': 4, 'speed_51': 6}

# radar gun's velocity in each clips
SPEED = {
    'speed_1': [49, 52, 54, 57, 54, 57, 54, 56, 56, 56, 55, 56, 57, 57, 57],
    'speed_2': [56, 58, 57, 58, 57, 60, 57, 58, 60, 57, 59, 60, 60, 61, 60],
    'speed_5': [50, 51, 51],
    'speed_48': [63, 61, 61, 51, 59],
    'speed_49': [51, 52, 54, 53],
    'speed_50': [60 ,62, 60, 51, 60, 58, 55, 60, 69, 63],
    'speed_51': [48, 47, 47, 44, 46, 50],
}

# old videos are filmed under 30 fps
FPS_30 = {'speed_1', 'speed_2', 'speed_5'}

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

def d2v(d, rlpp, interval):
    '''
    d: distance
    rlpp: real length per pixel
    interval: 1 / fps
    return velocity in mph
    '''
    return abs(d) * rlpp * 3600 / interval

def v2d(v, rlpp, interval):
    '''
    v: velocity, in mph
    rlpp: real length per pixel
    interval: 1 / fps
    return distance in miles
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
    
    calibration_data = [float(i[1]) for i in calibration_data]  # write position x

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

VELOCITY_DATA = 'velocity_data'
CALIBRATION = 'calibration'

class Velocity:
    def __init__(self, clip_name, resolution=[1920, 1080]):
        self.clip_name = clip_name
        self.save_path = osp.join(VELOCITY_DATA, f'label_data_{clip_name}.txt')
        self.calibration_path = osp.join(VELOCITY_DATA, CALIBRATION + '_' + clip_name[:clip_name.rfind('_')] + '.txt')

        self.draw_subplot = 1
        self.draw_row = 3
        self.draw_col = 3
        self.plt = plt.figure(figsize=(16, 9))

        self.img_x, self.img_y = resolution  # resolution
        self.fps = 30 if clip_name[:clip_name.rfind('_')] in FPS_30 else 60  # fps
        self.round = 2  # round digit in functions

        self.pos = []  # original trajectory data, [[position x, position y, number of frame]]
        self.distance = []  # distance data between frames for further calculation, [[d, dx, number of intervals, start frame]]   
        self.frame_pos = {}  # use to plot velocity back on trajectory figure, {frame: [pos x, pos y]}

    def setup(self):
        """
        if the clip has no calibration data, detect that frame first\n
        save calibration data into txt and load data\n
        do the setup for further use
        """
        self.calibration_x = []  # calibration data
        if not osp.isfile(self.calibration_path):
            calibration_data = get_calibration_data(self.clip_name)
            with open(self.calibration_path, 'w') as f:
                for line in calibration_data:
                    self.calibration_x.append(line)
                    f.write(str(line) + '\n')
        else:
            # print(f'{self.clip_name} has calibration data')
            with open(self.calibration_path, 'r') as f:
                temp = f.readlines()
            self.calibration_x = [float(i.strip()) for i in temp]

        # setup part
        pixel_length = abs(self.calibration_x[0] - self.calibration_x[1]) * self.img_x
        real_length = 60.5 / 5280  # miles, 1 mile = 1,760 yards = 5,280 feet = 63,360 inches
        self.rlpp = real_length / pixel_length  # real length per pixel

        self.interval = 1 / self.fps
        limit_v1, limit_v2 = 30, 110  # max and min velocity to detect
        self.limit_d1 = limit_v1 * self.interval / (3600 * self.rlpp)  # max pixel distance
        self.limit_d2 = limit_v2 * self.interval / (3600 * self.rlpp)  # min pixel distance

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
        plt.title('trajectory in normalized pixel')
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
        plt.title('trajectory in real pixel')
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
        plt.title('trajectory after deleting repeated detection')
        plt.xlim([0, self.img_x])
        plt.ylim([-self.img_y, 0])
        plt.axis('on')
        for x, y, frame in self.pos:
            plt.scatter(x, -y)

    def plot_velocity_between_detection(self):
        """

        """
        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title('velocity betweeen consecutive detections')
        plt.axis('on')
        for index, points in enumerate(zip(self.pos, self.pos[1:])):
            (*p1, f1), (*p2, f2) = points
            d, dx = dis(p1, p2), dis_x(p1, p2)
            number_of_interval = abs(f1 - f2)
            start_frame = min(f1, f2)
            self.distance.append([d, dx, number_of_interval, start_frame])  # distance have: d, dx, internvals, start frame
        
        self.distance.sort(key=lambda x: x[3])  # sort by the start frame
        for index, (d, _, number_of_intv, frame) in enumerate(self.distance):
            plt.scatter(frame, d2v(dx, self.rlpp, number_of_intv * self.interval))

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
        plt.title('velocity, after deleting unvalid ones')
        plt.axis('on')
        new_d = []
        for index, (d, dx, number_of_intv, frame) in enumerate(self.distance):
            if self.limit_d1 <= d <= self.limit_d2:
                plt.scatter(frame, d2v(dx, self.rlpp, number_of_intv * self.interval))
                new_d.append([d, dx, number_of_intv, frame])
        self.distance = new_d

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
        ret = plt.boxplot([d2v(i[1], self.rlpp, i[2] * self.interval) for i in self.distance])
        outliers = {round(i, self.round) for i in ret['fliers'][0].get_ydata()}

        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title('velocity, after deleting outliers in 1d-boxplot')
        plt.axis('on')
        new_d = []
        for index, (d, dx, number_of_intv, frame) in enumerate(self.distance):
            if round(d / number_of_intv, self.round) not in outliers:
                plt.scatter(frame, d2v(dx, self.rlpp, number_of_intv * self.interval))
                new_d.append([d, dx, number_of_intv, frame])
        self.distance = new_d

    def delete_outliers_2dbox(self):
        """
        4. all distances that are outliers in 2d-boxplot, delete
        similar to 1-d boxplot
        """
        x = np.array([frame for _, _, _, frame in self.distance])
        y = np.array([d / number_of_intv for d, _, number_of_intv, _ in self.distance])
        mask = outlier_boxplot_2d(x, y)
        self.distance = [d for d, m in zip(self.distance, mask) if not m]

        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title('velocity, after deleting outliers in 2d-boxplot')
        plt.axis('on')
        for d, dx, number_of_intv, frame in self.distance:
            plt.scatter(frame, d2v(dx, self.rlpp, number_of_intv * self.interval))

    def calculate_and_save(self):
        """
        calculate the velocity using distance x, not distance
        """
        max_velocity = max(d2v(dx, self.rlpp, number_of_intv * self.interval) for _, dx, number_of_intv, _ in self.distance)
        index = self.clip_name.rfind('_')
        video, clip = self.clip_name[:index], int(self.clip_name[index + 1:])
        error = (max_velocity - SPEED[video][clip - 1]) / SPEED[video][clip - 1] * 100
        print(f'pitch velocity in {self.clip_name}: {max_velocity:.2f} ({SPEED[video][clip - 1]}, {error:.1f}%)')

        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title('final figure, with real velocity')

        for d, dx, number_of_intv, frame in self.distance:
            plt.scatter(frame, d2v(dx, self.rlpp, number_of_intv * self.interval))
        plt.axhline(SPEED[video][clip - 1], color='g') # horizontal, real velocity
        plt.axhline(max_velocity, color='r') # horizontal, calculated velocity

    def plot_velocity_on_trajectory(self):
        """
        plot the max velocity range in trajectory
        """
        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title('trajectory with the max velocity')

        plt.xlim([0, self.img_x])
        plt.ylim([-self.img_y, 0])
        plt.axis('on')
        for x, y, frame in self.pos:
            plt.scatter(x, -y)

        max_velocity = 0
        max_frame = 0
        max_intv = 0
        for _, dx, num_of_intv, frame in self.distance:
            v = d2v(dx, self.rlpp, num_of_intv * self.interval)
            if v > max_velocity:
                max_velocity = v
                max_frame = frame
                max_intv = num_of_intv

        max_x_start, _ = self.frame_pos[max_frame]
        max_x_end, _ = self.frame_pos[max_frame + max_intv]
        # print(max_x_start, max_x_end)
        plt.axvline(max_x_start, color='r') # start frame with max velocity
        plt.axvline(max_x_end, color='r') # end frame with max velocity
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

    clips = f(1) + f(2) + f(5) + f(48) + f(49) + f(50) + f(51)

    for c in clips:
        V = Velocity(c)
        V.main()