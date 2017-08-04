# -*- coding:UTF-8 -*-

from __future__ import print_function
import speech_segmentation as seg

frame_size = 256
frame_shift = 128
sr = 16000

seg_point = seg.multi_segmentation("dialog4.wav",sr,frame_size,frame_shift,plot_seg=True)
print(seg_point)






