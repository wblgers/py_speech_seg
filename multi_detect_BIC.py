# -*- coding:UTF-8 -*-

from __future__ import print_function
import BIC.speech_segmentation as bic_seg

frame_size = 256
frame_shift = 128
sr = 16000

seg_point = bic_seg.multi_segmentation("duihua_sample.wav", sr, frame_size, frame_shift, plot_seg=True, save_seg=True,
                                   cluster_method='bic')
print('The segmentation point for this audio file is listed (Unit: /s)', seg_point)






