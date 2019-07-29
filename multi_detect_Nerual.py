# -*- coding:UTF-8 -*-

from __future__ import print_function
import BiLSTM.bilstm_speech_seg_predict as nerual_seg


seg_point = nerual_seg.multi_segmentation('1.wav')
print('The segmentation point for this audio file is listed (Unit: /s)', seg_point)