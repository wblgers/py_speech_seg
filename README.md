# py_speech_seg
A toolkit to implement segmentation on speech based on BIC

## Dependency
- Python>=3.5
- Librosa
- Numpy
- Scipy

You can use the installation of Anaconda to satisfy the required packages except [Librosa](https://github.com/librosa/librosa).

To install librosa, you can try the following command:

`conda install -c conda-forge librosa`
## Example Usage


1. Run script multi_detect.py to test the segmentation on a simple wav file:

    `python multi_detect.py`

   And you can get a speech segmentation result as showm below:

![Alt text](https://github.com/wblgers/py_speech_seg/raw/master/pictures/Demo1.png)
2. In the python script of multi_detect.py, there is a function call after some parameter settings:
    
    `seg_point = seg.multi_segmentation("dialog4.wav",sr,frame_size,frame_shift,plot_seg=False,save_seg=True)`
    
   To save the segmented audio into wav files, set the flag `save_seg=True`

   To plot out the wave figure in time domain with segmentation lines on, set the flag `plot_seg=True`


## Reference
*Speaker, Environment and Channel Change Detection and Clustering via the Bayesian Information Criterion*, by IBM T.J. Watson Research Center