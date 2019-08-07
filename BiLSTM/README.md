# About the dataset
This BiLSTM speech segmentation project uses AMI corpus, to reproduce the experiment, you can download the audio files listed in `prepare_dataset.py` one by one from the website:

http://groups.inf.ed.ac.uk/ami/download/

Another way to download the dataset quickly, you can refer to this project:

https://github.com/pyannote/pyannote-db-odessa-ami

There is a bash script `AMI/db_download/download.sh`, you can select the audio files you need.

BTW, audio files should be placed in this folder, I will clean the code and make the structure more comfortable later.
# About the speech segmentation algorithm
Please check the details in "Speaker Change Detection in Broadcast TV using Bidirectional Long Short-Term Memory Networks"