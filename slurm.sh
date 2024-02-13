#!/bin/sh
#SBATCH --partition=maxgpu
#SBATCH --job-name train_qr
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mail-type ALL
#SBATCH --mail-user 

source /path/to/etc/profile.d/conda.sh

#source ~/.bashrc

conda activate alyne #activate the venv
#cd /path/to/CVproject/ #if you lounch the command from a different location
# Train YOLOv8s on your dataset for 120 epochs
yolo detect train data=data/qr_code.yaml model=yolov8s.pt epochs=120 imgsz=640 name=qr_code 

exit

#to launch just type: sbatch slurm.sh
#for the qr_code training it took 12 minutes only