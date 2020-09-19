docker run -itd --gpus all --name tggan \
-e NB_USER=lem -e GRANT_SUDO=yes --user root \
-v $PWD:$HOME/work -w $HOME/work  -p 8893:8888 \
tongjiyiming/cuda-tensorflow:tggan
# tongjiyiming/cuda-tensorflow:tggan-netgan \
# bash
# sudo start-notebook.sh

# script with arguments
# sudo nvidia-docker run $4 --name $1 -e NB_USER=lem -e GRANT_SUDO=yes --user root \
# -v "$PWD":$PWD -w $PWD -p $3 $2 \
# start-notebook.sh 
