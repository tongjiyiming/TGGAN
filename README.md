# TGGAN
This is the project for the following paper, 
accepted in proceeding for 30th The Web Conference 2021, Ljubljana, Slovenia.
You can cite for now.
```
@article{zhang2020tg,
  title={TG-GAN: Deep Generative Models for Continuously-time Temporal Graph Generation},
  author={Zhang, Liming and Zhao, Liang and Qin, Shan and Pfoser, Dieter},
  journal={arXiv preprint arXiv:2005.08323},
  year={2020}
}
```

The main training and inference codes for different datasets are in `main_*.py` scripts.

There is also codes developed for dynamic graph metric in MMD distance evaluation.
The discrete-time graph metrics are in `evaluation.py`, 
and the folder `continuous_time_evaluation_and_DSBM_matlab` contains the continuous-time
graph metrics and also DSBM models. The referred libraries can be found in the folder too.
Please cite this paper properly if you need to use the evaluation codes.