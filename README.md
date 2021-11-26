# GRU-SSU-AMG
This code is the implementation of this paper (An Accurate GRU-Based Power Time-Series Prediction Approach With Selective State Updating and Stochastic Optimization).

# Environment version
TensorFlow-gpu = 1.14
Keras = 2.1.5

#Usage
You can run the wind-dataset-hyper-parameters-select-GRU-SSU-AMG.py file directly to implement the wind power forecasting task.  

Firstly, you need to place these two files (optimizers.py & recurrent.py) in the location specified by Keras (./anaconda3/envs/tf_1.14/Lib/site-packages/keras/).  

Then, you can execute the following command:  

```
python wind-dataset-hyper-parameters-select-GRU-SSU-AMG.py
```
# Figure
![](https://github.com/zwd2016/GRU-SSU-AMG/blob/main/framework.png)

# References
If you are interested, please cite this paper.  

@article{zhengaccurate,
  title={An Accurate GRU-Based Power Time-Series Prediction Approach With Selective State Updating and Stochastic Optimization},
  author={Zheng, Wendong and Chen, Gang},
  journal={IEEE Transactions on Cybernetics},
  publisher={IEEE},
  year= {2021},
  doi= {10.1109/TCYB.2021.3121312},
}
