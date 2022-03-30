# Non-Iterative Recovery from Nonlinear Observations using Generative Models

This repository contains the codes for the paper: Non-Iterative Recovery from Nonlinear Observations using Generative Models

-------------------------------------------------------------------------------------

## Dependencies

* Python 3.6

* Tensorflow 1.5.0

* Scipy 1.1.0

*  PyPNG

## Running the code

-------------------------------------------------------------------------------------

We provide the guideline to run our OneShot on the MNIST and celebA datasets. 

### Run wavelet_basis.py to generate the wavelet basis for Lasso-W
cd src
python wavelet_basis.py

### Run our OneShot and the compared methods(Lasso, CSGM, BIPG, OneShot) on the MNIST dataset for 1bit measurements 

cd src

python mnist_main_OneShot.py --nonlinear-model 1bit --batch-size 10 --num-outer-measurement-ls 25 50 100 200 400 --num-random-restarts 10 --max-update-iter 100 --method-ls  Lasso  CSGM BIPG OneShot --noise-std-ls 0.1 0.5  1  5 

python mnist_main_OneShot_imgn.py --nonlinear-model 1bit --batch-size 10 --num-outer-measurement-ls 25 50 100 200 400 --num-random-restarts 10 --max-update-iter 100 --method-ls  Lasso  CSGM BIPG OneShot --noise-std-ls 0.01 0.05 0.1 0.5 

### Run our OneShot and the compared methods(Lasso, CSGM, PGD, OneShot) on the MNIST dataset for cubic measurements 

cd src

python mnist_main_OneShot.py --nonlinear-model cubic --batch-size 10 --num-outer-measurement-ls 25 50 100 200 400 --num-random-restarts 10 --max-update-iter 100 --method-ls  Lasso CSGM PGD OneShot --noise-std-ls 0.1 0.5  1  5

### Run our OneShot and the compared methods(Lasso-W, CSGM, BIPG, OneShot) on the CelebA dataset for 1bit measurements 

cd src

python celebA_main_OneShot.py    --nonlinear-model 1bit --num-outer-measurement-ls 4000 6000 10000 15000 --num-random-restarts 3 --num-experiments 10 --max-update-iter 100 --method-ls Lasso-W CSGM BIPG OneShot --noise-std-ls 0.01 0.05  0.1  0.5


### Run our OneShot and the compared methods (Lasso-W, CSGM, PGD, OneShot) on the CelebA dataset for cubic measurements 

cd src

python celebA_main_OneShot.py    --nonlinear-model cubic --num-outer-measurement-ls  4000 6000 10000 15000 --num-random-restarts 3 --num-experiments 10 --max-update-iter 100 --method-ls  Lasso-W CSGM PGD OneShot --noise-std-ls 0.01 0.05  0.1  0.5

##  Our OneShot and the compared projected methods can be also implemented with faster projections (https://github.com/yuqili3/NPGD_linear_inverse_prob). They are denoted as BIFPG, FPGD and OneShotF in the paper.


## References

Large parts of the code are derived from [Bora et al.](https://github.com/AshishBora/csgm), [ Shah et al.](https://github.com/shahviraj/pgdgan), [Liu et al.] (https://github.com/selwyn96/Quant_CS), and [Raj et al.] (https://github.com/yuqili3/NPGD_linear_inverse_prob) 
