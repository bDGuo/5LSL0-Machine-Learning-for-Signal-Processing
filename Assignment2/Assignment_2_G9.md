# Linear Models

## Q1 & Q2

$\hat{p}_{model}\sim \mathcal{N}(y;f(x;\theta),\sigma I)$  
The negative log-likelihood cost function,  
$J(x,y;\theta)=-\sum_{i=0}^{m-1}log\hat{p}_{model}(y^{(i)}|x^{(i)},\theta)=-\sum_{i=0}^{m-1}-\frac{1}{2\sigma^2}[y^{(i)}-f(x^{(i)};\theta)][y^{(i)}-f(x^{(i)};\theta)]^T$
The following proves the negative log-likelihood cost function yields a maximum likelihood estimator of w and b.  
$\hat{\theta}=\mathop{\arg\min}\limits_{\theta} J(\theta)=\mathop{\arg\min}\limits_{\theta}\sum_{i=0}^{m-1}([y^{(i)}-f(x^{(i)};\theta)][y^{(i)}-f(x^{(i)};\theta)]^T)=\mathop{\arg\min}\limits_{\theta}\sum_{i=0}^{m-1}\left | \left | y^{(i)}-f(x^{(i)};\theta) \right |  \right | ^2_2=\mathop{\arg}\limits_{\theta}\sum_{i=0}^{m-1}2(y^{(i)}-f(x^{(i)};\theta))\frac{\partial f(x^{(i)};\theta)}{\partial \theta}=0$, where $\theta = [w_0,w_1,b]^T$  

## Q3  

Let $x\_^{(i)} = [x^{(i)},1], \mathrm{X}\_=[x\_^{(0)},x\_^{(1)},...,x\_^{(m-1)}], \mathrm{y}=[y^{(0)},y^{(1)},...,y^{(m-1)}]$
 $f(x^{(i)};\theta) = w^Tx^{(i)}+b = \theta^Tx\_^{(i)}$.  
$\hat{\theta}=(\mathrm{X}\mathrm{X}^T)^{-1}\mathrm{X}\mathrm{y}^T = [0.1,0.4,0]^T$

## Q4

$\hat{\theta}=[0.10,0.41,-0.05]^T$  
We would assume that $noise\sim \mathcal{N}(0,\sigma^2)$  

## Q5

I can not understand this question...  