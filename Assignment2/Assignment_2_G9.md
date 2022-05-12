# Linear Models

## Q1

$\hat{p}_{model}\sim \mathcal{N}(y;f(x;\theta),\sigma I)$  
The negative log-likelihood cost function,  
$J(x,y;\theta)=-\sum_{i=0}^{m-1}log\hat{p}_{model}(y^{(i)}|x^{(i)},\theta)=-\sum_{i=0}^{m-1}-\frac{1}{2\sigma^2}[y^{(i)}-f(x^{(i)};\theta)][y^{(i)}-f(x^{(i)};\theta)]^T$  
$\hat{\theta}=argmin_\theta J(\theta)=argmin_{\theta}\sum_{i=0}^{m-1}([y^{(i)}-f(x^{(i)};\theta)][y^{(i)}-f(x^{(i)};\theta)]^T)$  
<font color=red>How to prove this yield a maximum likelihood estimator?</font>

## Q2  
