# Linear Models

## Q1 & Q2

$\hat{p}_{model}\sim \mathcal{N}(y;f(x;\theta),\sigma I)$  
The negative log-likelihood cost function,  
$J(x,y;\theta)=-\sum_{i=0}^{m-1}log\hat{p}_{model}(y^{(i)}|x^{(i)},\theta)=-\sum_{i=0}^{m-1}-\frac{1}{2\sigma^2}[y^{(i)}-f(x^{(i)};\theta)]^2$  
The following proves the negative log-likelihood cost function yields a maximum likelihood estimator of w and b.  
$\hat{\theta}=\mathop{\arg\min}\limits_{\theta} J(\theta)=\mathop{\arg\min}\limits_{\theta}\sum_{i=0}^{m-1}\left | \left | y^{(i)}-f(x^{(i)};\theta) \right |  \right | ^2_2=\mathop{\arg}\limits_{\theta}\sum_{i=0}^{m-1}2(y^{(i)}-f(x^{(i)};\theta))\frac{\partial f(x^{(i)};\theta)}{\partial \theta}=0$, where $\theta = [w_0,w_1,b]^T$

Let $x\_^{(i)} = [x^{(i)},1]^T, \mathrm{X}\_=[x\_^{(0)},x\_^{(1)},...,x\_^{(m-1)}], \mathrm{y}=[y^{(0)},y^{(1)},...,y^{(m-1)}]$  
$f(x^{(i)};\theta) = w^Tx^{(i)}+b = \theta^Tx\_^{(i)}$.  
Then in form of matrix multiplication,
$\hat{\theta} = \mathop{\arg}\limits_{\theta} \mathrm{X}\_(\mathrm{y}-\theta^T\mathrm{X}\_)^T=0$  
$\hat{\theta}=(\mathrm{X}\_\mathrm{X}\_^T)^{-1}\mathrm{X}\_\mathrm{y}^T$

## Q3  

Let $x\_^{(i)} = [x^{(i)},1], \mathrm{X}\_=[x\_^{(0)},x\_^{(1)},...,x\_^{(m-1)}], \mathrm{y}=[y^{(0)},y^{(1)},...,y^{(m-1)}]$  
$f(x^{(i)};\theta) = w^Tx^{(i)}+b = \theta^Tx\_^{(i)}$.  
$\hat{\theta}=(\mathrm{X}\_\mathrm{X}\_^T)^{-1}\mathrm{X}\_\mathrm{y}^T = [0.1,0.4,0]^T$  
It is well-described by the regression model.

## Q4

$\hat{\theta}=[0.10,0.41,-0.05]^T$  
We could add a l2 regularization to J to avoid the overfitting caused by noisy sensor data.  

## Q5

I can not understand this question...  
$J(x,y;\theta)=\sum_{i=0}^{m-1}\frac{1}{2\sigma_i^2}[y^{(i)}-f(x^{(i)};\theta)]^2$  
$\hat{\theta}=\mathop{\arg\min}\limits_{\theta}\sum_{i=0}^{m-1}\left | \left | \sigma_i(y^{(i)}-f(x^{(i)};\theta)) \right |  \right | ^2_2 = \mathop{\arg}\limits_{\theta}\mathrm{X}\_ \underline{\sigma}^T(\mathrm{y}-\theta^T\mathrm{X}\_)^T=0$  
$\hat{\theta} = (\mathrm{X}\_\underline{\sigma}^T\mathrm{X}\_^T)^{-1}\mathrm{X}\_\underline{\sigma}^T\mathrm{y}^T$
Diagonal matrix $\underline{\sigma} = \begin{bmatrix}
\sigma_0 & & &\\
&\sigma_1 & &\\
& & ... & \\
& & & \sigma_N
\end{bmatrix}
$

## Q6

$\mathrm {X}=   \begin{bmatrix}
0  & 0&1&1\\
0  & 1&0&1
\end{bmatrix},y = \begin{bmatrix}
0  \\
1  \\
1  \\
0  
\end{bmatrix},\hat{\theta}=[0,0,0.5]^T$  
Note that $w_0=w_1=0$, only b=0.5. It means that for arbitrary inputs, our ouputs is always 0.5, which is optimal in regression model yet not ideally fit the data.

# Nonlinear functions  

## Q7  

- ReLu: $\frac{\mathrm{d} f}{\mathrm{d} x}= \begin{cases}
1,  x>0 \\
not\ differentiable, x=0 \\ 
0,x<0
\end{cases}$
- Sigmoid: $\frac{\mathrm{d} f}{\mathrm{d} x}=(1-\sigma(x))(\sigma(x))$
- Softmax: $\frac{\mathrm{d} f_j}{\mathrm{d} x}=-f_j^3$  

## Q8

- For ReLu, the gradient is 1;
- For Sigmoid, the gradient is 0;
- For Softmax, if only $x_j>>0$, and other $x_i$ is not in the same scale, the gradient will also be zero.  

# Shallow(i.e. not deep...) nonlinear models

## Q9

With one more linear mapping, the model is still linear which would omit same parameters as Q6. So it does not suffice.

## Q10

decision boundary: $h_1-2*h_2=0.5$
Check image under "figure/Q10.png"

## Q11

h1 = max(0,x1+x2),h2=max(0,x1+x2-1)  
x1+x2<0, h1=0,h2=0  
0<x1+x2<1,h1=x1+x2,h2=0  
x1+x2>1,h1=x1+x2,h2=x1+x2+1,x1+x2=2.5

# Binary classification with logistic regression

## Q12

For multi-class classfication problems, the softmax function would be useful It enables calculation of probability of groud truth among all classes.  

## Q13