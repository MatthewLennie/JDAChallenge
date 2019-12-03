# JDAChallenge

This read provides details for an interview challenge.

## Part 1

For this version of the implentation, I ingored any time series considerations. 
I did this on the principle of implementing a simple model first. 

For this model, I used an XGBoost model. 
I choose it as suitable for the size and complexity of the data in the task. 
The advantage of the tree model is the ability to handle interaction terms.
For instance, users may continue to ride in bad weather during a work day, when they have to get to work. 

You can of course generate interaction terms for a linear regression, but this was the simplest to implement. 
It performed reasonably well, in the assess function, I plot the absolute deviations and errors.
From this, we can see that the model has good predictive capabilities. 
It could be that a better model is needed, in this case, I would suggest getting more data, more features. 
It may be then beneficial to looking into a neural net solution. 
 
My code performs hyper-parameter search using gridCV. I did an intitial wider sweep of
the hyperparameters but reduced the subset to within a smaller search space. 
The hyperparameters ranges were chosen based on a reading of literature.

I would assess this model based on the business case for the model. 
i.e. is any predictive power that's better than chance already highly beneficial 
or does anything below 95% expose the company to too much volatilty?
How often will the model be used? Is there a latency requirement? 
Some of these questions may drive towards using neural networks or even an ensemble of methods, 
others will encourage the use of even cheaper linear regression. 
XGBoost seems to provide a good first estimation and has the advantage of being
extensible up to larger datasets.    

The current model provided Mean Absolute Error: 41.16758608653871 from the data which had:
Mean Absolute Deviations: 186.6
![alt text](https://github.com/MatthewLennie/JDAChallenge/blob/master/ImageOfPerformance.png)

### Testing
What I have done instead is provided a good level of testing for the loading module. 
From this sample you should be able to see my capabilities. 
I don't quite have the chance to roll through the full system to create test cases. 

But I would create some extra test cases that would perform sanity checks on the output of
the model i.e. bike usage is never negative etc.. 

Regarding the code base, I have run autopeap8 over the code. 
An example usage of the module is provided in the Pipeline file in the main function. 
Docstrings are provided. 

To see testing, run pytest -v in the main directory. 

### Deployment. 

It is outside of the scope of this task, but a good way to deploy this code would be 
as a microservice in a docker container, especially as XGBoost can be finniky about OS. 
In another coding interview I have demonstrated
my introductory knowledge in this area: https://github.com/MatthewLennie/BayesProject

For this project, a requirements.txt file is provided. 

## Part 2
 A larger production sized data set produces two main problems;
1. Large space and large processing requirements. 
2. Approaching real large data sets means that firstly, the data needs to be accessed
in a way that doesn't require all of the data be pulled on a single machine. 

These can create bottle necks in terms of memory read/write, network bandwidth for distributed training 
as well as processing bottle necks. 


Usually to handle these problems you would use a library which handles the boilerplate 
of doing lazy evalulation as well as mapping operations over multiple machines and then
reducing results. One example of a library is DASK; a library that I have been using in my research. 

Of course, once problems get to a certain size, we have to begin performing training
on multiple cores and then multiple nodes. 
XGBoost by default has paralell training implemented which is one of the reasons I choose it. 
New implementations from the NVidia Rapids project enables GPU learning of XGBoost, furthermore
DASK_ML provides the ability to train on multiple machines. 
Other algorithms such as neural networks can be also be trained across multiple machines, 
though care must be taken to adjust the training procedure to take into account the 
changes in batch size etc..

In principle we would want to build up the Machine Learning pipeline in a way that we can 
scale it. Frameworks such as MLFlow provide the infrastructure to do so. 
The current implementation has a number of bottle necks, especially in the data loading stage where the data is stored as .CSV. 
 
I have used DASK for multiple node computation but so far datasets have been small. 
I have take university courses in SQL previously but would have to get back up to speed. 


