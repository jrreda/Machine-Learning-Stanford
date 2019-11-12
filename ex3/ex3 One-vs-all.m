%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

% For this exercise, you will use logistic regression and neural networks to recognize handwritten digits (from 0 to 9).
% Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks.
% This exercise will show you how the methods you've learned can be used for this classication task. In the first part of the exercise, you will extend your previous implemention of logistic regression and apply it to one-vs-all classification.

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     ex3.mlx - MATLAB Live Script that steps you through the exercise
%     ex3data1.mat - Training set of hand-written digits
%     ex3weights.mat - Initial weights for the neural network exercise
%     submit.m - Submission script that sends your solutions to our servers
%     displayData.m - Function to help visualize the dataset
%     fmincg.m - Function minimization routine (similar to fminunc)
%     sigmoid.m - Sigmoid function
%     *lrCostFunction.m - Logistic regression cost function
%     *oneVsAll.m - Train a one-vs-all multi-class classifier
%     *predictOneVsAll.m - Predict using a one-vs-all multi-class classifier
%     *predict.m - Neural network prediction function	

%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%
%  After you have implemented vectorization for logistic regression,
%  you will now add regularization to the cost function.
fprintf('\nCompute cost and gradient for logistic regression with regularization...\n')

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('Cost: %f | Expected cost: 2.534819\n',J);
fprintf('Gradients:\n'); fprintf('%f\n',grad);
fprintf('Expected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003');


%% ================ Part 3: Tain One-Vs-All ================
% In this part of the exercise, you will implement one-vs-all classification
% by training multiple regularized logistic regression classifiers, 
% one for each of the  classes in our dataset (Figure 1).
% In the handwritten digits dataset, , but your code should work for any value of .
fprintf('\nTraining One-vs-All Logistic Regression...\n')

num_labels = 10; % 10 labels, from 1 to 10 
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 4: Predict for One-Vs-All ================
%  After training your one-vs-all classifier, you can now use it to predict the digit contained in a given image.
fprintf('\Predicting One-vs-All Logistic Regression...\n')

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

pause;