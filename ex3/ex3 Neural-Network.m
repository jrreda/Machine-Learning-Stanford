%% Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

% In the previous part of this exercise, you implemented multi-class logistic
% regression to recognize handwritten digits.However, logistic regression cannot
% form more complex hypotheses as it is only a linear classier. (You could add more
% features such as polynomial features to logistic regression, but that can be very
% expensive to train). In this part of the exercise, you will implement a neural network
% to recognize handwritten digits using the same training set as before.
% The neural network will be able to represent complex models that form non-linear hypotheses. 

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
%
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

load('ex3data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 2: Loading Pameters ================
% In this part of the exercise, we load some pre-initialized 
% neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
% Load saved matrices from file
load('ex3weights.mat'); 
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26


%% ================= Part 3: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(m);

for i = 1:10
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end
