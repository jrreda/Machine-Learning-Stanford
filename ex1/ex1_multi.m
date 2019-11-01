%% Machine Learning Online Class - Exercise 1: Linear regression with multiple variables

% In this part, you will implement linear regression with multiple variables to predict the prices of houses.

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Clear and Close Figures
clear all; close all; clc

%% ================ Part 1: Feature Normalization ================

% This section of the script will start by loading and displaying some values from this dataset.
% By looking at the values, note that house sizes are about 1000 times the number of bedrooms.
% When features differ by orders of magnitude, first performing feature scaling can make gradient descent converge much more quickly.
    % Your task here is to complete the code in featureNormalize.m to:
% 			1. Subtract the mean value of each feature from the dataset.
% 			2. After subtracting the mean, additionally scale (divide) the feature values by their respective "standard deviations".

fprintf('Loading data ...\n');

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

%%%%% Note: You have to complete the code in computeCost.m
% compute and display initial cost
[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];
fprintf('============================================================================\n');

%% ================ Part 2: Gradient Descent ================

% Previously, you implemented gradient descent on a univariate regression problem.
% The only difference now is that there is one more feature in the matrix X.
% The hypothesis function and the batch gradient descent update rule remain unchanged
fprintf('Running gradient descent ...\n');

% Run gradient descent
% Choose some alpha value
alpha = 0.1;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
%%%%% Note: You have to complete the code in gradientDescent.m
% compute and display initial cost
[theta, ~] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))
num_iters = 100;
alpha_vect = [0.01, 0.3, 0.1, 0.03, 0.01, 1];

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================


price = [1, featureNormalize([1650 3])] * theta; % Enter your price formula here

% ============================================================

fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f', price);
fprintf('============================================================================\n');

%% ================ Part 3: Selecting Learning Rate ================
% In this part of the exercise, you will get to try out dierent learning rates for the dataset and find a learning rate that converges quickly.
% You can change the learning rate by modifying the code below and changing the part of the code that sets the learning rate.
figure;
hold on;
% Choose some alpha value
for alpha = alpha_vect
	% Init Theta and Run Gradient Descent 
	theta = zeros(3, 1);
	[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

	% Plot the convergence graph
	figure;
	plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
	xlabel('Number of iterations');
	ylabel('Cost J');
	hold on

	% Display gradient descent's result
	fprintf('Theta computed from gradient descent: \n');
	fprintf(' %f \n', theta);
	fprintf('\n');

	fprintf('Program paused. Press enter to continue.\n');
	pause;
end
hold off 


fprintf('============================================================================\n');
%% ================ Part 3: Normal Equations ================

% The following code computes the closed form solution for linear regression using the normal equations.
% NOTE: You should complete the code in normalEqn.m
% After doing so, you should complete this code to predict the price of a 1650 sq-ft, 3 br house.

fprintf('Solving with normal equations...\n');
%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = [1, featureNormalize([1650 3])] * theta; % Enter your price formula here

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);