clc;
clear all;
close all;

x = loadMNISTImages('train-images-idx3-ubyte'); % train images
t = loadMNISTLabels('train-labels-idx1-ubyte'); % train labels

% Split training set into training and validation set with ratio 90/10
images_Tr = x(:,1:0.9*size(x,2));               % Training images              
labels_Tr = t(1:0.9*size(t,1),1);               % Training labels

images_Val = x(:,1:0.2*size(x,2));              % Validation images
labels_Val = t(1:0.2*size(t,1),1);              % Validation labels

% Load the test dataset. This dataset is untouched during the training
% process
images_T = loadMNISTImages('t10k-images-idx3-ubyte');   % Testing images
labels_T = loadMNISTLabels('t10k-labels-idx1-ubyte');   % Testing labels

% % Preview the first 36 samples from MNIST dataset (optional)
% figure
% colormap(gray)                  % set to grayscale
% 
% for i = 1:36                                % preview the first 36 samples
%     subplot(6,6,i)                          % plot them in 6x6 grid
%     digit = reshape(images_Tr(:,i), [28,28]);    % row = 28x28 image
%     imagesc(digit)                          % show the image
%     title(num2str(labels_Tr(i)))                 % show the label    
% end

% Reshape the images data into 4D matrix 28x28x1x60000 (1 is because of
% gray scale), and convert labels data into categorical in order to be used
% in Convolutional Neural Network
images_CNN = reshape(images_Tr,28,28,1,size(images_Tr,2));
labels_CNN = categorical(labels_Tr);

images_Val_CNN = reshape(images_Val,28,28,1,size(images_Val,2));
labels_Val_CNN = categorical(labels_Val);

images_T = reshape(images_T,28,28,1,size(images_T,2));
labels_T = categorical(labels_T);

% Define CNN architecture
layers = [...
    imageInputLayer([28 28 1])
    
    convolution2dLayer(5,8)             % Filter 1: 8 filters size 5x5
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)     % Pooling layer: 2x2 with Stride 2
    
    convolution2dLayer(5,10)            % Filter 2: 16 filters size 5x5
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)     % Pooling layer: 2x2 with Stride 2
    
    fullyConnectedLayer(10)             % Fully connected network layer
    softmaxLayer
    classificationLayer];               % Softmax layer with 10 classes

% Training options
options = trainingOptions('sgdm',...
    'ExecutionEnvironment','auto',...
    'InitialLearnRate', 0.1,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.5,...
    'LearnRateDropPeriod',1,...
    'ValidationData',{images_Val_CNN,labels_Val_CNN},...
    'ValidationFrequency',50,...
    'ValidationPatience',6,...
    'MaxEpochs',3,...
    'MiniBatchSize',256,...
    'Plots','training-progress');

% Train the network
net = trainNetwork(images_CNN,labels_CNN,layers,options);

% Test trained network
YPred = classify(net,images_T);
YTest = labels_T;

% Display the accuracy based on test set
test_accuracy = (sum(YPred == YTest)/numel(YTest))*100

% To test an image:
% classify(net,process('IMAGE_NAME.jpg'))


