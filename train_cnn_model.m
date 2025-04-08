clc;
clear;
close all;
% Parameters
imageSize = [64 64];  % Resize images to this size
datasetPath = 'C:\work\signal_dataset';  % Adjust to your dataset path
trainSplit = 0.8;

% Set up ImageDatastore with updated Read Function
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @(x) preprocessImage(x, imageSize));

% Split data into training and validation sets
[imdsTrain, imdsValidation] = splitEachLabel(imds, trainSplit, 'randomized');

% Define CNN Architecture
layers = [
    imageInputLayer([imageSize 1])

    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

% Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the network
[net, info] = trainNetwork(imdsTrain, layers, options);
% Save the trained network and training info
save('trainedNet.mat', 'net', 'info');

% Validate the model
YPred = classify(net, imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation) / numel(YValidation);
disp(['Validation Accuracy: ' num2str(accuracy * 100, '%.2f') '%']);

% --------- Preprocessing Function (Place at the End) ---------
function imgOut = preprocessImage(filename, imageSize)
    [img, map] = imread(filename);
    if ~isempty(map)
        img = ind2rgb(img, map); % Convert indexed image to RGB
        img = rgb2gray(img);
    elseif size(img, 3) == 3
        img = rgb2gray(img); % RGB to grayscale
    end
    imgOut = imresize(img, imageSize);
end
