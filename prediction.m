clc;
clear;
close all;

% -------------------- Load Trained CNN --------------------
load('trainedNet.mat', 'net');

% -------------------- Parameters --------------------
imageSize = [64 64];  % Same size used during training
testFolder = 'C:\work\signal_dataset';  % Adjust to your path
outputFigure = 'PredictionFigure.png';
outputCSV = 'prediction_results.csv';

% -------------------- Load Test Images --------------------
imdsTest = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @(x) preprocessImage(x, imageSize));

% -------------------- Predict Labels --------------------
predictedLabels = classify(net, imdsTest);
trueLabels = imdsTest.Labels;
accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
disp(['Prediction Accuracy on Test Folder: ' num2str(accuracy * 100, '%.2f') '%']);

% -------------------- Display Class Distribution --------------------
T = tabulate(trueLabels);
disp('Number of samples per true class:');
labels = T(:,1);
labelStrs = cellfun(@char, labels, 'UniformOutput', false);  % convert to char
disp(array2table(cell2mat(T(:,2:3)), 'VariableNames', {'Label','Count'}, 'RowNames', labelStrs));

% -------------------- Display and Save Prediction Images --------------------
numSamplesToDisplay = 12;
idxOccupied = find(trueLabels == 'occupied');
idxVacant = find(trueLabels == 'vacant');

% Pick 6 random samples from each class
selectedIdx = [randsample(idxOccupied, min(6, numel(idxOccupied))); ...
               randsample(idxVacant, min(6, numel(idxVacant)))];

figure('Name', 'Prediction Samples', 'NumberTitle', 'off', 'Color', 'w');
for i = 1:length(selectedIdx)
    img = readimage(imdsTest, selectedIdx(i));
    trueLbl = string(trueLabels(selectedIdx(i)));
    predLbl = string(predictedLabels(selectedIdx(i)));

    subplot(3, 4, i);
    imshow(img);
    title(['True: ' trueLbl ', Pred: ' predLbl], 'FontSize', 10);
end

% Save high-quality figure
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, outputFigure, '-dpng', '-r300'); % 300 DPI
disp(['Prediction image saved as: ' outputFigure]);

% -------------------- Save Predictions to CSV --------------------
fileNames = imdsTest.Files;
resultsTable = table(fileNames, trueLabels, predictedLabels, ...
    'VariableNames', {'Filename', 'TrueLabel', 'PredictedLabel'});

try
    writetable(resultsTable, outputCSV);
    disp(['Prediction results saved to ' outputCSV]);
catch
    warning('Unable to write CSV file. Permission issue or file is open.');
end

% -------------------- Save All Results --------------------
save('predictionResults.mat', 'trueLabels', 'predictedLabels', 'accuracy');

% -------------------- Optional Confusion Chart --------------------
if exist('confusionchart', 'file') == 2
    figure;
    confusionchart(trueLabels, predictedLabels);
    title('Confusion Matrix for Test Predictions');
else
    warning('confusionchart not available in this MATLAB version.');
end

% -------------------- Preprocessing Function --------------------
function imgOut = preprocessImage(filename, imageSize)
    [img, map] = imread(filename);
    if ~isempty(map)
        img = ind2rgb(img, map);
        img = rgb2gray(img);
    elseif size(img, 3) == 3
        img = rgb2gray(img);
    end
    imgOut = imresize(img, imageSize);
end
