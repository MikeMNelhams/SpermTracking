
%===============================================================================
% Read in gray scale demo image.
folder = "C:\Users\Shane\Desktop\Year 3\Mathematical and Data Modelling\Phase B\MDM3_B\OrientationJ_Data\1st source\Video 0\Frames\"; % Determine where demo folder is (works with all versions).
baseFileName = 'frame0.jpg';
first_bbox = [642 410 62 64];
% Get the full filename, with path prepended.
fullFileName = fullfile(folder, baseFileName);
% Check if file exists.
if ~exist(fullFileName, 'file')
	% The file doesn't exist -- didn't find it there in that folder.
	% Check the entire search path (other folders) for the file by stripping off the folder.
	fullFileNameOnSearchPath = baseFileName; % No path this time.
	if ~exist(fullFileNameOnSearchPath, 'file')
		% Still didn't find it.  Alert user.
		errorMessage = sprintf('Error: %s does not exist in the search path folders.', fullFileName);
		uiwait(warndlg(errorMessage));
		return;
	end
end
rgbImage_raw = imread(fullFileName);
rgbImage = imcrop(rgbImage_raw,first_bbox);
E = edge(rgb2gray(rgbImage),'canny',0.5);
% override some default parameters
params.minMajorAxis = 30;
params.maxMajorAxis = 90;

% note that the edge (or gradient) image is used
bestFits = ellipseDetection(E, params);
fprintf('Output %d best fits.\n', size(bestFits,1));
angle = bestFits(:,5);
fprintf('Angle is %d.',angle)
figure;
image(rgbImage);
%ellipse drawing implementation: http://www.mathworks.com/matlabcentral/fileexchange/289 
ellipse(bestFits(:,3),bestFits(:,4),bestFits(:,5)*pi/180,bestFits(:,1),bestFits(:,2),'k');
