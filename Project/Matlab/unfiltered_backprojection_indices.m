clc; clear; close all;

P = phantom(256); 
figure; imshow(P);

N = 90;
step = 180/N;
thetas = 0:step:179;

[R,xp] = radon(P,thetas);
figure; imagesc(thetas,xp,R); colormap(gray);

sinogram = R;
% sinogram = imread('sinogram.jpg');

numOfParallelProjections = size(sinogram,1);
numOfAngularProjections  = length(thetas); 

% convert thetas to radians
thetas = (pi/180)*thetas;

% set up the backprojected image
BPI = zeros(numOfParallelProjections,numOfParallelProjections);

% find the middle index of the projections
midindex = floor(numOfParallelProjections/2) + 1;


figure;
% loop over each projection
for r = 1:size(BPI,1)
  for c = 1:size(BPI,2)
    for i = 1:numOfAngularProjections
        t = thetas(i);
        x = c - midindex;
        y = r - midindex;
        % figure out which projections to add to which spots
        rotCoords = round(midindex + x*sin(t) + y*cos(t));

        % check which coords are in bounds
        if ((rotCoords > 0) && (rotCoords <= numOfParallelProjections))
            % summation
            BPI(r,c) = BPI(r,c) + sinogram(rotCoords,i)./numOfAngularProjections;
        end
        % visualization on the fly

    end
  end
end
im = mat2gray(BPI);
im = imrotate(im,90);
imshow(im);