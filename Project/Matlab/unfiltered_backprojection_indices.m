clc; clear; close all;

% pkg load image

P = phantom(256); 
figure; imshow(P);

N = 180;
step = 180/N;
thetas = 0:step:179;

[R,xp] = radon(P,thetas);

sinogram = R;
figure; imshow(sinogram,[]);

numOfParallelProjections = size(sinogram,1);
numOfAngularProjections  = length(thetas); 

% convert thetas to radians
thetas = (pi/180)*thetas;

% set up the backprojected image
BPI = zeros(numOfParallelProjections,numOfParallelProjections);

% find the middle index of the projections
midindex = floor(numOfParallelProjections/2) + 1;

% set up filter
filterMode = 'sheppLogan'; % 'sheppLogan' or 'ramLak'

if mod(numOfParallelProjections,2) == 0
    halfFilterSize = floor(1 + numOfParallelProjections);
else
    halfFilterSize = floor(numOfParallelProjections);
end

if strcmp(filterMode,'ramLak')
    filter = zeros(1,halfFilterSize);
    filter(1:2:halfFilterSize) = -1./([1:2:halfFilterSize].^2 * pi^2);
    filter = [fliplr(filter) 1/4 filter];
elseif strcmp(filterMode,'sheppLogan')
    filter = -2./(pi^2 * (4 * (-halfFilterSize:halfFilterSize).^2 - 1) );
end

% convolve sinogram with filter
sinogramFiltered = zeros(size(sinogram));
for i = 1:numOfAngularProjections
    sinogramFiltered(:,i) = conv(sinogram(:,i),filter,'same');
end

% loop over each projection
for r = 1:size(BPI,1)
  for c = 1:size(BPI,2)
    for i = 1:numOfAngularProjections
        t = thetas(i);
        x = c - midindex;
        y = - (r - midindex);
        
		% transformation
        rotCoords = midindex + round(x*cos(t) + y*sin(t));

        % check boundaries
        if ((rotCoords > 0) && (rotCoords <= numOfParallelProjections))
            % update BPI
            BPI(r,c) = BPI(r,c) + sinogramFiltered(rotCoords,i)./numOfAngularProjections;
        end
    end
  end
end

figure; imshow(BPI,[]);
