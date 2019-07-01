function [data] = loadMNISTdata(dataLocation)
%LOADMNISTDATA is a function to read the IDX file type which is comes with
%   mnist datasets 
%   [data] = loadMNISTdata(dataLocation) returns a matlab matrix with mnist
%    data in the provided location

fp = fopen(dataLocation);
assert(fp ~= -1, 'Error loading file....');

magic = fread(fp, 1, 'int32', 0 , 'ieee-be');
%assert(magic == 2051, ['Bad magic number in ', dataLocation, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

data = fread(fp, inf, 'unsigned char', 'ieee-be');
data = reshape(data, numCols, numRows, numImages);
data = permute(data,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
data = reshape(data, size(data, 1) * size(data, 2), size(data, 3));
% Convert to double and rescale to [0,1]
data = double(data) / 255;

end
