function neurFilters = makeGDPerfectreconstrFilterBank(S,Lt,centers)

numFilters = numel(centers);

freq = linspace(0,S / 2, ceil(Lt / 2));
% centers = linspace(1300,13000,numFilters);   % separation between the filters, in Hz
deltas = ones(numFilters) * 200;     % transition time between two adjacent filter, in Hz

neurFiltersFFT = zeros(ceil(Lt / 2), numFilters);

neurFiltersFFT(freq < centers(1) - deltas(1), 1) = 1;
indexization = boolean( (centers(1)-deltas(1) < freq) .* (freq < centers(1)+deltas(1)) );
neurFiltersFFT(indexization, 1) = cos(0.5 * pi * log2(1 + (freq(indexization) + deltas(1) - centers(1) ) / 2 / deltas(1)));

for filterNumber = 2:numFilters
    neurFiltersFFT(indexization, filterNumber) = sqrt( 1 - neurFiltersFFT(indexization, filterNumber-1).^2 ); 
    
    indexization = boolean( (centers(filterNumber-1)+deltas(filterNumber-1) < freq) .* (freq < centers(filterNumber)-deltas(filterNumber)) );
    neurFiltersFFT(indexization, filterNumber) = 1;
    
    indexization = boolean( (centers(filterNumber)-deltas(filterNumber) < freq) .* (freq < centers(filterNumber)+deltas(filterNumber)) );
    neurFiltersFFT(indexization, filterNumber) = cos(0.5 * pi * log2(1 + (freq(indexization) + deltas(filterNumber) - centers(filterNumber) ) / 2 / deltas(filterNumber)));
end

neurFilters = zeros(Lt, numFilters);
for filterNumber = 1:numFilters
    neurFilters(:,filterNumber) = real(fftshift(ifft( [neurFiltersFFT(:,filterNumber)' permute( neurFiltersFFT(end:-1:1, filterNumber), [2 1]) ] ) ) );
end

