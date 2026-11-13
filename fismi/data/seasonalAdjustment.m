function valuesSA = seasonalAdjustment(dates, values)
% Seasonal adjustment of price indices using X-11
% Input: MATLAB table with TIME_PERIOD and price indices
% Output: MATLAB table with seasonally adjusted price indices
    
    % Convert dates    
    dates = datetime(string(dates(:)), 'InputFormat', 'yyyy-MM-dd');
    dateNum = datenum(dates(:));

    % Initialize output
    valuesSA = values;

    for c = 1:size(values,2)

        colData = log(values(:,c));

        if sum(isfinite(colData)) > 50

            output = x11([dateNum, colData], 12, 'add');

            % Replace raw indices with adjusted indices
            valuesSA(:,c) = exp(output.d11);

        end
    end
end