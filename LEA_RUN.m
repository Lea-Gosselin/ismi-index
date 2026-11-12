clc;clear all; close all;

cd("GitRepo")
addpath("ismi-index/input/")
addpath("x13Toolbox/seas/")

[data,txt,cell]=xlsread("EAIPC.xlsx");

date = cell2mat(cell(2:end,1)) + 693960;
date_matlab = datetime(date, 'ConvertFrom', 'datenum');

tstat = nan(1,size(data,2));
for c=1:size(data,2)
    if sum(~isnan(data(:,c)))>50
        output =x11([date_matlab,log(data(:,c))],12,'add');
        data_sa_l(:,c)=output.d11;
        data_sa = diff(data_sa_l(:,c));
        valeurs_pacf(:, c) = parcorr(data_sa);
        valeurs_pacf_nonsa(:, c) = parcorr(diff(log(data(:,c))));
    end
end

share_sa_EA = sum(abs(valeurs_pacf(13,:))>(1.98/sqrt(length(data))))/size(data,2);
share_nsa_EA = sum(abs(valeurs_pacf_nonsa(13,:))>(1.98/sqrt(length(data))))/size(data,2);

xlswrite('M:\Léa\Data\EAIPC.xlsx',num2cell(data_sa_l),'sa','B2')
%%
clc;clear all; close all;

[data,txt,cell]=xlsread("M:\Léa\Data\UKIPC.xlsx");
addpath('exe\')
addpath('seas\')

date = cell(2:end,1);
date_matlab = datenum(date,'dd/MM/YYYY');

tstat = nan(1,size(data,2));
for c=1:size(data,2)
    if sum(~isnan(data(:,c)))>50
        output =x11([date_matlab,log(data(:,c))],12,'add');
        data_sa_l(:,c)=output.d11;
        data_sa = diff(data_sa_l(:,c));
        valeurs_pacf(:, c) = parcorr(data_sa);
        valeurs_pacf_nonsa(:, c) = parcorr(diff(log(data(:,c))));

    end
end

share_sa_UK = sum(abs(valeurs_pacf(13,:))>(1.98/sqrt(length(data))))/size(data,2);
share_nsa_UK = sum(abs(valeurs_pacf_nonsa(13,:))>(1.98/sqrt(length(data))))/size(data,2);

xlswrite('M:\Léa\Data\UKIPC.xlsx',num2cell(data_sa_l),'sa','B2')
