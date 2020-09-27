function  DataRange = setplusone1(DataRange)
% CopyRight：xiezhh
for k = 1:DataRange.Count
   DataRange.Value{k} = DataRange.Value{k}+1;    % 将单元格取值加1
end