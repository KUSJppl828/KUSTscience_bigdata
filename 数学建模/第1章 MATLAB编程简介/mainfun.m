function y = mainfun(x)
% 通过嵌套函数的方式编写函数
% CopyRight：xiezhh

y = subfun1(x) + subfun2(x);
    % 子函数1
    function y1 = subfun1(x1)
        y1 = (x1 + 1)^2;
    end
    % 子函数2
    function y2 = subfun2(x2)
        y2 = exp(x2);
    end
y = subfun3(y);
end
%%------------------------------------------
%%子函数3
%%------------------------------------------
function y = subfun3(x)
y = sqrt(x) - 1;
end