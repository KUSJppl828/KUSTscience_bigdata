function TextHandle = rotateticklabel(ha,tag,rot)

%   旋转坐标轴刻度标签的函数
%   ha   坐标系句柄（默认为当前坐标系）
%   tag  坐标轴标识字符串('X'或'Y')，默认旋转X轴标签
%   rot  旋转角度（单位：度）
%
%   Example:
%   x = 0:0.05:2*pi;
%   y = sin(x);
%   plot(x,y);
%   str = '这里是0|这里是1|这里是2|这里是3|这里是4|这里是5|这里是6|这里是7';
%   set(gca,'xtick',0:7,'xticklabel',str);
%
%   rotateticklabel(gca,'x',-30);
%
%   CopyRight：xiezhh（谢中华）

if ~ishandle(ha)
    warning('第一个输入参数应为坐标系句柄');
    return;
end

if ~strcmpi(get(ha,'type'),'axes')
    warning('第一个输入参数应为坐标系句柄');
    return;
end

if nargin == 1
    tag = 'X';
    rot = 0;
elseif nargin == 2
    if isnumeric(tag) && isscalar(tag)
        rot = tag;
        tag = 'X';
    elseif ischar(tag) && (strncmpi(tag,'x',1) || strncmpi(tag,'y',1))
        rot = 0;
    else
        warning('输入参数类型错误');
        return;
    end
else
    if ~isnumeric(rot) || ~isscalar(rot)
        warning('输入参数类型错误');
    end
    if ~ischar(tag) || (~strncmpi(tag,'x',1) && ~strncmpi(tag,'y',1))
        warning('输入参数类型错误');
    end
end

oldxticklabel = findobj('type','text','tag','oldxticklabel');
oldyticklabel = findobj('type','text','tag','oldyticklabel');
if strncmpi(tag,'x',1)
    if isempty(oldxticklabel)
        str = get(ha,'XTickLabel');
        x = get(ha,'XTick');
        yl = ylim(ha);
        set(ha,'XTickLabel',[]);
        y = zeros(size(x)) + yl(1) - range(yl)/70;
        TextHandle = text(x,y,str,'rotation',rot,...
            'Interpreter','none','tag','oldxticklabel');
    else
        set(oldxticklabel,'rotation',rot);
        TextHandle = oldxticklabel;
    end
else
    if isempty(oldyticklabel)
        str = get(ha,'YTickLabel');
        y = get(ha,'YTick');
        xl = xlim(ha);
        set(ha,'YTickLabel',[]);
        x = zeros(size(y)) + xl(1) - range(xl)/10;
        TextHandle = text(x,y,str,'rotation',rot,...
            'Interpreter','none','tag','oldyticklabel');
    else
        set(oldyticklabel,'rotation',rot);
        TextHandle = oldyticklabel;
    end
end

rot = mod(rot,360);
if rot>=0 && rot<180
   set(TextHandle,'HorizontalAlignment','right');
else
   set(TextHandle,'HorizontalAlignment','left');
end