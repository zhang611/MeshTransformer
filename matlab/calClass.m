num = 215;
class = zeros(1,400);
for i=1:400
    if i>260 && i<281
        continue
    end
    labels = load(['E:\transf_mesh\PSB_new\seg_consistent\',num2str(i),'.seg']);
    class(i) = max(labels);
end
maxnum = 0;
for i=1:400
    if class(i)>maxnum
        maxnum = class(i);
    end
    if mod(i,20)==0
        str = [num2str(i/20),': ',num2str(maxnum+1)];
        disp(str)
        maxnum = 0;
    end
end

