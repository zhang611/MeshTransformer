function SuccessRatio = Get_SuccessRatio(segResult,ground_truth,area)
%对比预测结果标签与ground-truth，得到匹配准确率
%segResult和area保持一直（都是行向量或都是列向量）
temp = double(string(segResult));
temp2 = temp == ground_truth;
temp3 = temp2 .* area;
SuccessRatio = sum(temp3) / sum(area);
