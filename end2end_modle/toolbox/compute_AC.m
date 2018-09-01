function ac=compute_AC(r_labels,labels)
%%% ac
        gnd=r_labels;
        res=labels';
        res = bestMap(gnd,res);
        ac = length(find(gnd == res))/length(gnd);

