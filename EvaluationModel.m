function eve = EvaluationModel(pre,ptrue,pro,cutoff)

   ind = find(~isnan(ptrue));
   pre = pre(ind,:);
   ptrue = ptrue(ind,:);

    pre = reshape(pre,[length(pre),1]);
    ptrue = reshape(ptrue,[length(ptrue),1]);


        if nargin==2
            cutoff=0.5;
            pro=0;
        elseif nargin==3
            cutoff=0.5;
        elseif nargin==4
            if cutoff==1
                 [~,cutoff,~,~] = AllAuc (pre,ptrue);     

            end
        elseif nargin<2
                ptrue = pre(:,end);
                pre = pre(:,1:end-1);
                [~,cutoff,~,~] = AllAuc (pre,ptrue); 
                pro=0;            
        end
       
        ytest = ptrue; ytest(ytest<=0)=-1;
        
        rangepr = unique(pre);
        if length(rangepr)==2
           yd = pre; yd(pre<=0)=-1;yd(pre>0)=1;
        else
            yd = pre; yd(pre<cutoff)=-1;yd(pre>=cutoff)=1;
        end
        
        
        
        

        
        record = zeros(length(ytest),2);
        record(:,1) = yd;
        record(:,2) = ytest;
        if ~pro
            try
                [~,~,~, auc]=perfcurve(record(:,2),record(:,1), 1);
            catch
                auc=-10;
            end
            
        else
            try
                [~,~,~, auc]=perfcurve(record(:,2),pre, 1);
            catch
                auc=-10;
            end
        end
     
        
        B= ytest-yd;
        tp = length(find((ytest==1)&(yd==1)));
        tptn=length(find(~B));
        tn = tptn-tp;
        fp = length(find(B==-2));
        fn = length(find(B==2));

        correctrate(1) = (tptn/(length(ytest)+ 10e-7))*100;
        correctrate(2) = tp/(tp+fp+10e-7);

        
        
        fpr = fp /(fp + tn + 10e-7);
        fnr = fn /(tp + fn + 10e-7); 
        tpr = tp /(tp + fn + 10e-7);
        tnr = tn / (tn + fp + 10e-7);

        accuracy = correctrate(:,1);
        precision = correctrate(:,2);
        Fscore = 2*tp/(2*tp+fp+fn+10e-7);
        
        eve = [accuracy  auc fpr fnr tpr tnr ];%tp/fp];% precision
