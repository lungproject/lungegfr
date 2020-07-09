clear
readNPY('../data/label.npy');

filstruct = dir("../data/Image/CT/");
filstruct = filstruct(3:end);
for i=1:length(filstruct)
    str = filstruct(i).name;
    ctImage2 = double(imread(strcat("../data/Image/CT/",str)));
    petImage2 = double(imread(strcat("../data/Image/PET/",str)));
    
    petImage2 = normalizationimg(imresize(petImage2,[64 64],'nearest'));
    ctImage2 = normalizationimg(imresize(ctImage2,[64 64],'nearest'));

    petImage = (petImage2-mean(petImage2(:)))/(std(petImage2(:)));
    ctImage = (ctImage2-mean(ctImage2(:)))/(std(ctImage2(:)));
    fuseImage2 = petImage2+ctImage2;
    fuseImage = (fuseImage2-mean(fuseImage2(:)))/(std(fuseImage2(:)));
           
    cts(:,:,i) = ctImage;
    pets(:,:,i)=petImage;
    fuses(:,:,i)=fuseImage;
    
    label(i,1) = str2num(str(1));
end
cts = permute(cts,[3 1 2]);
pets = permute(pets,[3 1 2]);
fuses = permute(fuses,[3 1 2]);
 
writeNPY(cts, '../data/ttxpsamplect.npy');
writeNPY(pets, '../data/ttxpsamplepet.npy');
writeNPY(fuses, '../data/ttxpsamplefuse.npy');
