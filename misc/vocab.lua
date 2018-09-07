itow = {'blue', 'red', 'green', 'violet', 'brown', 'white', 'cyan', 'salmon', 'yellow', 'silver', 'flat', 'stroke', 'style',
'Is', 'it', 'a', '?', 'in', 'the', 'image', 'digit', 'background', 'no', 'yes', '0', '1', '2','3', '4', '5',
'6', '7', '8', '9', '<START>', '<END>'}
wtoi = {}
for i = 1, #itow do  wtoi[itow[i]] = i end

itoanswer = {'yes', 'no'}
answertoi = {}
for i = 1, #itoanswer do  answertoi[itoanswer[i]] = i end

itocolors = {'blue', 'red', 'green', 'violet', 'brown'}
colorstoi = {}
for i = 1, #itocolors do  colorstoi[itocolors[i]] = i end

itobgcolors = {'white', 'cyan', 'salmon', 'yellow', 'silver'}
bgcolorstoi = {}
for i = 1, #itobgcolors do  bgcolorstoi[itobgcolors[i]] = i end

itostyles = {'flat', 'stroke'}
stylestoi = {}
for i = 1, #itostyles do  stylestoi[itostyles[i]] = i end