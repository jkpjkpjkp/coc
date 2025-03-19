# actor
a = 'wait, maybe one of the tool response is wrong. Let me cross-validate using other tools.'
aa = 'wait, maybe one of the tool response is wrong. Let me cross-validate using other parameters.'
b = 'i think i should get a general understanding of the image'
c = 'i think there are small details that are unnoticable by single tools.'
d = 'i think i will find the region of interest and zoom in. '
dd = 'i think i will crop-up the image and feed each crop to the tool. this way, i can get accurate results on finer details. '
e = 'i think i found an opportunity to synergize 2 tools.'
f = 'hey! i just had a very creative idea: '
g = 'i will re-use one of the tools, but with different parameters. '
h = 'i will use a tool i never used before. '
i = 'i recalled from my computer vision experience that something will be very useful here. i will implement it using the helpful python package '
j = 'let me think outside of the box here: what could be going on in the image that i currently overlooked. '
k = 'let us follow our plan and execute the next step: '
l = 'i think i have a solid and logical plan covering all corner cases: '
m = 'but is this response totally reliable? i think i can '
n = 'wait, this task is trickier than it seems. i neglected the possibility that '

prompts_1st_person = (a, aa, b, c, d, dd, e, f, g, h, i, j, k, l, m, n)

# 2nd person
a = 'Wait, maybe one of the tool responses is wrong. You should cross-validate using other tools.'
aa = 'Wait, maybe one of the tool responses is wrong. You should cross-validate using other parameters.'
b = 'You think you should get a general understanding of the image.'
c = 'You think there are small details that are unnoticeable by single tools.'
d = 'You think you will find the region of interest and zoom in.'
dd = 'You think you will crop up the image and feed each crop to the tool. This way, you can get accurate results on finer details.'
e = 'You think you found an opportunity to synergize 2 tools.'
f = 'Hey! You just had a very creative idea: '
g = 'You will re-use one of the tools, but with different parameters.'
h = 'You will use a tool you never used before.'
i = 'You recalled from your computer vision experience that something will be very useful here. You will implement it using the helpful Python package.'
j = 'Let you think outside of the box here: what could be going on in the image that you currently overlooked?'
k = 'Let you follow your plan and execute the next step: '
l = 'You think you have a solid and logical plan covering all corner cases: '
m = 'But is this response totally reliable? You think you can '
n = 'Wait, this task is trickier than it seems. You neglected the possibility that '

prompts_2nd_person = (a, aa, b, c, d, dd, e, f, g, h, i, j, k, l, m, n)

# critic
A = 'is there a step that simply assumes information without getting it from image? '
B = 'does the answer rests upon a single point of tool failure? '
C = 'is the deduction logically coherent? '