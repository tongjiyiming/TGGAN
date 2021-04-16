x=[20,27,50,91,100]
y1=[2.216,
2.54,
5.308,
7.629,
9.056];
y2=[1103.7,
1543,
16320,
25343,
27569
];
y3=[16.32,
36.38,
58,
125,
148
];
plot(x,y1,x,y2/100,x,y3);
plot1 = plot(x,y1,x,y2/100,x,y3,'LineWidth',2.5,'Parent',axes1);
set(plot1(1),'DisplayName','GraphRNN','Marker','diamond');
set(plot1(2),'Marker','o');
set(plot1(3),'DisplayName','DSBM','Marker','square');

% Create ylabel
ylabel('Time(s)/per enpoch','FontWeight','bold');

% Create xlabel
xlabel('Nodes size');

% Create title
title('Runtime','FontWeight','bold');

box(axes1,'on');
% Set the remaining axes properties
set(axes1,'FontWeight','bold','TitleFontWeight','normal');
