//run("Scale...", "x=.15 y=.15 z=1.0 width=7298 height=3779 depth=3 interpolation=Bilinear average create");
//setBatchMode(true);
open("D:/User/locisuer/Desktop/stage_test/Fused-1.tif");
rename("corr_me");
run("Split Channels");
for(i=1;i<=3;i++)
	{
		title ="C"+d2s(i,0)+"-corr_me";
		selectWindow(title);
		getStatistics(area, mean, min, max, std, histogram);
		run("Duplicate...", "title=deno");
		run("Gaussian Blur...", "sigma=10");
		//imageCalculator("Divide create 32-bit", title,"deno");
		imageCalculator("Subtract create 32-bit", title,"deno");
		selectImage("Result of "+title);
		rename("new"+title);
		getStatistics(area, mean2, min, max, std2, histogram);
		//run("Multiply...", "value="+d2s(mean,0));
		//run("Divide...", "value="+d2s(std,0));
		run("Multiply...", "value="+d2s(mean/mean2,0));
		selectWindow("deno");
		close();
		//selectWindow(title);
		//close();
	}
run("Merge Channels...", "c1=newC1-corr_me c2=newC2-corr_me c3=newC3-corr_me create keep");
