# Thesis_Trinity_Heringsp
README 

This document is meant to assist using the ‘sss_calc_and_complexity_th.py’, created by Pieter Herings. 

How to use the code 

**Step 1: Make sure the code is saved in the same directory as your MATLAB patient files **

**Step 2: The code only works on .mat (raw Shimmer 3)** datafiles with a specific name; “E001LS.mat”, “E002LS.mat” etc. Sean Read constructed these files in September 2020. Make sure your patient files are .mat and named correctly. 

**Step 3: Determine input variables of the get_sss() function: **

These values need to be determined for each patient file you try to run 

1. First input variable is the name of the patient file you want the average swing, stance and stride time from (for example “E001LS.mat”). 

2. cut-off value for the low pass Butterworth filter (value between 1 and 0), closer to 0 means that less high frequencies get passed, thus smoothening the signal (see Figure 1 and 2). If signal has a low hertz (~200), a higher value must be used for this cut-off value, because lower hertz means less datapoints and thus overall less noise.  (if 800 hertz use  ~0.005, if 200 hertz use ~ 0.05) 

3. multf_msv is a value which dictates where the max swing velocity peaks can be found in the signal (see Figure 3). The signal average and standard deviation, together with the multf_msv factor will dictate the lower limit of where max swing velocity peaks will be searched  (average + std * multf_msv). Lowering this value will thus increase your search area but also increase the change of finding false peaks caused by turning artefact or random noise. (common value ~ 1.8, but if no max swing velocities are found, try to lower the value) 

4. multf_tohs is a value which dictates where toe off and heel strike throughs will be searched (see Figure 4). Works same as multf_msv but than on the lower side of the signal. (common value ~ 0.3, if not a lot of toe off and heel strike throughs are found, lower this value). 

Before starting the code you should determine if you want to save the data you find in a csv file. Lines 322-331 in ‘swi_sta_str_calc_from_shimmer.py’ are used to save the data in a separate csv file. A ‘#’ before a line means that the code will ignore this, by removing the #’s in lines 322-331, the data will be saved. In line 325, after ‘open(‘ you can fill in the filename you want to use. Use line 327 only for the first time you use the code, it puts the variable names on top of the document, after running the first patient, put an # in front of line 327. 

 

Starting code  
type: get_sss(patienfilename, cutoff_butterworth, multf_msv, multf_tohs) under all the functions in the ‘swi_sta_str_calc_from_shimmer.py’ code.  

For example: get_sss('E002LS.mat', 0.005, 1.8, 0.5) 

Run the code 

Figure 1: Raw data of the angular velocity (column 7) of (‘E010LS.mat’) over time 

Shape 

Figure 2: Applying median filter & Butterworth filter (cut-off=0.005), respectively 

Shape 

Figure 3: Searching max swing velocity (msv) points (multf_msv = 1.8) 

ShapeText BoxShapeShape 

Figure 4: Searching throughs before and after msv peak and name toe off & heel strike, respectively (multf_tohs= 0.5) 

ShapeShapeText BoxShape 

 

Step 4: Determine input variables ‘complexity_index’ 

Matlab = filename I.e.  ‘E001LS.mat’ 

Hzlim = 200 Hz 

Samplength =  tol, maxsc) 

 

Common errors 

No peaks found 
You can recognize this error by the following message after running the code: 

Afbeelding met tekst

Automatisch gegenereerde beschrijving 

The reason this happens is because no msv peaks are found. Reasons for this could be that the data is still upside down, the signal is filtered with a too strong low pass filter or the search area for the peaks is higher than the max value of the signal itself.  
 
How to fix it 
If this happens at for instance patient ‘E010LS.mat’, add the number 10 to the swapbox in line 103. This will make sure the data gets flipped before running the rest of the code. 

Afbeelding met tekst

Automatisch gegenereerde beschrijving 

If the code still gives this error after you added the correct number to the swapbox: 

Try a higher value for cut-off (0.05 – 0.1); if the value is too low, only super low frequencies pass and the signal will more and more become a straight line (will thus not find peaks)   

Try a lower value for multf_msv; if the search area of the msv peaks is higher than the signal, no peaks will be found 

Not enough filtering 

When you see a graph like this, the swing stride and stance times that are calculated must not be used for analysis. It means the signal is not filtered strong enough resulting in loads of msv peaks found and because of that, toe off and heel strike peaks everywhere in the graph.  

Afbeelding met tekst, schrijfgerei, uitgelijnd, lijn

Automatisch gegenereerde beschrijving 

How to fix it 
Lower the cut-off value for the Butterworth filter (input variable) 

 
