# oss1
Source: D:\Documents\project_root\OpenSARShip_1\S1A_IW_GRDH_1SDV_20150823T221233_20150823T221258_007396_00A2CA_54D0\ReadMe.pdf

 
 
 
 
DOCUMENT  
Instruction of OpenSARShip 
  
 
Prepared by 
Shanghai Key Laboratory of Intelligent  
Sensing and Recognition 
Version 
1.0 
Date 
23rd March 2017 
Distribution 
Open 
 
 
 
 
Source 
http://opensar.sjtu.edu.cn/. 


Instruction of OpenSARShip                 Shanghai Key Laboratory of Intelligent Sensing and Recognition 
 
Contents 
1. Introduction of ship chip .................................................................................. 1 
1.1 
Size of ship chip ........................................................................................... 1 
1.2 
Patch ............................................................................................................. 2 
1.2.1 
GRD ................................................................................................... 2 
1.2.2 
SLC .................................................................................................... 3 
1.3 
Patch_Cal ...................................................................................................... 3 
1.4 
Patch_RGB ................................................................................................... 4 
1.5 
Patch_Uint8 .................................................................................................. 5 
2. Introduction of Ship.xml .................................................................................. 5 
3. Introduction of Metedata.xml .......................................................................... 8 
Appendix ................................................................................................................ 8 
 
 
 
 


Instruction of OpenSARShip                 Shanghai Key Laboratory of Intelligent Sensing and Recognition 
1 
 
Instruction of OpenSARShip 
OpenSARShip is organized in different folders which corresponding to different scenes 
of images. In each folder, there contains four subfolders providing four formats of ship 
chips, one Ship.xml, one Metedata.xml, and one ReadMe.pdf. Figure 1 gives an 
example of the content for a specific image. 
 
Figure 1. Organization of OpenSARShip 
1. Introduction of ship chip 
For each Sentinel-1 SAR image, four subfolders provide four formats of ship chips: 
original data, calibrated data, visualized data in pseudo-color, and visualized data in 
grey scale, respectively, so as to satisfy different requirements of users. Ship chips are 
named by pixel coordinates for the convenience of retrieval. 
1.1 Size of ship chip 
The determination of the chip size is illustrated in Figure 2，where A(
1
1
,
x y ) and 
B(
2
2
,
x
y ) represent the pixel coordinate of ship head and ship tail (with 180 degree 
ambiguity), and point O indicates the center of the ship. Then we calculate the value of 
1
2
x
x

 and 
1
2
y
y

, respectively, and choose the larger value multiplied by 2 to get 
d . Finally, taking point O as the center of chip, we get a square chip with the length of 
2d .    


Instruction of OpenSARShip                 Shanghai Key Laboratory of Intelligent Sensing and Recognition 
2 
 
A(x1,y1)
B(x2,y2)
O
x(range)
y(azimuth)
d=2max{|x1-x2|,|y1-y2|}
 
Figure 2. The size of the ship chip 
1.2 Patch 
For Patch folder, filenames can be identified with the following code:  
NNN_xHHH_yVVV.tif, where NNN indicates the ship type (cargo, tanker, fishing, et 
al.) reported by AIS, HHH is the pixel coordinate in x-axis, and VVV is the pixel 
coordinate in y-axis.  
OpenSARShip database contains two modes of products, the first one is Sentienl-1 
interferometric wide (IW) ground range detected (GRD) product, the other one is 
Sentienl-1 interferometric wide (IW) single look complex (SLC) product. 
1.2.1 GRD 
The original data format (stored in single-precision values) of GRD ship chip is 
illustrated in Figure 3. Via inputting the ship chip in MATLAB, we get a 
2
m m

 
dimension matrix, in which ( , ,1)
x y
 indicates the pixel amplitude value of VH channel, 
and ( , ,2)
x y
 indicates the pixel amplitude value of VV channel. 


Instruction of OpenSARShip                 Shanghai Key Laboratory of Intelligent Sensing and Recognition 
3 
 
m
VV channel single amplitude
VH channel single amplitude
 
Figure 3. Data format of GRD ship chip 
1.2.2 SLC 
The original data format (stored in single-precision values) of SLC ship chip is 
illustrated in Figure 4. Via inputting the ship chip in MATLAB, we get a 
4
m m

 
dimension matrix, in which ( , ,1)
x y
 indicates the real part value of pixel for VH 
channel, ( , ,2)
x y
 indicates the imaginary part value of pixel for VH channel, ( , ,3)
x y
 
indicates the real part value of pixel for VV channel, and ( , ,4)
x y
 indicates the 
imaginary part value of pixel for VV channel. 
m
m
VV channel single
 real part
VV channel single 
imaginary part
VH channel single 
imaginary part
VH channel single
 real part
 
Figure 4. Data format of SLC ship chip 
1.3 Patch_Cal 
For Patch_Cal folder, filenames can be identified with the following code:  
NNN_xHHH_yVVV.tif, where NNN indicates the ship type (cargo, tanker, fishing, et 
al.) reported by AIS, HHH is the pixel coordinate in x-axis, and VVV is the pixel 
coordinate in y-axis. 
We apply SNAP3.0 Toolbox to get radiometric calibrated product. The radiometric 
calibrated data format of ship chip is illustrated in Figure 5. Via inputting the ship chip 
in MATLAB, we get a 
2
m m

 dimension matrix, in which ( , ,1)
x y
 indicates the 


Instruction of OpenSARShip                 Shanghai Key Laboratory of Intelligent Sensing and Recognition 
4 
 
pixel normalized radar cross section (NRCS) value 
0
 of VH channel, and ( , ,2)
x y
 
indicates the pixel NRCS value 
0
 of VV channel. 
m
VV channel Calibrate
VH channel Calibrate
m
 
Figure 5. Data format of calibrated ship chip 
1.4 Patch_RGB 
For Patch_RGB folder, filenames can be identified with the following code: 
NNN_xHHH_yVVV_pp.png, where NNN indicates the ship type (cargo, tanker, 
fishing, et al.) reported by AIS, HHH is the pixel coordinate in x-axis, VVV is the pixel 
coordinate in y-axis, and pp indicates the polarization (vh/vv) of ship chip. 
Based on the calibrated ship chip, we employ MATLAB ColorMap (jet) to visualize 
ship chip with Pseudo color. As shown in Figure 6, for each ship chip, the RGB 
visualized ship chip are provided with VH and VV channel, respectively. 
 
Figure 6. Data format for RGB ship chip 


Instruction of OpenSARShip                 Shanghai Key Laboratory of Intelligent Sensing and Recognition 
5 
 
1.5 Patch_Uint8 
For Patch_Uint8 folder, filenames can be identified with the following code:  
Visual_NNN_xHHH_yVVV_pp.png, where NNN indicates the ship type (cargo, tanker, 
fishing, et al.) reported by AIS, HHH is the pixel coordinate in x-axis, VVV is the pixel 
coordinate in y-axis, and pp indicates the polarization (vh/vv) of ship chip. 
Based on the original ship chip, we apply image enhanced by MATLAB programming 
to visualize ship chip with 0-255 grey scale (stored in unsigned 8-bit (1-byte) integers). 
As shown in Figure 7, for each ship chip, the grey scale visualized ship chip are 
provided with VH and VV channel, respectively. 
 
Figure 7. Data format of Uint8 ship chip 
2. Introduction of Ship.xml 
As shown in Figure 8, the detailed information of each ship chip containing AIS 
messages, SAR ship signatures, and elaborated messages provided by MarineTraffic 
Website, are listed in a XML file named ship.xml. Via the pixel coordinates shown in 
the filename of each chip, users can retrieve the corresponding information in the 
Ship.xml file. 


Instruction of OpenSARShip                 Shanghai Key Laboratory of Intelligent Sensing and Recognition 
6 
 
SAR 
information
AIS 
information
Match 
information
MarineTraffic 
information
Figure 8. Detailed information of each ship chip 
The detailed information of each item provided in Ship.xml is clarified in Table 1. Some 
specific AIS information is further illustrated in appendix.  


Instruction of OpenSARShip                 Shanghai Key Laboratory of Intelligent Sensing and Recognition 
7 
 
Table 1. Detailed information of each ship chip 
SAR information 
Name 
Explanation 
Unit 
Time_UTC 
SAR acquisition time 
-- 
Head_x 
Pixel coordinate of ship head 
 (with 180 degree ambiguity) 
-- 
Head_y 
Pixel coordinate of ship head 
(with 180 degree ambiguity) 
-- 
Tail_x 
Pixel coordinate of ship tail 
(with 180 degree ambiguity) 
-- 
Tail_y 
Pixel coordinate of ship tail 
(with 180 degree ambiguity) 
-- 
UpperLeft_x 
Pixel coordinate of the upper left point of chip 
-- 
UpperLeft_y 
Pixel coordinate of the upper left point of chip 
-- 
LowerRight_x 
Pixel coordinate of the lower right point of chip 
-- 
LowerRight_y 
Pixel coordinate of the lower right point of chip 
-- 
Center_x 
Pixel coordinate of ship center 
-- 
Center_y 
Pixel coordinate of ship center 
-- 
Center_longitude 
Longitude of ship center 
degree 
Center_latitude 
Latitude of ship center 
degree 
Manual_Length 
Ship length measured manually 
m 
North_Direction 
Ship direction respect to north 
 (clockwise, from 0 to 360) 
degree 
Chip_Direction 
Ship direction respect to azimuth 
(clockwise, from 0 to 180, and -180-0) 
degree 
Incidence 
Incidence angle of ship center 
degree 
Elevation 
Elevation angle of ship center 
degree 
Slant_Range_Time 
Slant range time of ship center 
s 
AIS information 
Time 
AIS time stamp 
-- 
MMSI 
MMSI of ship 
-- 
Nav_Status 
Navigation status (further detailed in appendix) 
-- 
Rot 
Rotation rate 
degree 
Sog 
Speed of ground 
10*knot 
Pos_Acc 
Accuracy 
-- 
Longitude 
Longitude of ship reported by AIS 
degree 
Latitude 
Latitude of ship reported by AIS 
degree 
Projected_Longitude 
Longitude of ship by AIS-projected 
degree 
Projected_Latitude 
Latitude of ship by AIS-projected 
degree 
Cog 
Course of ground 
10*degree 
True_Head 
Ship head direction 
degree 
Name 
Name of the ship 
-- 
Ship_Type 
Ship type (further detailed in appendix) 
-- 
IMO 
IMO of ship 
-- 
Callsign 
Callsign 
-- 
Length 
Length reported by AIS 
m 
Width 
Width reported by AIS 
m 
Draught 
Draught 
10*m 
Msgid 
AIS code (meaningless) 
-- 
Match information 
Length_error 
Length error between matched SAR and AIS 
m 
Length_error_rate 
Length error between matched SAR and AIS 
divided by AIS length 
-- 
Distance_error 
distance error between matched SAR and AIS 
m 
MarineTraffic information 


Instruction of OpenSARShip                 Shanghai Key Laboratory of Intelligent Sensing and Recognition 
8 
 
Length_overall 
Ship length 
m 
Breadth_extreme 
Ship breadth 
m 
Elaborated_type 
Ship type 
-- 
Gross_tonnage 
Ship weight 
t 
Deadweight 
Ship dead weight 
t 
Flag 
Nationality of the ship 
-- 
Year_built 
Ship built year 
-- 
3. Introduction of Metedata.xml 
Besides, a XML file of the original SAR image name Metedata.xml is also provided. 
This Metedata.xml is contained in the product which can be downloaded from Sentinel 
Scientific Data hub.  
Appendix 
A. The specific explanation of Nav_Status code (Table 2) 
Table 2. Specific explanation of Nav_Status code 
Code 
Explanation 
0 
Under Way 
1 
At Anchor 
2 
Not Under Command 
3 
Restricted Maneuverability 
4 
Constrained by her draught 
5 
Moored 
6 
Aground 
7 
Engaged in Fishing 
8 
Under Way Sailing 
B. The specific explanation of Ship type code (Table 3) 
Table 3. Specific explanation of ship type code 
Code 
Explanation 
20 
Wing in ground (WIG), all ships of this type 
21 
Wing in ground (WIG), Hazardous category A 
22 
Wing in ground (WIG), Hazardous category B 
23 
Wing in ground (WIG), Hazardous category C 


Instruction of OpenSARShip                 Shanghai Key Laboratory of Intelligent Sensing and Recognition 
9 
 
24 
Wing in ground (WIG), Hazardous category D 
25 
Wing in ground (WIG), Reserved for future use 
26 
Wing in ground (WIG), Reserved for future use 
27 
Wing in ground (WIG), Reserved for future use 
28 
Wing in ground (WIG), Reserved for future use 
29 
Wing in ground (WIG), Reserved for future use 
30 
Fishing 
31 
Towing 
32 
Towing: length exceeds 200m or breadth exceeds 25m 
33 
Dredging or underwater ops 
34 
Diving ops 
35 
Military Ops 
36 
Sailing 
37 
Pleasure Craft 
38 
Reserved 
39 
Reserved 
40 
High speed craft (HSC), all ships of this type 
41 
High speed craft (HSC), Hazardous category A 
42 
High speed craft (HSC), Hazardous category B 
43 
High speed craft (HSC), Hazardous category C 
44 
High speed craft (HSC), Hazardous category D 
45 
High speed craft (HSC), Reserved for future use 
46 
High speed craft (HSC), Reserved for future use 
47 
High speed craft (HSC), Reserved for future use 
48 
High speed craft (HSC), Reserved for future use 
49 
High speed craft (HSC), No additional information 
50 
Pilot Vessel 
51 
Search and Rescue vessel 
52 
Tug 
53 
Port Tender 
54 
Anti-pollution equipment 
55 
Law Enforcement 


Instruction of OpenSARShip                 Shanghai Key Laboratory of Intelligent Sensing and Recognition 
10 
 
56 
Spare - Local Vessel 
57 
Spare - Local Vessel 
58 
Medical Transport 
59 
Ship according to RR Resolution No. 18 
60 
Passenger, all ships of this type 
61 
Passenger, Hazardous category A 
62 
Passenger, Hazardous category B 
63 
Passenger, Hazardous category C 
64 
Passenger, Hazardous category D 
65 
Passenger, Reserved for future use 
66 
Passenger, Reserved for future use 
67 
Passenger, Reserved for future use 
68 
Passenger, Reserved for future use 
69 
Passenger, No additional information 
70 
Cargo, all ships of this type 
71 
Cargo, Hazardous category A 
72 
Cargo, Hazardous category B 
73 
Cargo, Hazardous category C 
74 
Cargo, Hazardous category D 
75 
Cargo, Reserved for future use 
76 
Cargo, Reserved for future use 
77 
Cargo, Reserved for future use 
78 
Cargo, Reserved for future use 
79 
Cargo, No additional information 
80 
Tanker, all ships of this type 
81 
Tanker, Hazardous category A 
82 
Tanker, Hazardous category B 
83 
Tanker, Hazardous category C 
84 
Tanker, Hazardous category D 
85 
Tanker, Reserved for future use 
86 
Tanker, Reserved for future use 
87 
Tanker, Reserved for future use 


Instruction of OpenSARShip                 Shanghai Key Laboratory of Intelligent Sensing and Recognition 
11 
 
88 
Tanker, Reserved for future use 
89 
Tanker, No additional information 
90 
Other Type, all ships of this type 
91 
Other Type, Hazardous category A 
92 
Other Type, Hazardous category B 
93 
Other Type, Hazardous category C 
94 
Other Type, Hazardous category D 
95 
Other Type, Reserved for future use 
96 
Other Type, Reserved for future use 
97 
Other Type, Reserved for future use 
98 
Other Type, Reserved for future use 
99 
Other Type, No additional information 
 


# oss2
No bundled ReadMe.pdf found.
