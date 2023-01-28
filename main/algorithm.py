# In this file there will be the main algorithm
import collections
import operator
import os
import time
import networkx as nx
import numpy as np
import scipy.io
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from karateclub import DeepWalk
from karateclub import node_embedding as ne
import math
import statistics
import argparse
import logging

logger = logging.getLogger(__name__)

adjnoun_top = list([18, 3, 52, 44, 105, 25, 51, 28, 10, 26])
lesmis_top = list([12, 49, 56, 59, 26, 28, 65, 76, 63, 66])
polbooks_top = list([9, 85, 73, 31, 13, 4, 74, 67, 12, 75])
dolphins_top = list([15, 46, 38, 34, 30, 41, 19, 21, 39, 51])
power_top = list([2555, 2554, 2576, 2562, 4346, 4396, 4337, 4333, 4402, 4382])
PPI_top = list([814, 589, 813, 3545, 287, 3601, 1031, 3334, 1017, 1221, 1959, 1207, 2725, 2925, 3133, 1431, 2587, 713, 1177, 867, 2302, 1114, 2776, 2467, 3758, 1731, 3291, 2807, 2424, 1545, 1509, 1457, 1243, 639, 719, 3587, 735, 1543, 904, 293, 2870, 3016, 3785, 2564, 571, 1508, 2147, 1417, 3468, 1311, 48, 3511, 3425, 2125, 3014, 1462, 1422, 977, 2773, 2761, 1237, 1562, 2317, 378, 46, 2778, 1238, 3306, 1296, 298, 1235, 533, 2151, 1245, 2536, 361, 318, 22, 3413, 2333, 841, 2273, 651, 1793, 829, 3583, 2633, 2554, 2187, 131, 1383, 3563, 2203, 1097, 3012, 2522, 2661, 2357, 558, 2263, 2243, 3581, 1600, 2240, 944, 3305, 1316, 1663, 134, 409, 3464, 1066, 2063, 3656, 3379, 141, 1461, 3568, 2277, 3612, 2481, 1575, 706, 1920, 3770, 3033, 2625, 1599, 3178, 667, 3529, 399, 896, 3208, 1391, 561, 15, 3690, 1110, 606, 3555, 1005, 1507, 118, 1533, 2443, 1071, 227, 1491, 3298, 1914, 2211, 1587, 1247, 882, 1046, 535, 2472, 1636, 1045, 868, 438, 3500, 3176, 2468, 2311, 2976, 13, 2812, 1890, 1517, 660, 2863, 3556, 387, 590, 432, 240, 335, 2762, 2154, 3270, 1530, 481, 1410, 1070, 477, 441, 3213, 3693, 997, 1621, 1330, 121, 2321, 3385, 1023, 3807, 3534, 178, 2672, 2237, 3751, 670, 2442, 595, 3207, 291, 602, 104, 3610, 2139, 2136, 1250, 2706, 1551, 3851, 2811, 2766, 3231, 982, 1350, 1329, 3102, 2471, 3363, 1586, 766, 2816, 3634, 3705, 2992, 1548, 1848, 1012, 1830, 466, 2396, 1817, 823, 644, 3153, 1733, 1326, 278, 1063, 140, 3054, 2635, 2604, 1603, 317, 1076, 2738, 270, 2185, 1809, 3808, 3131, 55, 3019, 2019, 143, 3837, 2158, 1133, 851, 3008, 582, 3452, 2846, 2637, 2128, 1236, 1096, 2703, 1825, 1242, 3387, 385, 866, 731, 2411, 3418, 144, 1065, 1016, 2475, 2363, 1200, 1655, 1211, 1963, 1819, 693, 1093, 2864, 1885, 2854, 271, 1013, 2702, 3220, 1552, 835, 1789, 3559, 3299, 2624, 274, 2887, 2504, 1429, 3651, 3172, 2167, 3662, 730, 2682, 2179, 1240, 2801, 1241, 3092, 2209, 1402, 603, 1227, 826, 3301, 1067, 3843, 579, 126, 2403, 1822, 1438, 3433, 2662, 1547, 431, 3692, 1611, 1091, 1059, 2611, 210, 1239, 856, 549, 2517, 319, 3173, 3825, 1683, 1467, 3672, 3188, 2792, 2107, 2079, 1893, 717, 3804, 1130, 369, 626, 3069, 815, 44, 3794, 2291, 1836, 312, 1019, 3198, 1069, 2309, 1727, 3409, 2789, 1230, 954, 629, 2537, 1831, 517, 299, 3125, 2921, 1589, 1201, 1092, 2600, 1724, 540, 1941, 1, 2072, 2253, 1979, 236, 3731, 2501, 739, 908, 3838, 2407, 1937, 903, 625, 125, 636, 3707, 3567, 3020, 3733, 3582, 246, 198, 320, 721, 3302, 2078, 1946, 1634, 1105, 3079, 1996, 1293, 3847, 676, 605, 3470, 1281, 1220, 203, 191, 3354, 786, 500, 421, 3373, 3370, 2575, 1205, 1058, 244, 204, 3494, 3088, 830, 570, 3708, 2379, 1264, 675, 3368, 1618, 1534, 2524, 1328, 3605, 2462, 1834, 1500, 60, 3495, 3224, 1687, 2429, 3021, 642, 233, 192, 2799, 1947, 1182, 202, 3314, 2132, 1477, 1365, 2361, 1382, 2168, 1370, 707, 1930, 3149, 2916, 1094, 157, 2979, 3300, 2258, 504, 2632, 2631, 1449, 3236, 2601, 2372, 2230, 2998, 2936, 2754, 749, 167, 1840, 1337, 163, 3501, 2313, 1886, 1863, 1167, 47, 2645, 2181, 1248, 758, 3238, 3136, 577, 1743, 1095, 627, 254, 2705, 2109, 1043, 3289, 416, 3117, 2493, 3565, 2033, 3646, 1671, 655, 3777, 2897, 2802, 2482, 2404, 2214, 273, 3736, 824, 3781, 2654, 2437, 1444, 1398, 275, 1333, 695, 2967, 940, 375, 181, 3611, 2141, 1124, 1246, 433, 3496, 1347, 906, 2840, 562, 3566, 3408, 2541, 1132, 1816, 3401, 3161, 3737, 3553, 2988, 2298, 3399, 1346, 734, 350, 2508, 2466, 2283, 1653, 3691, 3251, 2445, 3432, 3076, 2873, 1489, 797, 4, 3653, 3647, 779, 573, 521, 2463, 2064, 1922, 1759, 964, 524, 1737, 342, 3170, 365, 3684, 2881, 2269, 2459, 754, 635, 3430, 2731, 909, 3281, 1741, 724, 3492, 3483, 3089, 2315, 2199, 1394, 428, 3722, 3554, 1938, 1927, 1785, 1765, 228, 2221, 883, 1413, 808, 3179, 3025, 1362, 1129, 263, 1873, 1299, 183, 3655, 3378, 1732, 1021, 1372, 54, 3209, 2973, 596, 1772, 1327, 748, 2500, 1131, 3619, 3229, 948, 759, 2910, 2075, 2049, 1157, 3211, 2183, 1771, 1318, 853, 740, 471, 1797, 2834, 2217, 1594, 1224, 918, 697, 1899, 1709, 979, 3416, 3062, 2768, 2210, 3560, 2934, 2788, 1805, 3740, 3652, 2991, 2558, 1627, 939, 337, 3226, 3061, 3057, 678, 19, 3682, 3482, 3405, 3047, 2614, 2503, 2279, 2159, 1815, 1504, 414, 313, 3137, 2880, 2839, 2717, 1851, 887, 12, 3085, 2970, 3346, 1173, 3249, 2819, 2793, 2656, 1428, 722, 3713, 3539, 2469, 1989, 1951, 1866, 963, 2441, 2004, 1060, 1003, 3809, 3317, 2899, 3023, 3002, 2228, 1609, 1494, 3734, 1736, 1389, 447, 3813, 2035, 1940, 3124, 2848, 2775, 1864, 3623, 2092, 1193, 547, 302, 3792, 3739, 3009, 2244, 3438, 2295, 1780, 2586, 2427, 2406, 2362, 3123, 2980, 1616, 3485, 2066, 1573, 286, 2577, 2166, 2098, 647, 90, 3044, 2995, 1901, 1577, 1395, 1344, 3806, 2275, 2137, 1134, 548, 2233, 1967, 1529, 1433, 1339, 2370, 1839, 1644, 472, 3669, 1356, 765, 285, 49, 2574, 1479, 1266, 1081, 64, 3497, 3077, 799, 216, 2962, 2160, 981, 3752, 3488, 2397, 1931, 1511, 1282, 1223, 897, 3441, 2557, 2545, 2245, 1267, 498, 2330, 2272, 1778, 718, 27, 3361, 2579, 2365, 2268, 708, 136, 2419, 2398, 419, 3038, 2584, 2552, 1837, 1550, 917, 888, 3628, 3388, 3171, 1879, 1774, 600, 220, 188, 3630, 2708, 2655, 1917, 3183, 2735, 1776, 686, 3263, 3100, 2919, 1806, 1652, 847, 3158, 1824, 1033, 953, 553, 545, 3264, 3203, 1432, 328, 2074, 757, 3535, 3421, 1305, 753, 3451, 3140, 2498, 366, 3598, 2942, 2885, 2737, 2602, 1560, 1506, 1485, 1291, 902, 828, 534, 156, 3412, 2866, 2287, 2095, 1855, 1832, 1039, 910, 3310, 2997, 1576, 550, 420, 326, 166, 3474, 3350, 2178, 1971, 1787, 1740, 1278, 818, 3312, 2316, 2201, 1792, 1643, 1602, 1006, 720, 265, 2889, 850, 205, 133, 114, 3374, 2824, 1146, 776, 376, 122, 3850, 3774, 2827, 2394, 1756, 1538, 744, 347, 223, 3475, 3414, 2926, 2478, 2352, 1399, 1106, 237, 2818, 2639, 2518, 2499, 1966, 1881, 1623, 3832, 3260, 3129, 1415, 554, 2673, 1334])
blogcatalog_top = list([232, 8269, 1680, 9525, 4651, 6958, 4983, 9843, 4838, 4373, 3261, 458, 1008, 8972, 3338, 1326, 4946, 175, 8475, 3406, 448, 2998, 6057, 738, 5624, 10139, 8156, 2326, 1985, 4668, 1225, 3396, 7805, 1000, 1195, 9918, 6872, 3101, 644, 10, 3769, 7097, 745, 4996, 3607, 5555, 725, 14, 1151, 9480, 3121, 5029, 6034, 9357, 2623, 6070, 9996, 6427, 555, 577, 7060, 3560, 6633, 9943, 9122, 225, 7988, 7532, 806, 8774, 6949, 1931, 6747, 3197, 9419, 8978, 6910, 3527, 614, 4583, 1401, 829, 5258, 4930, 5767, 9820, 5844, 5680, 3657, 666, 445, 7450, 7372, 8524, 4455, 2627, 1864, 9516, 5284, 6104, 1451, 5730, 532, 7681, 7495, 3829, 2368, 2240, 1409, 6389, 6372, 9708, 268, 10287, 2624, 3275, 1886, 1245, 910, 5771, 877, 9187, 8858, 4707, 4214, 1702, 6114, 2902, 2312, 1283, 9185, 275, 6396, 3425, 6090, 8867, 6107, 6072, 4647, 4255, 287, 8755, 3055, 8943, 2215, 7701, 6727, 6000, 4804, 1412, 6685, 6186, 711, 8927, 6434, 4621, 3223, 1046, 997, 5230, 4732, 2350, 9875, 9104, 6454, 3525, 3461, 1433, 1373, 1145, 8588, 8366, 6569, 5666, 4134, 3295, 2954, 1108, 393, 290, 5067, 2996, 4489, 3754, 3907, 2774, 2785, 2531, 1730, 8887, 4512, 2795, 342, 8139, 8083, 6758, 6468, 414, 2155, 1977, 525, 8323, 6812, 3926, 3858, 3750, 2494, 661, 8647, 3689, 2511, 1841, 1282, 363, 8627, 6871, 4789, 2976, 499, 6592, 2860, 135, 9058, 6859, 4087, 3537, 2695, 702, 173, 10268, 8179, 4842, 2204, 9968, 4161, 3873, 3764, 282, 8893, 6956, 4068, 3796, 2637, 3138, 797, 160, 8529, 7380, 5566, 3857, 3606, 3400, 1442, 1056, 8301, 6550, 6140, 3578, 2230, 1810, 6359, 6128, 5322, 5631, 5255, 1366, 6385, 3333, 3268, 2937, 2833, 1285, 858, 8560, 5084, 3113, 8796, 6175, 5150, 1291, 1217, 583, 91, 8637, 7926, 6877, 6707, 6331, 5545, 4547, 3497, 3450, 950, 857, 640, 543, 399, 7421, 2084, 190, 40, 7841, 7304, 5687, 4994, 4309, 1823, 1473, 1135, 9372, 5003, 2649, 2520, 207, 9652, 9564, 8440, 8233, 7365, 6340, 6054, 3489, 2992, 2907, 1309, 646, 245, 8938, 8054, 6661, 6304, 5151, 2731, 2442, 2069, 1951, 7085, 6113, 4496, 3154, 3091, 925, 340, 7223, 6714, 5716, 5489, 5171, 4153, 3385, 2108, 1272, 650, 429, 228, 6174, 5465, 5004, 3992, 9215, 8247, 6687, 5606, 5068, 4858, 4795, 3082, 2596, 2202, 1903, 942, 8816, 8790, 6767, 5376, 1842, 1664, 754, 505, 12, 8451, 7894, 6748, 5215, 4766, 4500, 2516, 2076, 1434, 964, 15, 8602, 8346, 7575, 7414, 6647, 4664, 2940, 2818, 2664, 1523, 394, 9963, 8889, 8334, 6640, 6539, 6071, 5428, 5152, 4101, 3701, 2453, 1705, 969, 883, 8638, 8091, 7130, 6763, 6512, 5996, 4384, 4348, 4230, 4131, 2610, 2447, 2184, 2065, 1346, 918, 586, 209, 10248, 7921, 6724, 6565, 6415, 5493, 5449, 5264, 4108, 4079, 3851, 1402, 352, 27, 8420, 7200, 6906, 5568, 4907, 3733, 3087, 3036, 2226, 2032, 1566, 1330, 970, 9295, 8386, 8107, 4593, 4560, 4263, 1876, 1875, 678, 392, 82, 10191, 6546, 6112, 5849, 5543, 5203, 3550, 3239, 2802, 1643, 473, 346, 243, 9911, 9138, 8309, 7247, 4826, 3878, 3476, 2621, 2099, 802, 568, 447, 364, 71, 9435, 9259, 7359, 3360, 3322, 2831, 2337, 2195, 1295, 1104, 730, 527, 10224, 8158, 7374, 6638, 5885, 5604, 4993, 3559, 3409, 2536, 2199, 2034, 549, 9527, 7551, 7307, 7234, 6016, 5975, 5002, 4506, 3742, 3366, 2611, 2550, 2415, 1935, 1739, 1074, 811, 8500, 7131, 6272, 5392, 4757, 4114, 3625, 3258, 3050, 2357, 2048, 1733, 573, 483, 475, 9026, 8826, 8553, 5373, 4565, 3962, 3334, 2597, 1455, 570, 194, 85, 81, 7970, 7227, 6316, 6235, 5742, 5345, 4678, 3935, 3414, 2823, 2380, 2279, 2260, 2036, 1845, 1704, 1155, 204, 8497, 6618, 6456, 4835, 4724, 4671, 4612, 4530, 4004, 2799, 2341, 2185, 1635, 1594, 1533, 1190, 347, 48, 9181, 8028, 7360, 6211, 5930, 5235, 2482, 1416, 1385, 833, 8100, 8053, 7026, 6630, 5924, 5660, 5424, 5190, 4444, 3819, 3741, 3693, 3257, 3094, 3072, 2043, 1821, 356, 218, 0, 9832, 9478, 9152, 9097, 7292, 7204, 5721, 5193, 4245, 3351, 3265, 3212, 2651, 1737, 1576, 1568, 995, 605, 10296, 9692, 8395, 7636, 6976, 6286, 4960, 4328, 3656, 3146, 3145, 774, 638, 373, 233, 9644, 9444, 7839, 7191, 6222, 5162, 4598, 4541, 3973, 3856, 3140, 1912, 1494, 1234, 985, 61, 35, 9822, 8180, 7017, 6831, 6815, 5905, 5835, 5098, 5090, 4827, 4599, 4503, 3776, 3740, 3192, 2914, 2697, 2303, 1779, 762, 518, 8916, 6885, 6804, 5221, 5189, 5050, 5008, 3643, 3448, 3059, 2882, 1890, 1806, 1683, 541, 294, 9886, 7375, 7064, 6899, 6260, 6065, 6003, 5704, 2391, 1724, 47, 32, 9829, 9524, 9171, 8862, 7217, 7009, 6829, 6806, 6481, 5636, 4195, 3902, 2952, 1516, 998, 712, 348, 322, 8556, 8479, 7420, 5940, 4870, 4158, 4021, 3602, 2331, 2057, 1873, 1867, 1454, 775, 341, 10040, 9704, 9586, 9077, 8680, 7838, 7128, 6726, 6704, 5843, 4722, 4535, 4413, 4379, 4100, 3649, 3526, 1934, 1493, 1463, 9503, 9349, 8382, 7579, 6369, 5926, 5852, 5484, 5375, 5220, 4813, 4389, 3723, 2225, 1982, 1754, 1637, 791, 619, 9218, 7925, 7561, 6985, 6914, 6562, 6354, 5594, 5584, 5111, 4961, 4850, 4414, 4403, 4218, 4135, 4124, 3465, 2861, 2773, 2557, 1159, 849, 827, 212, 183, 8932, 8735, 8221, 8094, 7564, 6843, 5699, 5633, 5504, 5497, 5425, 5138, 4495, 4150, 3893, 3222, 1577, 1380, 1226, 1166, 1150, 481, 64, 9847, 9482, 7500, 7321, 6916, 5177, 5141, 5100, 4901, 4685, 4287, 4237, 3229, 2665, 2636, 2209, 2117, 8991, 8840, 8631, 8603, 8062, 7964, 7929, 7436, 6317, 5784, 4190, 4023, 3416, 2186, 2114, 2061, 2030, 2028, 1999, 1458, 1320, 1219, 10028, 9889, 9397, 8698, 8314, 8299, 8238, 6757, 5663, 5240, 5185, 5183, 5034, 4427, 4406, 4349, 3906, 3488, 2721, 2187, 1897, 1162, 1109, 837, 799, 10012, 9675, 9673, 9439, 9091, 8437, 8373, 8295, 6691, 6612, 6577, 5783, 5635, 5628, 5226, 4744, 4490, 4419, 4416, 4277, 3724, 3604, 3235, 2173, 1794, 1781, 606, 372, 127, 38, 10067, 9848, 9793, 9161, 8762, 8456, 7794, 7624, 7589, 6641, 6544, 6410, 5502, 4836, 4009, 3822, 3725, 3502, 3181, 2810, 2535, 2394, 2091, 1761, 1388, 1341, 990, 627, 376, 230, 66, 9987, 9354, 9262, 8784, 8767, 8416, 7753, 7623, 7325, 5944, 5873, 5197, 4894, 4313, 4146, 4049, 3797, 3712, 2987, 2352])
wikipedia_top = list([2, 1, 3, 6, 5, 4, 11, 8, 7, 13, 9, 16, 14, 12, 24, 10, 89, 22, 80, 17, 19, 25, 2406, 15, 40, 20, 35, 28, 619, 1171, 977, 30, 57, 18, 4565, 2083, 187, 473, 27, 3285, 132, 33, 156, 865, 26, 95, 37, 36, 862, 239, 21, 4602, 4190, 303, 123, 112, 92, 32, 23, 499, 1334, 150, 3057, 254, 1980, 1640, 786, 596, 103, 3784, 815, 86, 45, 1061, 983, 736, 726, 235, 101, 73, 31, 4124, 567, 333, 58, 1955, 668, 510, 396, 346, 2755, 2443, 1082, 914, 377, 47, 1907, 1407, 512, 356, 169, 130, 97, 52, 51, 3699, 2696, 1256, 569, 370, 336, 91, 29, 2203, 1219, 1111, 65, 42, 1624, 1101, 717, 265, 75, 66, 2413, 2055, 1932, 1926, 1140, 719, 555, 108, 43, 2865, 2816, 1550, 490, 330, 266, 3685, 1838, 710, 620, 69, 64, 2910, 2768, 2303, 789, 528, 287, 76, 39, 3853, 3768, 3710, 1650, 1481, 1238, 527, 350, 328, 268, 233, 70, 50, 44, 38, 3828, 3052, 2906, 2344, 2129, 1947, 1491, 1371, 242, 188, 172, 158, 74, 62, 4714, 2898, 2821, 2726, 2629, 2272, 2034, 1515, 1429, 1015, 641, 375, 315, 297, 223, 4168, 4086, 2665, 1982, 1887, 1599, 1471, 818, 768, 661, 340, 295, 250, 3978, 3370, 3294, 2462, 1933, 1871, 1390, 1227, 1208, 1056, 613, 479, 421, 284, 207, 142, 78, 72, 3935, 3070, 2973, 2882, 2846, 2174, 2144, 1764, 1756, 1742, 1136, 491, 363, 314, 197, 168, 139, 129, 116, 90, 59, 4189, 4090, 3731, 3573, 2920, 2764, 2319, 1749, 1000, 933, 774, 352, 304, 237, 205, 143, 93, 82, 41, 34, 4508, 4506, 4447, 4135, 3991, 3502, 3346, 2737, 2535, 2353, 1807, 1723, 1397, 1086, 898, 653, 598, 456, 199, 4394, 3665, 2365, 1866, 990, 838, 741, 732, 585, 536, 438, 388, 286, 190, 134, 100, 85, 48, 4533, 4069, 3595, 3251, 2505, 2079, 1882, 1842, 1763, 1379, 1217, 1201, 1166, 900, 867, 841, 723, 694, 603, 515, 423, 236, 220, 81, 4500, 4443, 3892, 3805, 3453, 2986, 2732, 2628, 2604, 2440, 2414, 2173, 1264, 1239, 1190, 1188, 1035, 997, 866, 857, 667, 616, 485, 344, 322, 316, 283, 262, 259, 160, 157, 131, 127, 61, 4436, 4043, 3505, 3445, 3236, 3013, 2875, 2123, 1922, 1921, 1625, 1432, 1430, 1392, 1367, 1314, 1065, 890, 677, 670, 625, 606, 570, 502, 437, 429, 386, 368, 305, 248, 177, 84, 83, 67, 4253, 3902, 3354, 3233, 2609, 1772, 1636, 1338, 1244, 1150, 1034, 919, 738, 608, 574, 342, 306, 232, 202, 186, 147, 140, 102, 3956, 3568, 3367, 3310, 3169, 3029, 3016, 2735, 2270, 2227, 2137, 1777, 1702, 1457, 1452, 1127, 1108, 1041, 994, 976, 973, 910, 816, 586, 546, 506, 380, 372, 243, 229, 226, 179, 159, 138, 133, 122, 98, 4155, 3879, 3803, 3022, 2947, 2651, 2329, 2182, 2065, 1718, 1448, 1106, 884, 880, 709, 599, 545, 532, 509, 457, 430, 337, 274, 166, 163, 145, 111, 110, 107, 77, 63, 4226, 4051, 3742, 3702, 3683, 3626, 3602, 3399, 3307, 3145, 2926, 2461, 2282, 2224, 2102, 1902, 1674, 1473, 1431, 1250, 1210, 1149, 1096, 1055, 1002, 962, 943, 821, 662, 643, 597, 454, 282, 280, 183, 151, 149, 87, 49, 4624, 4458, 4317, 4235, 4004, 3839, 3830, 3601, 3230, 3191, 3125, 2936, 2694, 2682, 2596, 2591, 2489, 2470, 2380, 2231, 2210, 2199, 1958, 1862, 1671, 1585, 1551, 1544, 1428, 1260, 1110, 1044, 961, 935, 704, 692, 675, 650, 447, 446, 428, 409, 405, 389, 382, 379, 373, 293, 175, 152, 94, 53, 46, 4753, 4507, 4143, 4007, 3875, 3869, 3741, 3474, 3276, 3151, 3083, 2863, 2742, 2570, 2502, 2340, 2167, 1946, 1900, 1859, 1818, 1816, 1810, 1809, 1632, 1631, 1455, 1415, 1414, 1247, 1187, 1103, 1049, 980, 854, 762, 718, 681, 671, 636, 556, 471, 441, 398, 281, 247, 218, 194, 167, 118, 56, 4612, 4605, 4574, 4566, 4521, 4509, 4122, 4070, 3981, 3961, 3580, 3451, 3412, 3101, 3011, 2895, 2867, 2714, 2592, 2464, 2384, 2149, 2115, 2044, 1918, 1753, 1703, 1511, 1424, 1387, 1307, 1300, 1168, 1133, 1059, 1028, 985, 847, 827, 743, 721, 712, 444, 395, 279, 214, 212, 114, 104, 88, 68, 4698, 4678, 4415, 4330, 4175, 4138, 3857, 3824, 3775, 3727, 3644, 3406, 3345, 3334, 3227, 3168, 3165, 3092, 3037, 3017, 2999, 2889, 2834, 2818, 2658, 2512, 2393, 2375, 2370, 2263, 2252, 2111, 2047, 2029, 1899, 1822, 1775, 1681, 1664, 1597, 1588, 1513, 1507, 1365, 1186, 995, 939, 877, 869, 836, 825, 759, 727, 691, 591, 526, 495, 462, 415, 354, 326, 313, 302, 300, 148, 119, 55, 4694, 4664, 4452, 4268, 4059, 3969, 3867, 3822, 3592, 3351, 3289, 3238, 3156, 3147, 3142, 2904, 2744, 2743, 2689, 2666, 2631, 2577, 2474, 2444, 2430, 2397, 2273, 2269, 2162, 2121, 2095, 2046, 1994, 1937, 1879, 1856, 1680, 1629, 1626, 1622, 1610, 1571, 1566, 1499, 1477, 1362, 1325, 1282, 1277, 1252, 1246, 1232, 1119, 1102, 1042, 954, 901, 858, 796, 711, 656, 614, 588, 562, 504, 439, 362, 327, 256, 206, 174, 128, 4771, 4733, 4579, 4474, 4382, 4357, 4343, 4272, 4197, 4137, 4024, 3938, 3886, 3837, 3724, 3697, 3649, 3628, 3607, 3562, 3388, 3296, 3273, 3216, 3207, 3193, 3127, 3081, 2900, 2800, 2738, 2729, 2719, 2420, 2391, 2258, 2238, 2213, 2197, 2180, 2069, 2052, 2027, 2023, 2012, 1990, 1984, 1979, 1957, 1952, 1847, 1731, 1653, 1648, 1643, 1620, 1595, 1591, 1558, 1296, 1287, 1259, 1159, 1137, 1084, 1079, 1077, 899, 874, 852, 802, 792, 687, 631, 617, 589, 550, 549, 513, 496, 478, 474, 448, 390, 320, 307, 298, 272, 253, 246, 195, 144, 137, 136, 105, 99, 60, 54, 4743, 4550, 4503, 4494, 4336, 3845, 3811, 3792, 3790, 3720, 3655, 3539, 3467, 3429, 3353, 3271, 3080, 2750, 2691, 2645, 2600, 2576, 2568, 2534, 2321, 2184, 2171, 2127, 2063, 2008, 1956, 1765, 1758, 1661, 1627, 1557, 1545, 1528, 1357, 1229, 1215, 1174, 1152, 1126, 1076, 1075, 1062, 1053, 996, 972, 913, 889, 875, 822, 800, 772, 769, 760, 611, 592, 417, 399, 351, 334, 299, 260, 230, 162, 125, 117, 115, 106, 4709, 4668, 4596, 4554, 4478, 4366, 4211, 4209])
# ---------------------------------------------------------------
#
# Function to load the specified dataset and place it in a
# NetworkX Graph structure, removing self loops and isolated
# nodes from the graph.
# @param: file - the name of the input file
# @param: variable_name - data structure
# @return: G - NetworkX graph of the dataset without self loops
#              and isolated nodes
# ---------------------------------------------------------------
def load_graph(file):
    os.chdir("../datasets")
    if ".mat" in file:
        raw_data = scipy.io.loadmat(file, squeeze_me=True)
        variable_name = "network"
        data = raw_data[variable_name]
        logger.info("loading mat file %s", file)
        G = nx.to_networkx_graph(data, create_using=nx.Graph, multigraph_input=False)
    else:
        logger.info("loading gml file %s", file)
        G = nx.read_gml(file, label="id")
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    mapping = dict(zip(G, range(0, len(list(G.nodes())))))
    G = nx.relabel_nodes(G, mapping)
    return G


# ---------------------------------------------------------------
#
# Function to load the specified dataset and place it in a
# NetworkX Graph structure.
# @param: file - the name of the input file
# @param: variable_name - data structure
# @return: G - NetworkX graph of the dataset
# ---------------------------------------------------------------
def load_default_graph(file):
    os.chdir("../datasets")
    if ".mat" in file:
        raw_data = scipy.io.loadmat(file, squeeze_me=True)
        variable_name = "network"
        data = raw_data[variable_name]
        logger.info("loading mat file %s", file)
        G = nx.to_networkx_graph(data, create_using=nx.Graph, multigraph_input=False)
    else:
        logger.info("loading gml file %s", file)
        G = nx.read_gml(file, label="id")
    return G

# ---------------------------------------------------------------
#
# Function to find the neighborhood of a node, which can be of
# different level specified by the cutoff, in the graph passed.
# @param: G - the graph
# @param: node - the node of which we have to find neighbors
# @param: level - the level specifying the cutoff on path length
#                 from initial node
# @return: res - all the nodes being in a level-neighborhood (does
#                nor return the initial node in the neighborhood)
# ---------------------------------------------------------------
def neighborhood(G, node, level):
    res = nx.single_source_dijkstra_path_length(G, node, cutoff=level)
    del res[node]
    return res

# does return the initial node as part of the neighborhood
def neighborhood_including_node(G, node, level):
    res = nx.single_source_dijkstra_path_length(G, node, cutoff=level)
    return res


# ---------------------------------------------------------------
#
# Loading graph and its DeepWalk embedding
# @param: file - the name of the input file
# @param: variable_name - data structure
# @return: G, embedding - NetworkX graph of the network in the
#                         input file and its embedding (DeepWalk)
# ---------------------------------------------------------------
def deepwalk_embedding(file):
    G = load_default_graph(file)
    N = G.number_of_nodes()
    r = math.ceil(N/2)
    dim = min(128, r)
    logger.info("DeepWalk embedding of %s network", file)
    start_time = time.time()
    model = DeepWalk(walk_length=100, dimensions=dim, window_size=10)
    model.fit(G)
    embedding = model.get_embedding()
    print("------- %s seconds ---------" % (time.time() - start_time))
    return G, embedding

# ---------------------------------------------------------------
#
# Computing euclidean distance between all embedded nodes
# @param: embedding - embedding of nodes
# @return: distances - structure containing the euclidean
#                      distances between each pair of embedded
#                      nodes
# ---------------------------------------------------------------
def precomputing_euclidean(embedding):
    start = time.time()
    distances = dict()
    for i in range(len(embedding)):
        for j in range(len(embedding)):
            alias = (i,j)
            distances[alias] = -np.linalg.norm(embedding[i]-embedding[j])
    logger.info("Precomputing euclidean distances between node embeddings in %s seconds" % (time.time()-start))
    return distances

# ---------------------------------------------------------------
#
# Compute the core number (the largest value k of a k-core
# containing that node) of each node.
# @param: G - NetworkX graph
# @return: k_core - dictionary keyed by node which values are the
#                   core numbers
# ---------------------------------------------------------------
def compute_k_core_values(G):
    start_time = time.time()
    k_core = nx.core_number(G)
    logger.info("Computed K-Core values in %s seconds" % (time.time() - start_time))
    return k_core

# ---------------------------------------------------------------
#
# An other method to compute k_shell values.
# ---------------------------------------------------------------

def check(h, d):
    f = 0
    for i in h.nodes():
        if (h.degree(i) <= d):
            f = 1
            break
    return f


def find_nodes(h, it):
    s = []
    for i in h.nodes():
        if (h.degree(i) <= it):
            s.append(i)
    return s


def kShell_values(h):
    start = time.time()
    it = 1
    tmp = []
    buckets = []

    while (1):
        flag = check(h, it)
        if (flag == 0):
            it += 1
            buckets.append(tmp)
            tmp = []
        if (flag == 1):
            node_set = find_nodes(h, it)
            for each in node_set:
                h.remove_node(each)
                tmp.append(each)
        if (h.number_of_nodes() == 0):
            buckets.append(tmp)
            break

    core_values = dict()

    value = 1

    for b in buckets:
        for n in b:
            core_values[n] = value
        value += 1
    logger.info(("kShell in %s seconds") % (time.time()-start))
    return core_values


def asp_s(H, G):
    n = H.number_of_nodes()
    asp_value = 0
    if not (nx.is_connected(H)):
        diameter = nx.diameter(G)
    for i in H:
        for j in H:
            if nx.has_path(H, source=i, target=j):
                asp_value += nx.shortest_path_length(H, source=i, target=j)
            else:
                asp_value += diameter
    asp_value = asp_value / (n * (n-1))
    return asp_value

# ---------------------------------------------------------------
#
# Algorithms for finding ASP for a graph H
# @param: H - graph
# @return: asp_value - average shortest path of graph H
# ---------------------------------------------------------------

def asp(H):
    n = H.number_of_nodes()
    asp_value = 0
    if not (nx.is_connected(H)):
        diameter = nx.diameter(H)
    for i in H:
        for j in H:
            if nx.has_path(H, source=i, target=j):
                asp_value += nx.shortest_path_length(H, source=i, target=j)
            else:
                asp_value += diameter
    asp_value = asp_value / (n * (n-1))
    return asp_value

# ---------------------------------------------------------------
#
# LRASP (Local Relative change of Average Shortest Path
# Centrality) algorithm to obtain the k nodes with higher
# LRASP measures.
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def local_rasp(args):
    G = load_default_graph(args.input)
    lrasp = dict.fromkeys(list(G.nodes), 0)
    for i in G:
        neighbors = neighborhood_including_node(G, i, level=2)
        H = G.subgraph(neighbors.keys())
        if H is not None:
            H_i = H.copy()
            H_i.remove_node(i)
            if H_i is not None:
                asp_value_H = asp(H)
                lrasp[i] = abs(asp_s(H_i, H) - asp_value_H) / asp_value_H
    lrasp = dict(sorted(lrasp.items(), key=operator.itemgetter(1), reverse=True))
    print(lrasp)
    k = args.top
    return list(lrasp.keys())[:k]


# ---------------------------------------------------------------
#
# NLC index algorithm to obtain the influence of nodes
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def nlc(args):
    G, deepwalk_graph = deepwalk_embedding(args.input)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    core_values = compute_k_core_values(G)
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    for i in G:
        neighbors = neighborhood(G, i, level=3)
        for j in neighbors:
            nlc_indexes[i] += (core_values[i] * math.exp(- np.linalg.norm(deepwalk_graph[i] - deepwalk_graph[j])))
    # sorting in descending order the nodes based on their NLC index  values
    nlc_indexes = dict(sorted(nlc_indexes.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    return list(nlc_indexes.keys())[:k]


# ---------------------------------------------------------------
#
# NLC index algorithm to obtain the influence of nodes
# but with NetMF embedding
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def nlc2(args):
    G = load_graph(args.input)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.input)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    for i in G:
        neighbors = neighborhood(G, i, level=3)
        for j in neighbors:
            nlc_indexes[i] += (core_values[i] * math.exp(distances[i,j]))
    # sorting in descending order the nodes based on their NLC index  values
    nlc_indexes = dict(sorted(nlc_indexes.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    return list(nlc_indexes.keys())[:k]


# ---------------------------------------------------------------
#
# NLC modified second algorithm to obtain the influence of nodes,
# considering the degrees of nodes.
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def nlc_modified_second(args):
    G = load_graph(args.input)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.input)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    # Computing degree for each node
    start_time4 = time.time()
    degrees = {node: val for (node, val) in G.degree()}
    logger.info("Computing degree of each node in %s seconds" % (time.time() - start_time4))
    # Computing the NLC index of each node
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    logger.info("Computing NLC index of each node")
    for i in G:
        neighbors = neighborhood(G, i, level=3)
        for j in neighbors:
            nlc_indexes[i] += (core_values[i] * math.exp(distances[i,j]))
    # returning top 10 influential nodes
    result = dict.fromkeys(nlc_indexes.keys(), 0)
    for node in result:
        result[node] = nlc_indexes[node] * degrees[node]
    result = dict(sorted(result.items(), key=operator.itemgetter(1), reverse=False))
    k = args.top
    return list(result.keys())[:k]


# ---------------------------------------------------------------
#
# NLC modified third algorithm to obtain the influence of nodes,
# considering the ratio between the node's degree and the total
# degree of neighborhood of each node.
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def nlc_modified_third(args):
    G = load_graph(args.input)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.input)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    # Computing degree for each node
    start_time4 = time.time()
    degrees = {node: val for (node, val) in G.degree()}
    logger.info("Computing degree of each node in %s seconds" % (time.time() - start_time4))
    # Computing the NLC index of each node
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    neighborhood_degree = dict.copy(nlc_indexes)
    influence = dict.copy(nlc_indexes)
    logger.info("Computing NLC index and degree ratio of each node")
    for i in G:
        neighbors = neighborhood(G, i, level=3)
        neighborhood_degree[i] += degrees[i]
        for j in neighbors:
            nlc_indexes[i] += (core_values[i] * math.exp(distances[i, j]))
            neighborhood_degree[i] += degrees[j]
        influence[i] = degrees[i] / neighborhood_degree[i]
    # returning top 10 influential nodes
    result = dict.fromkeys(nlc_indexes.keys(), 0)
    for node in result:
        result[node] = nlc_indexes[node] * influence[node]
    result = dict(sorted(result.items(), key=operator.itemgetter(1), reverse=False))
    k = args.top
    return list(result.keys())[:k]


def nlc_triangle(args):
    G = load_graph(args.input)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.input)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    triangles_pernode = dict.fromkeys(core_values.keys(), 0)
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    for i in G:
        neighbors = neighborhood_including_node(G, i, level=3)
        triangles_pernode[i] += sum(nx.triangles(nx.subgraph(G, neighbors))) / 3
        del neighbors[i]
        for j in neighbors:
            nlc_indexes[i] += (core_values[i] * triangles_pernode[i] * math.exp(distances[i, j]))
    # sorting in descending order the nodes based on their NLC index  values
    nlc_indexes = dict(sorted(nlc_indexes.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    return list(nlc_indexes.keys())[:k]


def nlc_ksd(args):
    G = load_graph(args.input)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.input)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    average_core_number = statistics.mean(list(core_values.values()))
    # Computing degree for each node
    degrees = {node: val for (node, val) in G.degree()}
    average_degree = statistics.mean(list(degrees.values()))
    ratio = average_core_number / average_degree
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    for i in G:
        neighbors = neighborhood(G, i, level=2)
        for j in neighbors:
            nlc_indexes[i] += ((core_values[i] + core_values[j]) + ratio * (degrees[i] + degrees[j])) * math.exp(distances[i, j])
    # sorting in descending order the nodes based on their NLC index  values
    nlc_indexes = dict(sorted(nlc_indexes.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    return list(nlc_indexes.keys())[:k]


def bf_num_spaths(G):
    n_spaths = dict.fromkeys(G, 0.0)
    for source in G:
        for target in G:
            if source == target:
                continue
            for path in nx.all_shortest_paths(G, source, target):
                for node in path[1:]: # ignore firs element (source == node)
                    n_spaths[node] += 1 # this path passes through `node`
    return n_spaths


def nlc_shks(args):
    G = load_graph(args.input)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.input)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    # Computing degree for each node
    degrees = {node: val for (node, val) in G.degree()}
    # Computing average node stress
    S = statistics.mean(list(bf_num_spaths(G).values()))
    alfa = 0.0000154 * S + 22.34
    c = dict.fromkeys(core_values.keys(), 0)
    sh = dict.fromkeys(core_values.keys(), 0)
    I = dict.fromkeys(core_values.keys(), 0)
    C = dict.fromkeys(core_values.keys(), 0)
    IS = dict.fromkeys(core_values.keys(), 0)
    SHKS = dict.fromkeys(core_values.keys(), 0)
    for v in G:
        neighbors = neighborhood(G, v, level=1)
        for u in neighbors:
            pvu = 1 / (len(neighbors.keys()) )
            neighbors2 = neighborhood(G, u, level=1)
            intersection = {i: neighbors[i] for i in set(neighbors.keys()).intersection(set(neighbors2.keys()))}
            second_factor = 0
            for w in intersection:
                neighbors3 = neighborhood(G, w, level=1)
                pvw = 1 / (len(neighbors.keys()))
                pwu = 1 / (len(neighbors3.keys()))
                second_factor += pvw * pwu
            term = math.pow(pvu + second_factor, 2)
            c[v] += term
        sh[v] = 1 / c[v] #sh
        I[v] = alfa * sh[v] + core_values[v]

    for v in G:
        neighbors = neighborhood(G, v, level=1)
        for u in neighbors:
            C[v] += (I[v] + I[u])

    for v in G:
        neighbors = neighborhood(G, v, level=1)
        for u in neighbors:
            IS[v] += C[u]

    for v in G:
        neighbors = neighborhood(G, v, level=1)
        for u in neighbors:
            SHKS[v] += IS[u] * math.exp(distances[v,u])

    # sorting in descending order the nodes based on their NLC index  values
    SHKS = dict(sorted(SHKS.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    return list(SHKS.keys())[:k]


# ---------------------------------------------------------------
#
# KDEC algorithm to obtain the k top nodes in the
# ranking obtained.
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def kdec(args):
    G = load_graph(args.input)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    # Computing degrees
    degrees = {node: deg for (node, deg) in G.degree()}
    # Computing weights
    weights = {key: core_values[key] * degrees[key] for key in core_values}
    # KDEC structure
    kdec = dict.fromkeys(core_values.keys(), 0.0);
    for i in G:
        neighbors = neighborhood(G, i, level=1)
        for j in neighbors:
            eff_ij = (1 - math.log(1 / degrees[j]))
            kdec[i] += ((weights[i] * weights[j]) / (math.pow(eff_ij, 2)))
    # sorting in descending order the nodes based on their values
    kdec = dict(sorted(kdec.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    return list(kdec.keys())[:k]

# ---------------------------------------------------------------
#
# KDEC algorithm version which considers the embedding nodes too.
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def nlc_kdec(args):
    G = load_graph(args.input)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.input)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    # Computing degrees
    degrees = {node: deg for (node, deg) in G.degree()}
    # Computing weights
    weights = {key: core_values[key] * degrees[key] for key in core_values}
    # print(weights)
    # KDEC structure
    nlckdec = dict.fromkeys(core_values.keys(), 0.0)
    for i in G:
        neighbors = neighborhood(G, i, level=1)
        for j in neighbors:
            eff_ij = (1 - math.log(1 / degrees[j]))
            nlckdec[i] += ((weights[i] * weights[j]) / (math.pow(eff_ij, 2))) * math.exp(distances[i,j])

    # sorting in descending order the nodes based on their values
    nlckdec = dict(sorted(nlckdec.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    return list(nlckdec.keys())[:k]


# def gravity_model_modified(args):
#     G = load_graph(args.input)
#     nodes = G.number_of_nodes()
#     # Using NetMF embedding from karateclub module
#     logger.info("NetMF embedding of network in %s dataset", args.input)
#     start_time1 = time.time()
#     model = ne.NetMF()
#     model.fit(G)
#     embedding = model.get_embedding()
#     logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
#     G.remove_edges_from(list(nx.selfloop_edges(G)))
#     # Precomputing euclidean distances between nodes in embedding
#     distances = precomputing_euclidean(embedding)
#     # Computing the degree of each node
#     degrees = {node: val for (node, val) in G.degree()}
#     # Computing k-core value of each node
#     core_values = compute_k_core_values(G)
#     ks_max = max(core_values.values())
#     ks_min = min(core_values.values())
#     attraction_coefficient = dict()
#     force_node = dict()
#     # K-SHell based on gravity centrality with NetMF embedding for distances
#     KSGCNetMF = dict.fromkeys(core_values.keys(), 0)
#     for i in G:
#         neighbors = neighborhood(G, i, level=3) # Computing 3rd-order neighborhood of node i
#         for j in neighbors:
#             alias = (i, j)
#             attraction_coefficient[alias] = math.exp((core_values[i] - core_values[j]) / (ks_max - ks_min))
#             force_node[alias] = attraction_coefficient[alias] * ((degrees[i] * degrees[j]) / distances[alias])
#             KSGCNetMF[i] += force_node[alias]
#     # sorting in descending order
#     nlc_indexes = dict(sorted(KSGCNetMF.items(), key=operator.itemgetter(1), reverse=True))
#     # returning top 10 influential nodes
#     k = args.top
#     print(list(KSGCNetMF.keys())[:k])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help=".mat input file path of original network")
    parser.add_argument("--top", default=10, type=int,
                        help="#nodes top nodes")
    parser.add_argument('--nlc', dest="nlc", action="store_true",
                        help="using NLC to compute influence of nodes")
    parser.set_defaults(nlc=False)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s')  # include timestamp
    # if args.nlc:
    #     logger.info("Algorithm NLC starting")
    #     start = time.time()
    #     nlc(args)
    #     logger.info("Algorithm NLC concluded in %s seconds" % (time.time() - start))
    # else:
    #     logger.info("Algorithm NLC modified starting")
    #     start = time.time()
    #     nlc_modified(args)
    #     logger.info("Algorithm NLC modified concluded in %s seconds" % (time.time() - start))
    # else:
    #     logger.info("Algorithm NLC second modified starting")
    #     start = time.time()
    #     nlc_modified_second(args)
    #     logger.info("Algorithm NLC second modified concluded in %s seconds" % (time.time() - start))
    # else:
    #     logger.info("Algorithm NLC third modified starting")
    #     start = time.time()
    #     nlc_modified_third(args)
    #     logger.info("Algorithm NLC third modified concluded in %s seconds" % (time.time() - start))
    # else:
    #      logger.info("Algorithm LRASP starting")
    #      start = time.time()
    #      local_rasp(args)
    #      logger.info("Algorithm LRASP concluded in %s seconds" % (time.time() - start))
    #else:
    #    logger.info("Algorithm gravity model KSGCNetMF starting")
    #    start = time.time()
    #    gravity_model_modified(args)
    #    logger.info("Algorithm gravity model KSGCNetMF concluded in %s seconds" % (time.time() - start))
    # else:
    #     logger.info("Algorithm KDEC starting")
    #     start = time.time()
    #     kdec(args)
    #     logger.info("Algorithm KDEC concluded in %s seconds" % (time.time() - start))
    # else:
    #     logger.info("Algorithm NLC KDEC starting")
    #     start = time.time()
    #     nlc_kdec(args)
    #     logger.info("Algorithm NLC KDEC concluded in %s seconds" % (time.time() - start))
    G = load_graph(args.input)
    length = G.number_of_nodes()
    nlc_counter = dict.fromkeys(list(G.nodes()), 0)
    nlc2_counter = dict.fromkeys(list(G.nodes()), 0)
    nlc_second_mod_counter = dict.fromkeys(list(G.nodes()), 0)
    nlc_third_mod_counter = dict.fromkeys(list(G.nodes()), 0)
    nlc_ksd_counter = dict.fromkeys(list(G.nodes()), 0)
    # for i in range(1000):
    #     nodes = nlc(args)
    #     for j in nodes:
    #         nlc_counter[j] += 1
    # nlc_counter = dict(sorted(nlc_counter.items(), key=operator.itemgetter(1), reverse=True))
    # for i in range(1000):
    #     nodes = nlc2(args)
    #     for j in nodes:
    #         nlc2_counter[j] += 1
    # nlc2_counter = dict(sorted(nlc2_counter.items(), key=operator.itemgetter(1), reverse=True))
    # for i in range(1000):
    #     nodes = nlc_modified_second(args)
    #     for j in nodes:
    #         nlc_second_mod_counter[j] += 1
    # nlc_second_mod_counter = dict(sorted(nlc_second_mod_counter.items(), key=operator.itemgetter(1), reverse=True))
    # for i in range(1000):
    #     nodes = nlc_modified_third(args)
    #     for j in nodes:
    #         nlc_third_mod_counter[j] += 1
    # nlc_third_mod_counter = dict(sorted(nlc_third_mod_counter.items(), key=operator.itemgetter(1), reverse=True))
    for i in range(1000):
        nodes = nlc(args)
        for j in nodes:
            nlc_counter[j] += 1
    nlc_counter = dict(sorted(nlc_counter.items(), key=operator.itemgetter(1), reverse=True))
    # print("NLC")
    # print(list(nlc_counter.keys())[:args.top])
    # print("NLC2")
    # print(list(nlc2_counter.keys())[:args.top])
    # print("NLC second modified")
    # print(list(nlc_second_mod_counter.keys())[:args.top])
    # print("NLC third modified")
    # print(list(nlc_third_mod_counter.keys())[:args.top])
    print("NLC")
    top = list(nlc_counter.keys())[:args.top]
    first = lesmis_top[:args.top]
    count = 0
    for i in top:
        count += first.count(i)
    print(count)

