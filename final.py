from VR_Assignment_Rutul.part1.coinDetection import pipeline

from VR_Assignment_Rutul.part2.panorama import panorama


for i in range(0,1):
    pipeline(f"coins/{i}.jpg", f"output_coin_detection_{i}")
for i in range(1,2):
    panorama(f"input{i}", f"output_panorama_{i}")

