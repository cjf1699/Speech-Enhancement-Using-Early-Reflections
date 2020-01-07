import logging
from handler import get_az
rir_dict = {#1.9572: '/home/cjf/workspace/Matlab/RirsOfRooms/RT60_0.583/dist_1.9572/'
            #1.6555: '/home/cjf/workspace/Matlab/RirsOfRooms/RT60_0.47149/dist_1.6555/'
            1.6948: '/home/cjf/workspace/Matlab/RirsOfRooms/RT60_0.31077/dist_1.6948/'
            }
logger = logging.getLogger("test_get_az")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='./test_get_az_1.70.log', mode='w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

for dist in rir_dict:
    logger.info("dist:{}".format(dist))
    for az_idx in range(1, 73):
        read_path = rir_dict[dist] + 'source_{}.binary'.format(az_idx)
        azs = get_az(read_path)
        logger.info("direct:{}, reflections:{}".format((az_idx-1)*5, azs))
        
