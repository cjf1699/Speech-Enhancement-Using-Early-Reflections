import matplotlib.pyplot as plt
import logging
import numpy as np
plt.switch_backend('agg')
logging.basicConfig(level=logging.DEBUG,
                    filename='./log/'  + 'test.log',
                    filemode='w',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

logger = logging.getLogger("train_logger")
logger.info("This is not a bug")
a = np.array([1,2,3,4])
fig, ax = plt.subplots(1, 1)
ax.plot(a)
