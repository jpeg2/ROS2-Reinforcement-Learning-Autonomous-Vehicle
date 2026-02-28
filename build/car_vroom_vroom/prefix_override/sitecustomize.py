import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/contranickted/sim_ws/src/car_vroom_vroom/install/car_vroom_vroom'
