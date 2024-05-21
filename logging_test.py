import logging

# create an instance of the logger
logger = logging.getLogger()

# logging set up 
log_format = logging.Formatter('%(asctime)-15s %(levelname)-2s %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(log_format)

# add the handler
logger.addHandler(sh)
logger.setLevel(logging.INFO)


def do_something():
    return None

def call_do_something(i):
  # This will obviously throw and exception
    # for i in range(0, 10):
    #     if i == 2:
    #         raise Exception('This is an exception')
    #     print('i', i)
    if i == 2:
        raise KeyError('This is an exception')
    print('i', i)

def what(i):
    print('i', i)
    # call_do_something(i)
    try:
        call_do_something(i)
    except KeyError as e:
        raise AttributeError('An error occurred while calling the function call_do_something()') from e
        # raise Exception('An error occurred while calling the function call_do_something()') from e
# logging exception with logger.error()
# try:
#     p = [1, 2, 3]
#     for i in p:
#         try:
#             what(i)
#         except KeyError as e:
#             logger.exception(e)
#             logger.error('WHATTTT')
#             # continue
# except AttributeError as e:
#     logger.exception(e)
#     logger.error('An error occurred while calling the function call_do_something()')
#     # raise Exception('An error occurred while calling the function call_do_something()') from e

# call_do_something()
from torch_geometric.datasets import PPI

dataset = PPI(root='/datasets')
data = dataset[0]
print(data)